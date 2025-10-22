# Understanding `masked_normalize` Function

## Overview

The `masked_normalize` function is a **critical utility** for computing loss in SFT (Supervised Fine-Tuning). It computes a **masked sum** and normalizes it by a constant, which allows us to:

1. ✅ Sum values only where a mask is True (value = 1)
2. ✅ Ignore masked-out positions (value = 0)
3. ✅ Normalize by a constant (e.g., number of tokens, gradient accumulation steps)

## Why Do We Need This?

### The Problem in SFT

In Supervised Fine-Tuning, we concatenate prompt and output into a single sequence:

```
Sequence: [P1, P2, P3, O1, O2, O3, PAD, PAD]
           ^-prompt-^  ^-output-^ ^-padding-^
```

**But we only want to compute loss on the OUTPUT tokens!**

- ❌ Don't compute loss on prompt tokens (model shouldn't learn to "predict" the prompt)
- ❌ Don't compute loss on padding tokens (they're meaningless)
- ✅ Only compute loss on output tokens (what we're training the model to generate)

### The Solution: Masking

We create a **response_mask** that marks which positions to include:

```
Sequence:      [P1,  P2,  P3,  O1,  O2,  O3,  PAD, PAD]
response_mask: [ 0,   0,   0,   1,   1,   1,   0,   0]
                 ^-ignore-^    ^-include-^   ^-ignore-^
```

Then we use `masked_normalize` to:
1. Sum log-probabilities only where mask = 1
2. Normalize by the number of output tokens (3 in this case)

---

## Function Signature

```python
def masked_normalize(
    tensor: torch.Tensor,        # The values to sum
    mask: torch.Tensor,           # Which positions to include (1) or ignore (0)
    dim: int | None = None,       # Dimension to sum along (or None for all)
    normalize_constant: float = 1.0,  # Divide sum by this value
) -> torch.Tensor:
```

---

## How It Works: Step-by-Step

### Step 1: Apply the Mask

```python
masked_tensor = tensor * mask.float()
```

**What this does:**
- Multiplies each element by its corresponding mask value
- Positions with mask=1 keep their original value
- Positions with mask=0 become 0

**Example:**

```python
tensor = [[1.0, 2.0, 3.0],    mask = [[1, 0, 1],
          [4.0, 5.0, 6.0]]             [1, 1, 0]]

masked_tensor = [[1.0, 0.0, 3.0],
                 [4.0, 5.0, 0.0]]
```

**Why multiply instead of indexing?**
- ✅ Keeps the tensor shape intact
- ✅ Allows dimension-specific operations
- ✅ More efficient than creating new tensors with indexing

### Step 2: Sum Along Dimension

```python
masked_sum = torch.sum(masked_tensor, dim=dim)
```

The `dim` parameter controls **which dimension to sum over**:

| `dim` value | Behavior | Example Result |
|-------------|----------|----------------|
| `None` | Sum **all** elements → scalar | `13.0` |
| `0` | Sum along **rows** (collapse batch dim) | `[5.0, 5.0, 3.0]` |
| `1` | Sum along **columns** (collapse seq dim) | `[4.0, 9.0]` |
| `-1` | Sum along **last dimension** | `[4.0, 9.0]` |

**Continuing our example:**

```python
masked_tensor = [[1.0, 0.0, 3.0],
                 [4.0, 5.0, 0.0]]

# dim=None: sum all elements
masked_sum = 1.0 + 0.0 + 3.0 + 4.0 + 5.0 + 0.0 = 13.0

# dim=0: sum each column
masked_sum = [1.0+4.0, 0.0+5.0, 3.0+0.0] = [5.0, 5.0, 3.0]

# dim=1: sum each row
masked_sum = [1.0+0.0+3.0, 4.0+5.0+0.0] = [4.0, 9.0]
```

### Step 3: Normalize by Constant

```python
return masked_sum / normalize_constant
```

**What is `normalize_constant`?**

It depends on the use case:

| Use Case | `normalize_constant` | Purpose |
|----------|---------------------|---------|
| **Average per-token loss** | Number of output tokens | Makes loss independent of sequence length |
| **Average per-example loss** | Batch size | Makes loss independent of batch size |
| **Gradient accumulation** | Accumulation steps | Scales gradients correctly |

**Example:**

```python
masked_sum = 13.0
normalize_constant = 2.0

result = 13.0 / 2.0 = 6.5
```

---

## Complete Example: SFT Loss Computation

Let's see how `masked_normalize` is used to compute **SFT loss**:

### Setup

```python
# Log-probabilities from the model (negative, since they're logs of probs < 1)
log_probs = [[-0.5, -0.3, -0.8, -0.2, -0.1],
             [-0.4, -0.6, -0.3, -0.5, -0.2]]

# Response mask: which tokens are part of the output?
response_mask = [[0, 0, 1, 1, 1],  # First 2 are prompt, last 3 are output
                 [0, 1, 1, 1, 0]]  # First 1 is prompt, middle 3 are output, last 1 is padding

# Total number of output tokens
num_output_tokens = response_mask.sum() = 6
```

### Compute Masked Sum

```python
masked_tensor = log_probs * response_mask.float()
# Result: [[0.0, 0.0, -0.8, -0.2, -0.1],
#          [0.0, -0.6, -0.3, -0.5, 0.0]]
```

### Compute Average Log-Probability

```python
avg_log_prob = masked_normalize(
    tensor=log_probs,
    mask=response_mask,
    dim=None,  # Sum all elements
    normalize_constant=num_output_tokens  # = 6
)
# Result: (-0.8 - 0.2 - 0.1 - 0.6 - 0.3 - 0.5) / 6 = -2.5 / 6 = -0.4167
```

### Compute Loss

```python
loss = -avg_log_prob = 0.4167
```

**Why negate?**
- Log-probabilities are negative (log of values < 1)
- Higher probabilities → more negative log-probs → better model
- We want loss to be positive and decrease during training
- Negating makes loss positive and aligns with optimization (minimize loss)

---

## Dimension Examples

Let's explore different `dim` values with a concrete example:

### Example Setup

```python
tensor = [[1.0, 2.0, 3.0],
          [4.0, 5.0, 6.0]]

mask = [[1, 0, 1],
        [1, 1, 0]]

normalize_constant = 2.0
```

### `dim=None`: Sum All Elements

```python
result = masked_normalize(tensor, mask, dim=None, normalize_constant=2.0)
# Step 1: masked_tensor = [[1.0, 0.0, 3.0], [4.0, 5.0, 0.0]]
# Step 2: masked_sum = 1.0 + 3.0 + 4.0 + 5.0 = 13.0
# Step 3: result = 13.0 / 2.0 = 6.5
print(result)  # tensor(6.5)
```

**Shape:** scalar (0D tensor)

### `dim=0`: Sum Along Rows (Collapse Batch Dimension)

```python
result = masked_normalize(tensor, mask, dim=0, normalize_constant=2.0)
# Step 1: masked_tensor = [[1.0, 0.0, 3.0], [4.0, 5.0, 0.0]]
# Step 2: masked_sum = [1.0+4.0, 0.0+5.0, 3.0+0.0] = [5.0, 5.0, 3.0]
# Step 3: result = [5.0/2.0, 5.0/2.0, 3.0/2.0] = [2.5, 2.5, 1.5]
print(result)  # tensor([2.5, 2.5, 1.5])
```

**Shape:** `(3,)` - one value per column

**Interpretation:** Average contribution from each position across the batch

### `dim=1`: Sum Along Columns (Collapse Sequence Dimension)

```python
result = masked_normalize(tensor, mask, dim=1, normalize_constant=2.0)
# Step 1: masked_tensor = [[1.0, 0.0, 3.0], [4.0, 5.0, 0.0]]
# Step 2: masked_sum = [1.0+0.0+3.0, 4.0+5.0+0.0] = [4.0, 9.0]
# Step 3: result = [4.0/2.0, 9.0/2.0] = [2.0, 4.5]
print(result)  # tensor([2.0, 4.5])
```

**Shape:** `(2,)` - one value per row (batch item)

**Interpretation:** Average value for each example in the batch

### `dim=-1`: Sum Along Last Dimension (Same as `dim=1` here)

```python
result = masked_normalize(tensor, mask, dim=-1, normalize_constant=2.0)
# Same as dim=1 for 2D tensor
print(result)  # tensor([2.0, 4.5])
```

---

## Key Insights

### 1. Why Multiply by Mask Instead of Indexing?

❌ **Bad approach (indexing):**
```python
# This changes the shape and loses position information!
masked_values = tensor[mask]  # Flattened 1D array
```

✅ **Good approach (multiplication):**
```python
# This keeps the shape intact, just zeros out masked positions
masked_tensor = tensor * mask.float()  # Same shape as original
```

### 2. Why Convert Mask to Float?

```python
mask.float()  # or mask.to(tensor.dtype)
```

- Masks are typically **boolean** (`True`/`False`) or **integer** (`0`/`1`)
- Tensors with values are typically **float**
- PyTorch requires matching dtypes for element-wise operations
- Converting to float allows smooth multiplication

### 3. When to Use Which `dim`?

| Use Case | `dim` value | Why? |
|----------|-------------|------|
| **Total loss for batch** | `None` | Sum all output tokens across all examples |
| **Per-position statistics** | `0` | Compare how different positions contribute |
| **Per-example loss** | `1` or `-1` | Separate loss for each example in batch |

### 4. Connection to SFT Loss

The standard **SFT loss** (negative log-likelihood) is:

```python
loss = -masked_normalize(
    tensor=log_probs,              # Log-probabilities from model
    mask=response_mask,            # Which tokens are output (not prompt/padding)
    dim=None,                      # Sum all output tokens
    normalize_constant=response_mask.sum()  # Number of output tokens
)
```

**This computes:**
- ✅ Average log-probability of output tokens
- ✅ Ignores prompt and padding tokens
- ✅ Normalizes by sequence length (fair comparison)
- ✅ Negates to get positive loss (lower is better)

---

## Mathematical Formulation

### General Form

$$
\text{masked\_normalize}(\mathbf{T}, \mathbf{M}, d, c) = \frac{\sum_{i \in d} (\mathbf{T} \odot \mathbf{M})_i}{c}
$$

Where:
- $\mathbf{T}$ = tensor (values to sum)
- $\mathbf{M}$ = mask (which elements to include)
- $\odot$ = element-wise multiplication (Hadamard product)
- $d$ = dimension to sum along (or all dimensions if `None`)
- $c$ = normalize_constant

### SFT Loss Form

$$
\mathcal{L}_{\text{SFT}} = -\frac{1}{|\mathcal{O}|} \sum_{t \in \mathcal{O}} \log P(y_t | y_{<t}, x)
$$

Where:
- $\mathcal{O}$ = set of output token positions (where mask = 1)
- $|\mathcal{O}|$ = number of output tokens (normalize_constant)
- $x$ = prompt
- $y_t$ = token at position $t$

**Implementation:**
```python
loss = -masked_normalize(
    tensor=log_probs,                    # log P(y_t | y_{<t}, x)
    mask=response_mask,                  # indicator for t ∈ O
    dim=None,                            # sum over all t
    normalize_constant=response_mask.sum()  # |O|
)
```

---

## Common Pitfalls and Tips

### ⚠️ Pitfall 1: Forgetting to Convert Mask to Float

```python
# ❌ This might cause dtype mismatch errors
masked_tensor = tensor * mask

# ✅ Always convert mask to match tensor dtype
masked_tensor = tensor * mask.float()
```

### ⚠️ Pitfall 2: Wrong Normalize Constant

```python
# ❌ Normalizing by total sequence length (includes prompt/padding!)
normalize_constant = tensor.shape[1]

# ✅ Normalize by actual output tokens
normalize_constant = response_mask.sum()
```

### ⚠️ Pitfall 3: Not Understanding Dimensions

```python
# For batch of log_probs: [batch_size, seq_len]

# dim=None → scalar (total average)
# dim=0 → [seq_len] (per-position average across batch)
# dim=1 → [batch_size] (per-example average)

# Choose based on what you need!
```

### 💡 Tip 1: Verify Your Mask

Always check that your mask sums to the expected number:

```python
print(f"Number of output tokens: {response_mask.sum().item()}")
# Should match your expectation!
```

### 💡 Tip 2: Combine with torch.no_grad() for Evaluation

```python
with torch.no_grad():
    eval_loss = -masked_normalize(
        log_probs, response_mask, dim=None, 
        normalize_constant=response_mask.sum()
    )
```

### 💡 Tip 3: Use Gradient Accumulation Constant

For gradient accumulation across multiple microbatches:

```python
normalize_constant = num_output_tokens * gradient_accumulation_steps
```

This ensures gradients are scaled correctly when accumulating.

---

## Summary

**`masked_normalize` is a simple but powerful utility that:**

1. ✅ **Applies a mask** to zero out unwanted elements
2. ✅ **Sums** along a specified dimension (or all dimensions)
3. ✅ **Normalizes** by a constant (e.g., sequence length, batch size)

**It's essential for SFT because:**
- Allows computing loss only on output tokens
- Ignores prompt and padding tokens
- Normalizes for fair comparison across different sequence lengths

**Key equation for SFT loss:**
```python
loss = -masked_normalize(log_probs, response_mask, dim=None, 
                         normalize_constant=response_mask.sum())
```

This computes the **negative average log-likelihood** of the output tokens, which is exactly what we want to minimize during supervised fine-tuning! 🎯
