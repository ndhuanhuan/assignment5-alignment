# `masked_normalize` Visual Cheatsheet 🎯

## Quick Reference

```python
masked_normalize(tensor, mask, dim=None, normalize_constant=1.0)
```

**Returns:** Sum of masked elements, divided by constant

---

## Visual Flow

```
┌──────────────────────────────────────────────────────────────┐
│                    Input: tensor & mask                       │
│                                                               │
│   tensor:  [ 1.0,  2.0,  3.0,  4.0,  5.0 ]                  │
│   mask:    [   1,    0,    1,    1,    0 ]                  │
└──────────────────────────────────────────────────────────────┘
                            ↓
┌──────────────────────────────────────────────────────────────┐
│              Step 1: Apply Mask (multiply)                    │
│                                                               │
│   tensor * mask.float()                                       │
│   = [ 1.0,  0.0,  3.0,  4.0,  0.0 ]                         │
│       ✓    ✗     ✓     ✓     ✗                              │
└──────────────────────────────────────────────────────────────┘
                            ↓
┌──────────────────────────────────────────────────────────────┐
│               Step 2: Sum (along dimension)                   │
│                                                               │
│   torch.sum(masked_tensor, dim=dim)                          │
│   = 1.0 + 3.0 + 4.0 = 8.0                                    │
└──────────────────────────────────────────────────────────────┘
                            ↓
┌──────────────────────────────────────────────────────────────┐
│        Step 3: Normalize (divide by constant)                │
│                                                               │
│   masked_sum / normalize_constant                            │
│   = 8.0 / 2.0 = 4.0                                          │
└──────────────────────────────────────────────────────────────┘
                            ↓
                    Result: 4.0
```

---

## Dimension Visualization (2D Example)

### Setup

```python
tensor = [[1, 2, 3],      mask = [[1, 0, 1],
          [4, 5, 6]]               [1, 1, 0]]

normalize_constant = 2.0
```

### Step 1: Apply Mask

```
tensor:        mask:        masked_tensor:
┌─────────┐   ┌─────────┐   ┌─────────┐
│ 1  2  3 │ × │ 1  0  1 │ = │ 1  0  3 │
│ 4  5  6 │   │ 1  1  0 │   │ 4  5  0 │
└─────────┘   └─────────┘   └─────────┘
```

### dim=None: Sum All → Scalar

```
┌─────────┐
│ 1  0  3 │  ──→  1 + 0 + 3 + 4 + 5 + 0 = 13.0
│ 4  5  0 │
└─────────┘       13.0 / 2.0 = 6.5 ✓

Result: scalar (6.5)
```

### dim=0: Sum Rows → Column Vector

```
┌─────────┐
│ 1  0  3 │
│ 4  5  0 │
└─────────┘
  ↓  ↓  ↓
 [5][5][3]  ──→  [5.0, 5.0, 3.0] / 2.0
                 = [2.5, 2.5, 1.5] ✓

Result: shape (3,)
```

### dim=1: Sum Columns → Row Vector

```
         ┌─────────┐
         │ 1  0  3 │ ──→  4
         │ 4  5  0 │ ──→  9
         └─────────┘

         [4.0, 9.0] / 2.0 = [2.0, 4.5] ✓

Result: shape (2,)
```

---

## SFT Loss Computation

### Visual Example

```
Sequence:    [ P1   P2   O1   O2   O3  PAD ]
             ┌────┬────┬────┬────┬────┬────┐
log_probs:   │-0.5│-0.3│-0.8│-0.2│-0.1│-0.9│
             └────┴────┴────┴────┴────┴────┘
                         ↓
             ┌────┬────┬────┬────┬────┬────┐
mask:        │  0 │  0 │  1 │  1 │  1 │  0 │
             └────┴────┴────┴────┴────┴────┘
             prompt  │←─ output ─→│ pad
                         ↓
             ┌────┬────┬────┬────┬────┬────┐
masked:      │ 0.0│ 0.0│-0.8│-0.2│-0.1│ 0.0│
             └────┴────┴────┴────┴────┴────┘
                         ↓
             sum = -0.8 + (-0.2) + (-0.1) = -1.1
                         ↓
             normalized = -1.1 / 3 = -0.367
                         ↓
             loss = -(-0.367) = 0.367 ✓
```

**Code:**
```python
avg_log_prob = masked_normalize(
    log_probs, mask, dim=None, normalize_constant=3.0
)
loss = -avg_log_prob  # 0.367
```

---

## Common Patterns

### Pattern 1: Total Average Loss

```python
# Average loss across entire batch
loss = -masked_normalize(
    tensor=log_probs,
    mask=response_mask,
    dim=None,  # ← Sum ALL elements
    normalize_constant=response_mask.sum()
)
```

### Pattern 2: Per-Example Loss

```python
# Loss for each example in batch
per_example_loss = -masked_normalize(
    tensor=log_probs,
    mask=response_mask,
    dim=1,  # ← Sum each row
    normalize_constant=response_mask.sum(dim=1)
)
# Shape: [batch_size]
```

### Pattern 3: Per-Token Statistics

```python
# Average contribution of each token position
per_token_avg = masked_normalize(
    tensor=log_probs,
    mask=response_mask,
    dim=0,  # ← Sum each column
    normalize_constant=response_mask.sum(dim=0).clamp(min=1)
)
# Shape: [seq_len]
```

---

## Quick Lookup: Dimension Behavior

| Input Shape | `dim` | Output Shape | Meaning |
|-------------|-------|--------------|---------|
| `(B, S)` | `None` | `()` | Total average across all |
| `(B, S)` | `0` | `(S,)` | Per-token average |
| `(B, S)` | `1` or `-1` | `(B,)` | Per-example average |
| `(B, S, V)` | `None` | `()` | Total average across all |
| `(B, S, V)` | `0` | `(S, V)` | Per-batch-item average |
| `(B, S, V)` | `1` | `(B, V)` | Per-sequence average |
| `(B, S, V)` | `2` or `-1` | `(B, S)` | Per-vocab average |

---

## Shape Transformations

### Example: (2, 5) Tensor

```
Input:  ┌─────────────────┐
        │ • • • • • │  5   Shape: (2, 5)
        │ • • • • • │  5
        └─────────────────┘
           2   2   2   2   2

dim=None:   • → scalar

dim=0:      ┌───────────────┐
            │ • • • • • │     Shape: (5,)
            └───────────────┘

dim=1:      ┌─┐
            │•│  Shape: (2,)
            │•│
            └─┘
```

---

## Code Templates

### Template 1: Basic Usage

```python
import torch
from cs336_alignment.sft import masked_normalize

# Your data
tensor = torch.randn(batch_size, seq_len)
mask = torch.randint(0, 2, (batch_size, seq_len))
constant = mask.sum()

# Compute masked average
result = masked_normalize(tensor, mask, dim=None, 
                          normalize_constant=constant)
```

### Template 2: SFT Loss

```python
# Inside training loop
log_probs = get_response_log_probs(
    model, input_ids, labels
)['log_probs']

# Compute loss only on output tokens
num_tokens = response_mask.sum()
loss = -masked_normalize(
    log_probs, response_mask, dim=None, 
    normalize_constant=num_tokens
)

# Backward pass
loss.backward()
```

### Template 3: Gradient Accumulation

```python
# With gradient accumulation
num_tokens = response_mask.sum()
accumulation_steps = 4

loss = -masked_normalize(
    log_probs, 
    response_mask, 
    dim=None, 
    normalize_constant=num_tokens * accumulation_steps
)

# Accumulate gradients
loss.backward()

if step % accumulation_steps == 0:
    optimizer.step()
    optimizer.zero_grad()
```

---

## Debug Checklist

When things don't work, check:

- [ ] **Mask dtype**: Should be float for multiplication
  ```python
  mask = mask.float()  # or mask.bool().float()
  ```

- [ ] **Normalize constant**: Should match number of True values
  ```python
  assert normalize_constant == mask.sum().item()
  ```

- [ ] **Dimension**: Correct for your use case
  - `None` for total average
  - `0` for per-column
  - `1` or `-1` for per-row

- [ ] **Shape compatibility**: tensor and mask must match
  ```python
  assert tensor.shape == mask.shape
  ```

- [ ] **Zero division**: Constant should be > 0
  ```python
  normalize_constant = max(mask.sum(), 1e-8)
  ```

---

## Performance Tips

### ✅ Do:
```python
# Reuse mask computation
mask_sum = mask.sum()
result = masked_normalize(tensor, mask, dim=None, 
                          normalize_constant=mask_sum)
```

### ✅ Do:
```python
# Convert mask once
mask_float = mask.float()
result = tensor * mask_float
```

### ❌ Avoid:
```python
# Don't convert inside loop repeatedly
for i in range(1000):
    result = tensor * mask.float()  # Converts every iteration!
```

---

## When to Use What

| Scenario | `dim` | `normalize_constant` |
|----------|-------|---------------------|
| **SFT Loss** | `None` | `mask.sum()` |
| **Per-example metrics** | `1` | `mask.sum(dim=1)` |
| **Token-level analysis** | `0` | `mask.sum(dim=0)` |
| **Gradient accumulation** | `None` | `mask.sum() * grad_accum_steps` |
| **Weighted average** | depends | Custom weight sum |

---

## Mathematical Summary

$$
\text{masked\_normalize}(\mathbf{T}, \mathbf{M}, d, c) = \frac{\sum_{i \in d} (\mathbf{T} \odot \mathbf{M})_i}{c}
$$

Where:
- $\mathbf{T}$ = tensor
- $\mathbf{M}$ = mask
- $\odot$ = element-wise multiplication
- $d$ = dimension
- $c$ = normalize_constant

---

## Remember

🔑 **Key insight:** `masked_normalize` = masked sum ÷ constant

💡 **SFT use case:** Average log-prob of output tokens only

⚡ **Performance:** Convert mask to float once, reuse

🎯 **Debugging:** Check mask sum == expected count
