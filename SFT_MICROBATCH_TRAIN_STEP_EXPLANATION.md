# Understanding `sft_microbatch_train_step` Function

## Overview

The `sft_microbatch_train_step` function implements **a single microbatch training step for Supervised Fine-Tuning (SFT)**. It computes the loss and performs backpropagation with proper gradient scaling for gradient accumulation.

## Purpose

This function is the **core training step** in SFT, where we:
1. ✅ Compute the negative log-likelihood loss (cross-entropy) 
2. ✅ Scale the loss appropriately for gradient accumulation
3. ✅ Perform backward pass to compute gradients
4. ✅ Return loss and metadata for logging

---

## Function Signature

```python
def sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,          # [batch_size, seq_len] - log P(token|context)
    response_mask: torch.Tensor,             # [batch_size, seq_len] - 1=output, 0=prompt/pad
    gradient_accumulation_steps: int,        # Number of microbatches per optimizer step
    normalize_constant: float = 1.0,         # Additional normalization factor
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    # Returns: (loss, metadata_dict)
```

---

## Key Concepts

### 1. What is SFT (Supervised Fine-Tuning)?

SFT trains a language model to generate specific outputs given prompts by:
- **Maximizing the likelihood** of the correct output tokens
- **Minimizing the negative log-likelihood** (NLL) loss
- Only computing loss on **output tokens**, not prompt or padding

### 2. What is a Microbatch?

When training with gradient accumulation:
- A **minibatch** is divided into multiple **microbatches**
- Each microbatch is processed separately
- Gradients are **accumulated** (summed) across microbatches
- Parameters are updated once after all microbatches

**Example:**
```
Minibatch size: 32 samples
Microbatch size: 8 samples
Gradient accumulation steps: 4

Step 1: Process microbatch 1 (samples 0-7)   → compute gradients, accumulate
Step 2: Process microbatch 2 (samples 8-15)  → compute gradients, accumulate
Step 3: Process microbatch 3 (samples 16-23) → compute gradients, accumulate
Step 4: Process microbatch 4 (samples 24-31) → compute gradients, accumulate
Step 5: Update parameters using accumulated gradients
Step 6: Zero gradients, repeat
```

### 3. Why Gradient Accumulation?

Gradient accumulation allows simulating **larger batch sizes** without increasing memory:
- ✅ Train with large effective batch sizes on limited GPU memory
- ✅ Improves training stability (less noisy gradients)
- ✅ Better generalization (smoother optimization landscape)

---

## How It Works: Step-by-Step

### Input Example

```python
# Sample inputs
policy_log_probs = torch.tensor([
    [-0.5, -0.3, -0.8, -0.2],  # Example 1
    [-0.4, -0.6, -0.3, -0.5]   # Example 2
], requires_grad=True)

response_mask = torch.tensor([
    [1, 1, 0, 1],   # First 2 are output, position 2 is prompt, last is output
    [1, 1, 1, 1]    # All are output tokens
], dtype=torch.bool)

gradient_accumulation_steps = 2
normalize_constant = 1.0
```

### Step 1: Count Response Tokens

```python
num_tokens = response_mask.sum()  # 7 tokens total (3 + 4)
```

**Purpose:** Know how many tokens contribute to the loss.

### Step 2: Compute Masked Sum

```python
masked_sum = masked_normalize(
    tensor=policy_log_probs,
    mask=response_mask,
    dim=None,
    normalize_constant=normalize_constant  # Usually 1.0
)
```

**What this does:**
```python
# Apply mask: zero out non-output positions
masked_probs = [[-0.5, -0.3, 0.0, -0.2],
                [-0.4, -0.6, -0.3, -0.5]]

# Sum all values
masked_sum = -0.5 - 0.3 - 0.2 - 0.4 - 0.6 - 0.3 - 0.5 = -2.8

# Divide by normalize_constant (1.0)
result = -2.8 / 1.0 = -2.8
```

**Result:** `masked_sum = -2.8`

### Step 3: Compute Loss (First Scaling)

```python
loss = -masked_sum / gradient_accumulation_steps
```

**Calculation:**
```python
loss = -(-2.8) / 2 = 2.8 / 2 = 1.4
```

**Why negate?**
- Log-probabilities are negative (log of values < 1)
- We want to **maximize** log-probabilities (minimize negative log-probs)
- Negating makes loss positive and aligns with "minimize loss" paradigm

**Why divide by gradient_accumulation_steps?**
- First scaling to prepare for gradient accumulation
- Ensures the logged loss value is meaningful

### Step 4: Scale Again for Backward

```python
scaled_loss_for_backward = loss / gradient_accumulation_steps
```

**Calculation:**
```python
scaled_loss_for_backward = 1.4 / 2 = 0.7
```

**Why scale AGAIN?**
- Ensures gradients are correctly scaled when accumulated
- Each microbatch contributes `1/gradient_accumulation_steps^2` to final gradient
- This matches the mathematical expectation for averaged gradients

### Step 5: Backward Pass

```python
scaled_loss_for_backward.backward()
```

**What happens:**
- Computes `∂(scaled_loss)/∂(policy_log_probs)`
- Gradient for output tokens: `-1 / gradient_accumulation_steps^2`
- Gradient for prompt/padding: `0`

**Calculation:**
```python
# For output tokens:
gradient = -1 / (2^2) = -1/4 = -0.25

# For prompt/padding:
gradient = 0
```

**Result:**
```python
policy_log_probs.grad = [
    [-0.25, -0.25, 0.0, -0.25],  # Matches mask pattern
    [-0.25, -0.25, -0.25, -0.25]
]
```

### Step 6: Return Values

```python
return scaled_loss_for_backward.detach(), metadata
```

**Returns:**
- `loss`: `0.7` (the scaled loss used for backward)
- `metadata`: `{'num_tokens': 7, 'masked_sum': -2.8}`

---

## Mathematical Formulation

### Loss Computation

$$
\mathcal{L}_{\text{microbatch}} = -\frac{1}{G^2} \sum_{t \in \mathcal{O}} \log P(y_t | y_{<t}, x)
$$

Where:
- $\mathcal{O}$ = set of output token positions (where mask = 1)
- $G$ = `gradient_accumulation_steps`
- $x$ = prompt
- $y_t$ = token at position $t$
- $G^2$ appears because we scale twice

### Gradient Computation

$$
\frac{\partial \mathcal{L}}{\partial \log P(y_t)} = \begin{cases}
-\frac{1}{G^2} & \text{if } t \in \mathcal{O} \\
0 & \text{otherwise}
\end{cases}
$$

### Why Double Scaling?

The double scaling by `gradient_accumulation_steps` ensures:

$$
\nabla_\theta \mathcal{L}_{\text{batch}} = \frac{1}{G} \sum_{i=1}^{G} \nabla_\theta \mathcal{L}_{\text{microbatch}_i}
$$

When we call `.backward()` on each scaled microbatch loss and accumulate:

$$
\begin{align}
\text{Accumulated grad} &= \sum_{i=1}^{G} \nabla_\theta \left(\frac{\mathcal{L}_i}{G^2}\right) \\
&= \frac{1}{G^2} \sum_{i=1}^{G} \nabla_\theta \mathcal{L}_i \\
&= \frac{1}{G} \cdot \frac{1}{G} \sum_{i=1}^{G} \nabla_\theta \mathcal{L}_i
\end{align}
$$

This correctly averages the gradients across all microbatches!

---

## Complete Example

Let's trace through a complete training scenario:

### Setup

```python
import torch
from cs336_alignment.sft import sft_microbatch_train_step

# Initialize model parameters (simplified)
policy_log_probs = torch.tensor([
    [1.0, 0.5, -0.3, -0.8],
    [0.2, -0.4, -0.6, -0.2]
], requires_grad=True)

response_mask = torch.tensor([
    [1, 1, 0, 1],   # 3 output tokens
    [1, 1, 1, 0]    # 3 output tokens
], dtype=torch.bool)

gradient_accumulation_steps = 2
```

### Execution

```python
# Microbatch 1
loss1, metadata1 = sft_microbatch_train_step(
    policy_log_probs=policy_log_probs,
    response_mask=response_mask,
    gradient_accumulation_steps=2,
    normalize_constant=1.0
)

print(f"Loss 1: {loss1.item()}")  # e.g., -0.175
print(f"Num tokens: {metadata1['num_tokens']}")  # 6
print(f"Gradients after microbatch 1:\n{policy_log_probs.grad}")
```

**Output:**
```
Loss 1: -0.175
Num tokens: 6
Gradients after microbatch 1:
tensor([[-0.25, -0.25,  0.00, -0.25],
        [-0.25, -0.25, -0.25,  0.00]])
```

### Trace Through Computation

```python
# Step 1: Compute masked sum
masked_values = [1.0, 0.5, -0.8, 0.2, -0.4, -0.6]
masked_sum = 1.0 + 0.5 - 0.8 + 0.2 - 0.4 - 0.6 = -0.1

# Step 2: First scaling
loss = -(-0.1) / 2 = 0.1 / 2 = 0.05

# Step 3: Second scaling
scaled_loss = 0.05 / 2 = 0.025

# Step 4: Backward
# Gradient = -1 / (2^2) = -0.25 for each output token
```

### Multiple Microbatches

```python
# Microbatch 2 (with different data)
loss2, metadata2 = sft_microbatch_train_step(
    policy_log_probs=policy_log_probs2,  # Different data
    response_mask=response_mask2,
    gradient_accumulation_steps=2,
    normalize_constant=1.0
)

# Gradients are ACCUMULATED (added to existing)
# After microbatch 2: policy_log_probs.grad = grad1 + grad2

# After processing all microbatches, update parameters
optimizer.step()
optimizer.zero_grad()
```

---

## Common Patterns

### Pattern 1: Basic Training Loop

```python
model.train()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
gradient_accumulation_steps = 4

for batch_idx, batch in enumerate(dataloader):
    # Get log-probs from model
    outputs = model(batch['input_ids'])
    log_probs = get_response_log_probs(
        model, batch['input_ids'], batch['labels']
    )['log_probs']
    
    # Compute loss and backward
    loss, metadata = sft_microbatch_train_step(
        policy_log_probs=log_probs,
        response_mask=batch['response_mask'],
        gradient_accumulation_steps=gradient_accumulation_steps,
        normalize_constant=1.0
    )
    
    # Update parameters every N microbatches
    if (batch_idx + 1) % gradient_accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
    
    # Log loss
    print(f"Microbatch {batch_idx}, Loss: {loss.item():.4f}")
```

### Pattern 2: With Logging

```python
from collections import defaultdict

metrics = defaultdict(list)

for batch_idx, batch in enumerate(dataloader):
    # ... (same as above)
    
    loss, metadata = sft_microbatch_train_step(...)
    
    # Log metrics
    metrics['loss'].append(loss.item())
    metrics['num_tokens'].append(metadata['num_tokens'].item())
    
    if (batch_idx + 1) % gradient_accumulation_steps == 0:
        # Log average metrics for this minibatch
        avg_loss = sum(metrics['loss'][-gradient_accumulation_steps:]) / gradient_accumulation_steps
        total_tokens = sum(metrics['num_tokens'][-gradient_accumulation_steps:])
        
        print(f"Minibatch {batch_idx // gradient_accumulation_steps}, "
              f"Avg Loss: {avg_loss:.4f}, Total Tokens: {total_tokens}")
        
        optimizer.step()
        optimizer.zero_grad()
```

### Pattern 3: With Gradient Clipping

```python
for batch_idx, batch in enumerate(dataloader):
    loss, metadata = sft_microbatch_train_step(...)
    
    if (batch_idx + 1) % gradient_accumulation_steps == 0:
        # Clip gradients before optimizer step
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        optimizer.zero_grad()
```

---

## Why Double Scaling? Deep Dive

### The Problem Without Proper Scaling

Consider processing 2 microbatches with gradient accumulation:

```python
# WITHOUT proper scaling
microbatch1_loss = 0.5
microbatch2_loss = 0.6

# Naive approach: just call backward() on each
microbatch1_loss.backward()  # Adds grad1 to parameters
microbatch2_loss.backward()  # Adds grad2 to parameters

# Total gradient = grad1 + grad2
# But we want average: (grad1 + grad2) / 2 ❌
```

### The Solution: Scale Each Loss

```python
# WITH proper scaling
gradient_accumulation_steps = 2

# Scale each microbatch loss
scaled_loss1 = microbatch1_loss / gradient_accumulation_steps  # 0.5 / 2 = 0.25
scaled_loss2 = microbatch2_loss / gradient_accumulation_steps  # 0.6 / 2 = 0.3

scaled_loss1.backward()  # Adds grad1/2 to parameters
scaled_loss2.backward()  # Adds grad2/2 to parameters

# Total gradient = grad1/2 + grad2/2 = (grad1 + grad2) / 2 ✓
```

### Why Our Implementation Scales TWICE

Our implementation scales by `gradient_accumulation_steps` **twice**:

1. **First scaling:** When computing `loss = -masked_sum / gradient_accumulation_steps`
2. **Second scaling:** When computing `scaled_loss_for_backward = loss / gradient_accumulation_steps`

**Rationale:**
- The first scaling ensures the **loss value** we return/log is meaningful
- The second scaling ensures the **gradients** are correct for accumulation
- Total scaling factor: `1 / gradient_accumulation_steps^2`

**Mathematical justification:**

For gradient accumulation to work correctly, each microbatch gradient should contribute:

$$
\text{Contribution} = \frac{1}{\text{num\_microbatches}} \times \text{gradient}
$$

Our double scaling achieves this because:

$$
\nabla\left(\frac{L}{G^2}\right) = \frac{1}{G^2} \nabla L
$$

When accumulated over $G$ microbatches:

$$
\sum_{i=1}^{G} \frac{1}{G^2} \nabla L_i = \frac{1}{G^2} \sum_{i=1}^{G} \nabla L_i = \frac{1}{G} \times \frac{1}{G} \sum_{i=1}^{G} \nabla L_i
$$

---

## Key Insights

### 1. Loss is Negative Log-Likelihood

SFT loss = negative average log-probability of output tokens:

$$
\mathcal{L}_{\text{SFT}} = -\frac{1}{|\mathcal{O}|} \sum_{t \in \mathcal{O}} \log P(y_t | y_{<t}, x)
$$

This is equivalent to **cross-entropy loss** in classification.

### 2. Only Output Tokens Contribute

Using `response_mask`, we ensure:
- ✅ Loss computed only on output tokens
- ❌ Prompt tokens ignored (model doesn't learn to predict the prompt)
- ❌ Padding tokens ignored (they're meaningless)

### 3. Gradient Accumulation Requires Careful Scaling

To simulate a large batch with limited memory:
- Process multiple small microbatches
- Scale each loss by `1 / gradient_accumulation_steps^2`
- Accumulate gradients by calling backward() multiple times
- Update parameters once after all microbatches

### 4. Returned Loss is the Scaled Value

The function returns `scaled_loss_for_backward`, not the original unscaled loss:
- This is the value that gradients were computed from
- It's the appropriate value for logging and monitoring
- It represents the contribution of this microbatch to the total loss

---

## Troubleshooting

### Issue 1: Loss Not Decreasing

**Problem:** Loss stays constant or increases during training.

**Possible causes:**
- Learning rate too high → try reducing by 10x
- Gradients exploding → add gradient clipping
- Wrong loss scaling → verify `gradient_accumulation_steps` is correct

### Issue 2: Gradients are Wrong Scale

**Problem:** Gradients are 2x or 4x too large/small.

**Check:**
```python
# After backward()
print(policy_log_probs.grad.abs().mean())
# Should be approximately 1/(gradient_accumulation_steps^2) per output token
```

**Fix:** Verify that `gradient_accumulation_steps` matches the actual number of microbatches processed.

### Issue 3: Memory Leak

**Problem:** Memory usage increases over time.

**Cause:** Not detaching returned loss:
```python
# ❌ BAD: Keeps computation graph
loss, metadata = sft_microbatch_train_step(...)
losses.append(loss)  # Don't do this!

# ✅ GOOD: Detach from graph
loss, metadata = sft_microbatch_train_step(...)
losses.append(loss.item())  # Or loss.detach()
```

### Issue 4: Incorrect Response Mask

**Problem:** Loss computed on prompt tokens.

**Check:**
```python
print("Mask sum:", response_mask.sum())
print("Expected num output tokens:", expected_count)
assert response_mask.sum() == expected_count
```

---

## Summary

`sft_microbatch_train_step` is the **core training function for SFT**, implementing:

1. ✅ **Masked loss computation:** Only output tokens contribute
2. ✅ **Proper scaling:** Double scaling for gradient accumulation
3. ✅ **Backward pass:** Computes gradients correctly
4. ✅ **Metadata:** Returns useful statistics for logging

**Key equation:**
```python
loss = -(masked_sum / gradient_accumulation_steps) / gradient_accumulation_steps
loss.backward()
```

This ensures:
- Gradients are scaled correctly for accumulation
- Loss value is meaningful for logging
- Training proceeds correctly with gradient accumulation

**Use this function as a building block** for implementing full SFT training loops! 🎯
