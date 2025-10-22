# `sft_microbatch_train_step` Visual Cheatsheet 🎯

## Quick Reference

```python
loss, metadata = sft_microbatch_train_step(
    policy_log_probs,              # [B, S] - model's log-probabilities
    response_mask,                 # [B, S] - 1=output, 0=prompt/pad
    gradient_accumulation_steps,   # int - number of microbatches per step
    normalize_constant=1.0         # float - additional scaling factor
)
```

**Returns:** `(scaled_loss, metadata_dict)`

---

## Visual Flow

```
┌──────────────────────────────────────────────────────────────────┐
│               Input: Log-Probs & Response Mask                    │
│                                                                   │
│  log_probs:  [[-0.5, -0.3, -0.8, -0.2],                         │
│               [-0.4, -0.6, -0.3, -0.5]]                          │
│                                                                   │
│  mask:       [[  1,    1,    0,    1],                           │
│               [  1,    1,    1,    1]]                           │
│              output  |  prompt | output                          │
└──────────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────────┐
│                 Step 1: Count Response Tokens                     │
│                                                                   │
│  num_tokens = mask.sum() = 7                                     │
└──────────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────────┐
│              Step 2: Compute Masked Sum                           │
│                                                                   │
│  masked_normalize(log_probs, mask, dim=None, normalize_const)   │
│                                                                   │
│  masked = [[-0.5, -0.3, 0.0, -0.2],                             │
│            [-0.4, -0.6, -0.3, -0.5]]                            │
│                                                                   │
│  sum = -0.5 - 0.3 - 0.2 - 0.4 - 0.6 - 0.3 - 0.5 = -2.8         │
│  masked_sum = -2.8 / 1.0 = -2.8                                 │
└──────────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────────┐
│            Step 3: First Scaling (Negate & Scale)                │
│                                                                   │
│  loss = -masked_sum / gradient_accumulation_steps                │
│       = -(-2.8) / 2                                              │
│       = 2.8 / 2                                                  │
│       = 1.4                                                      │
└──────────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────────┐
│         Step 4: Second Scaling (For Gradient Accumulation)       │
│                                                                   │
│  scaled_loss = loss / gradient_accumulation_steps                │
│              = 1.4 / 2                                           │
│              = 0.7                                               │
└──────────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────────┐
│                  Step 5: Backward Pass                            │
│                                                                   │
│  scaled_loss.backward()                                          │
│                                                                   │
│  Gradient = -1 / (gradient_accumulation_steps^2)                 │
│           = -1 / 4 = -0.25  (for output tokens)                 │
│           = 0               (for prompt/padding)                 │
│                                                                   │
│  log_probs.grad = [[-0.25, -0.25, 0.0, -0.25],                 │
│                    [-0.25, -0.25, -0.25, -0.25]]                │
└──────────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────────┐
│                  Step 6: Return Loss & Metadata                   │
│                                                                   │
│  return (0.7, {'num_tokens': 7, 'masked_sum': -2.8})           │
└──────────────────────────────────────────────────────────────────┘
```

---

## Key Formula

### Loss Computation

```
┌─────────────────────────────────────────────────────────────┐
│  masked_sum = Σ(log_probs * mask) / normalize_constant     │
│            ↓                                                 │
│  loss = -masked_sum / gradient_accumulation_steps           │
│            ↓                                                 │
│  scaled_loss = loss / gradient_accumulation_steps           │
│            ↓                                                 │
│  scaled_loss.backward()                                     │
└─────────────────────────────────────────────────────────────┘

Total scaling: 1 / (gradient_accumulation_steps^2)
```

### Gradient Formula

```
         ┌  -1                                    ┐
         │ ───────────────────────────  if mask=1│
grad =   │ gradient_accumulation_steps²          │
         │                                        │
         │  0                           if mask=0│
         └                                        ┘
```

---

## Double Scaling Explained

### Why Scale Twice?

```
Without Scaling:
  Microbatch 1: loss=0.5 → backward() → grad1
  Microbatch 2: loss=0.6 → backward() → grad2
  Total grad = grad1 + grad2  ❌ Too large!

With Single Scaling (÷G):
  Microbatch 1: loss=0.5/2=0.25 → backward() → grad1/2
  Microbatch 2: loss=0.6/2=0.30 → backward() → grad2/2  
  Total grad = (grad1 + grad2) / 2  ✓ Correct average

With Double Scaling (÷G²):
  Microbatch 1: loss=0.5/4=0.125 → backward() → grad1/4
  Microbatch 2: loss=0.6/4=0.150 → backward() → grad2/4
  Total grad = (grad1 + grad2) / 4  ✓ Also correct!

Our implementation uses double scaling ÷G² because:
- First ÷G: Scales loss value for logging
- Second ÷G: Scales gradients for accumulation
```

### Visual Comparison

```
       Microbatch                Accumulated Gradient
       
  ┌─────────────┐              ┌─────────────────┐
  │ MB1: L₁/G²  │──────────→   │   grad₁/G²      │
  └─────────────┘              └─────────────────┘
                                        ↓
  ┌─────────────┐                      +
  │ MB2: L₂/G²  │──────────→   ┌─────────────────┐
  └─────────────┘              │ grad₁/G²        │
                               │     +           │
                               │ grad₂/G²        │
                               └─────────────────┘
                                        ↓
  ┌─────────────┐                      +
  │ MB3: L₃/G²  │──────────→   ┌─────────────────┐
  └─────────────┘              │ Σ(grad_i) / G²  │
                               └─────────────────┘
                                        ↓
                               optimizer.step()
```

---

## Gradient Accumulation Example

### Setup

```python
gradient_accumulation_steps = 4
batch_size_per_microbatch = 8
total_batch_size = 32  # 4 × 8
```

### Training Loop

```
┌──────────────────────────────────────────────────────────┐
│ Minibatch (32 samples)                                   │
│                                                           │
│  ┌────────────┐                                          │
│  │ Microbatch │  Samples 0-7    loss/16 → backward()    │
│  │     1      │                 (accumulate grads)       │
│  └────────────┘                                          │
│       ↓                                                   │
│  ┌────────────┐                                          │
│  │ Microbatch │  Samples 8-15   loss/16 → backward()    │
│  │     2      │                 (accumulate grads)       │
│  └────────────┘                                          │
│       ↓                                                   │
│  ┌────────────┐                                          │
│  │ Microbatch │  Samples 16-23  loss/16 → backward()    │
│  │     3      │                 (accumulate grads)       │
│  └────────────┘                                          │
│       ↓                                                   │
│  ┌────────────┐                                          │
│  │ Microbatch │  Samples 24-31  loss/16 → backward()    │
│  │     4      │                 (accumulate grads)       │
│  └────────────┘                                          │
│       ↓                                                   │
│  ┌────────────┐                                          │
│  │  Optimizer │  Update parameters                       │
│  │    Step    │  Zero gradients                          │
│  └────────────┘                                          │
└──────────────────────────────────────────────────────────┘
```

---

## Code Template

### Basic Usage

```python
import torch
from cs336_alignment.sft import sft_microbatch_train_step

# Your data
policy_log_probs = model_output_log_probs  # [batch, seq_len]
response_mask = your_mask                   # [batch, seq_len]
gradient_accumulation_steps = 4

# Single microbatch step
loss, metadata = sft_microbatch_train_step(
    policy_log_probs=policy_log_probs,
    response_mask=response_mask,
    gradient_accumulation_steps=gradient_accumulation_steps,
    normalize_constant=1.0
)

print(f"Loss: {loss.item():.4f}")
print(f"Tokens: {metadata['num_tokens']}")
```

### Full Training Loop

```python
model.train()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
gradient_accumulation_steps = 4

for minibatch_idx, microbatches in enumerate(dataloader):
    for microbatch_idx, batch in enumerate(microbatches):
        # Forward pass
        outputs = model(batch['input_ids'])
        log_probs = get_response_log_probs(
            model, batch['input_ids'], batch['labels']
        )['log_probs']
        
        # Compute loss and backward
        loss, metadata = sft_microbatch_train_step(
            policy_log_probs=log_probs,
            response_mask=batch['response_mask'],
            gradient_accumulation_steps=gradient_accumulation_steps
        )
        
        print(f"Microbatch {microbatch_idx}, Loss: {loss.item():.4f}")
    
    # Update after all microbatches
    optimizer.step()
    optimizer.zero_grad()
    print(f"Minibatch {minibatch_idx} complete\n")
```

---

## Input/Output Shapes

```
Inputs:
  policy_log_probs:  [batch_size, sequence_length]  (requires_grad=True)
  response_mask:     [batch_size, sequence_length]  (dtype=torch.bool)
  gradient_acc_steps: scalar int
  normalize_constant: scalar float

Outputs:
  loss:              scalar tensor (detached)
  metadata:          dict with:
                       - 'num_tokens': scalar int
                       - 'masked_sum': scalar tensor (detached)

Side Effects:
  policy_log_probs.grad is modified (gradients accumulated)
```

---

## Common Values

| Parameter | Typical Values | Notes |
|-----------|---------------|-------|
| `gradient_accumulation_steps` | 1, 2, 4, 8, 16 | Higher = larger effective batch |
| `normalize_constant` | 1.0 | Usually left at default |
| `loss` (returned) | -5.0 to 5.0 | Depends on model quality |
| `grad` (computed) | -1 to 0 | Negative for output tokens |

---

## Gradient Magnitude

```
For gradient_accumulation_steps = G:

Expected gradient per output token:
  grad = -1 / G²

Examples:
  G=1  → grad = -1.00  (no accumulation)
  G=2  → grad = -0.25  (our example)
  G=4  → grad = -0.0625
  G=8  → grad = -0.015625
```

---

## Loss Interpretation

```
Loss Value     Meaning
──────────────────────────────────────────
< -2.0        Very good (high probability)
-2.0 to -1.0  Good
-1.0 to 0.0   Moderate
0.0 to 1.0    Weak
> 1.0         Poor (low probability)

Note: Loss can be negative because:
- If log_probs are positive (prob > 1 impossible, but can happen early in training)
- The masked_sum is negative (usual case)
- We negate it: -(-value) = +value
```

---

## Debugging Checklist

- [ ] **Log-probs are negative**: Check `policy_log_probs.mean() < 0`
- [ ] **Mask matches output**: Verify `response_mask.sum()` equals expected tokens
- [ ] **Gradients have correct scale**: Check `grad ≈ -1/G²` for output tokens
- [ ] **Loss is reasonable**: Typically in range -5 to 5
- [ ] **No NaN/Inf**: Check `torch.isnan(loss)` and `torch.isinf(loss)`
- [ ] **Requires grad**: Ensure `policy_log_probs.requires_grad == True`

---

## Quick Debug Commands

```python
# Check input shapes
print(f"Log probs shape: {policy_log_probs.shape}")
print(f"Mask shape: {response_mask.shape}")

# Check values
print(f"Log probs range: [{policy_log_probs.min():.2f}, {policy_log_probs.max():.2f}]")
print(f"Num output tokens: {response_mask.sum()}")

# Check gradients after backward
print(f"Gradient range: [{policy_log_probs.grad.min():.4f}, {policy_log_probs.grad.max():.4f}]")
print(f"Gradient at output positions: {policy_log_probs.grad[response_mask].mean():.4f}")
print(f"Expected gradient: {-1/(gradient_accumulation_steps**2):.4f}")
```

---

## Performance Tips

### ✅ Do:
```python
# Detach loss for logging
losses.append(loss.item())

# Clear gradients after optimizer step
optimizer.step()
optimizer.zero_grad()

# Use mixed precision
with torch.cuda.amp.autocast():
    loss, meta = sft_microbatch_train_step(...)
```

### ❌ Don't:
```python
# Keep loss with grad attached
losses.append(loss)  # Memory leak!

# Forget to zero gradients
# (gradients keep accumulating forever)

# Call backward() twice on same loss
loss.backward()
loss.backward()  # Error!
```

---

## Mathematical Summary

### Core Equation

$$
\mathcal{L} = -\frac{1}{G^2} \sum_{t \in \mathcal{O}} \log P(y_t | y_{<t}, x)
$$

Where:
- $\mathcal{O}$ = output token positions
- $G$ = `gradient_accumulation_steps`
- $P(y_t | y_{<t}, x)$ = probability of token $y_t$ given context

### Gradient

$$
\frac{\partial \mathcal{L}}{\partial \log P(y_t)} = \begin{cases}
-\frac{1}{G^2} & \text{if } t \in \mathcal{O} \\
0 & \text{otherwise}
\end{cases}
$$

---

## Remember

🔑 **Key insight:** Double scaling (÷G²) enables correct gradient accumulation

💡 **SFT loss:** Negative log-likelihood of output tokens only

⚡ **Accumulation:** Process multiple microbatches, update once

🎯 **Goal:** Maximize probability of correct output tokens

---

## When to Use

| Scenario | Use This Function? |
|----------|-------------------|
| Training SFT model | ✅ Yes |
| Evaluating model | ❌ No (no backward needed) |
| Computing loss only | ❌ Use simpler function |
| RLHF/GRPO training | ❌ Different loss function |
| Gradient accumulation | ✅ Yes (designed for this) |
