# Understanding `get_response_log_probs` Function

## What Does This Function Do?

`get_response_log_probs` computes **how likely each token in the sequence was** according to the model. This is essential for:

1. **Training**: Computing the loss (how wrong was the model?)
2. **Evaluation**: Measuring model performance
3. **Reinforcement Learning**: Comparing old and new policies in GRPO

## The Big Picture

```
INPUT:
┌────────────────────────────────────────┐
│ input_ids: [10, 20, 30, 40]            │  What model sees
│ labels:    [20, 30, 40, 50]            │  What model should predict
└────────────────────────────────────────┘
                ↓
        ┌──────────────┐
        │  FORWARD PASS │
        └──────────────┘
                ↓
┌────────────────────────────────────────┐
│ logits: [batch, seq, vocab_size]       │  Raw scores for all tokens
│        (e.g., [2, 4, 50000])            │
└────────────────────────────────────────┘
                ↓
        ┌──────────────┐
        │  LOG-SOFTMAX  │
        └──────────────┘
                ↓
┌────────────────────────────────────────┐
│ log_probs of ALL tokens                │
│ [batch, seq, vocab_size]                │
└────────────────────────────────────────┘
                ↓
        ┌──────────────┐
        │  GATHER      │  Extract only label tokens
        └──────────────┘
                ↓
OUTPUT:
┌────────────────────────────────────────┐
│ log_probs: [batch, seq]                │  Log P(actual tokens)
│ [-0.5, -1.2, -0.8, -2.1]               │
│                                         │
│ Optional: token_entropy [batch, seq]   │
└────────────────────────────────────────┘
```

## Step-by-Step Explanation

### Step 1: Forward Pass

```python
logits = model(input_ids).logits  # [batch_size, seq_length, vocab_size]
```

**What happens:**
- Model processes input_ids through all its layers
- At each position, outputs scores for every possible next token
- These scores (logits) are **raw and unnormalized**

**Example:**
```
input_ids = [[10, 20, 30]]

Position 0: Model sees [10], outputs logits for all 50,000 tokens
            logits[0, 0] = [2.1, -0.5, 1.3, ..., 0.8]  (50,000 values)

Position 1: Model sees [10, 20], outputs logits for all 50,000 tokens
            logits[0, 1] = [1.8, 0.2, -0.3, ..., 1.5]  (50,000 values)

Position 2: Model sees [10, 20, 30], outputs logits for all 50,000 tokens
            logits[0, 2] = [0.5, 1.1, 2.3, ..., -0.2]  (50,000 values)
```

### Step 2: Convert to Log-Probabilities

```python
log_softmax = torch.nn.functional.log_softmax(logits, dim=-1)
```

**What happens:**
- Converts raw logits to log-probabilities using softmax
- `log_softmax(x) = x - log(sum(exp(x)))`
- More numerically stable than `log(softmax(x))`

**Why log-probabilities?**
1. **Numerical stability**: Regular probabilities can be very small (e.g., 1e-20)
2. **Math convenience**: Multiplication becomes addition in log space
3. **Loss computation**: Cross-entropy directly uses log-probabilities

**Example at position 0:**
```
Raw logits:        [2.1, -0.5, 1.3, 0.8, ...]
After softmax:     [0.42, 0.03, 0.19, 0.11, ...]  (probabilities sum to 1.0)
After log_softmax: [-0.87, -3.51, -1.66, -2.20, ...]  (log probabilities)
```

### Step 3: Extract Log-Probabilities of Actual Tokens

```python
log_probs = torch.gather(log_softmax, -1, labels.unsqueeze(-1)).squeeze(-1)
```

**What happens:**
This is the KEY step! We have log-probabilities for ALL 50,000 tokens at each position, but we only care about the tokens that ACTUALLY appeared (the labels).

**Breaking down `torch.gather`:**

```python
# Step 3a: Prepare labels for gathering
labels.unsqueeze(-1)
# Before: [batch, seq]          e.g., [[20, 30, 40]]
# After:  [batch, seq, 1]       e.g., [[[20], [30], [40]]]

# Step 3b: Gather from vocabulary dimension
torch.gather(log_softmax, dim=-1, index=labels.unsqueeze(-1))
# From log_softmax[batch, seq, vocab_size], extract values at indices specified by labels
# Result: [batch, seq, 1]

# Step 3c: Remove extra dimension
.squeeze(-1)
# Result: [batch, seq]
```

**Visual Example:**

```
Position 0:
  labels[0] = 20  (we want token 20)
  log_softmax[0] has 50,000 values: [-0.87, ..., -1.66 (at index 20), ..., -2.20]
  Extract: log_softmax[0, 20] = -1.66
  Meaning: log P(token_20 | input_10) = -1.66
           P(token_20 | input_10) = exp(-1.66) ≈ 0.19 (19% probability)

Position 1:
  labels[1] = 30  (we want token 30)
  log_softmax[1] has 50,000 values: [-1.2, ..., -0.85 (at index 30), ..., -3.1]
  Extract: log_softmax[1, 30] = -0.85
  Meaning: log P(token_30 | input_10,20) = -0.85
           P(token_30 | input_10,20) = exp(-0.85) ≈ 0.43 (43% probability)

Position 2:
  labels[2] = 40  (we want token 40)
  Extract: log_softmax[2, 40] = -2.1
  Meaning: log P(token_40 | input_10,20,30) = -2.1
           P(token_40 | input_10,20,30) = exp(-2.1) ≈ 0.12 (12% probability)

Final result: log_probs = [-1.66, -0.85, -2.1]
```

### Step 4: Optionally Compute Entropy

```python
if return_token_entropy:
    result['token_entropy'] = compute_entropy(logits)
```

**What happens:**
- Measures how uncertain the model is at each position
- High entropy = model is unsure (many tokens have similar probabilities)
- Low entropy = model is confident (one token has much higher probability)

**Example:**
```
Position 0: probabilities = [0.42, 0.19, 0.15, 0.11, 0.08, ...]
            Fairly spread out → Moderate entropy (≈2.3)

Position 1: probabilities = [0.85, 0.05, 0.03, 0.02, 0.02, ...]
            Peaked on one token → Low entropy (≈0.7)

Position 2: probabilities = [0.25, 0.25, 0.25, 0.25, ...]
            Uniform → High entropy (≈1.39)
```

## Why We DON'T Mask Here

The function computes log-probs for **all positions**, including:
- Prompt tokens
- Output tokens  
- Padding tokens

**Why not mask?**
1. **Separation of concerns**: This function computes probabilities, the caller decides which to use
2. **Flexibility**: Different use cases need different masking
   - Training: Mask prompt and padding
   - Evaluation: Maybe mask only padding
   - GRPO: Need log-probs for entire response
3. **Efficiency**: Masking is cheap, forward pass is expensive (do it once)

**Where masking happens:**
```python
# In the training loop:
log_probs = get_response_log_probs(model, input_ids, labels)['log_probs']
# log_probs shape: [batch, seq]  includes all positions

masked_log_probs = log_probs * response_mask
# Only keep log-probs where response_mask is True

loss = -masked_log_probs.sum() / response_mask.sum()
# Compute loss only on output tokens
```

## Connection to Training Loss

This function is the foundation of the **cross-entropy loss**:

```python
# What get_response_log_probs computes:
log_probs = get_response_log_probs(model, input_ids, labels)['log_probs']
# log_probs[i, j] = log P(labels[i,j] | input_ids[i, :j])

# Cross-entropy loss (negative log-likelihood):
loss = -log_probs[response_mask].mean()

# Why negative?
# - High log-prob (e.g., -0.1) → Low loss → Good!
# - Low log-prob (e.g., -5.0) → High loss → Bad!
# We want to MAXIMIZE log-prob = MINIMIZE negative log-prob
```

## Practical Example

Let's trace through a complete example:

```python
# Input
input_ids = [[10, 20, 30]]  # "What is"
labels    = [[20, 30, 40]]  # "is 2+2?"

# Step 1: Forward pass
logits = model(input_ids).logits
# Shape: [1, 3, 50000]
# logits[0, 0, :] = scores for all 50k tokens given [10]
# logits[0, 1, :] = scores for all 50k tokens given [10, 20]
# logits[0, 2, :] = scores for all 50k tokens given [10, 20, 30]

# Step 2: Convert to log-probs
log_softmax = F.log_softmax(logits, dim=-1)
# Shape: [1, 3, 50000]
# log_softmax[0, 0, :] = log-probs for all 50k tokens

# Step 3: Extract log-probs of actual tokens
# At position 0: Want log P(token_20)
log_probs[0, 0] = log_softmax[0, 0, 20]  = -0.87

# At position 1: Want log P(token_30)
log_probs[0, 1] = log_softmax[0, 1, 30]  = -1.23

# At position 2: Want log P(token_40)
log_probs[0, 2] = log_softmax[0, 2, 40]  = -2.15

# Final result
log_probs = [[-0.87, -1.23, -2.15]]

# Interpretation:
# Position 0: 42% chance of predicting token_20 correctly
# Position 1: 29% chance of predicting token_30 correctly  
# Position 2: 12% chance of predicting token_40 correctly
```

## Common Use Cases

### 1. **SFT Training Loss**
```python
outputs = get_response_log_probs(model, input_ids, labels)
log_probs = outputs['log_probs']  # [batch, seq]

# Apply response mask (only train on output tokens)
masked_log_probs = log_probs * response_mask
loss = -masked_log_probs.sum() / response_mask.sum()
loss.backward()
```

### 2. **GRPO (Policy Comparison)**
```python
# Current policy
new_log_probs = get_response_log_probs(model, input_ids, labels)['log_probs']

# Old policy (frozen)
with torch.no_grad():
    old_log_probs = get_response_log_probs(ref_model, input_ids, labels)['log_probs']

# Compute KL divergence or ratio
ratio = torch.exp(new_log_probs - old_log_probs)
```

### 3. **Model Confidence Analysis**
```python
outputs = get_response_log_probs(model, input_ids, labels, return_token_entropy=True)
log_probs = outputs['log_probs']
entropy = outputs['token_entropy']

# Find positions where model is uncertain
uncertain_positions = entropy > threshold

# Find positions where model was wrong and uncertain
wrong_and_uncertain = (log_probs < -2.0) & (entropy > 2.0)
```

## Key Takeaways

1. **Purpose**: Compute log P(actual_token | context) for each position
2. **Method**: Forward pass → log-softmax → gather
3. **Output**: Log-probabilities of labels + optional entropy
4. **No masking**: Returns values for all positions; masking done by caller
5. **Foundation**: Core building block for SFT loss and GRPO

Your implementation is **correct and efficient**! 🎉
