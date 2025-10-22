# Understanding `compute_entropy` for Language Models

## What is Entropy?

**Entropy** is a measure of **uncertainty** or **randomness** in a probability distribution.

### Intuitive Examples

**Low Entropy (Certain/Confident):**
```
Token probabilities: [0.95, 0.02, 0.02, 0.01]
Model says: "I'm 95% sure the next token is 'the'"
Entropy ≈ 0.3 (LOW - very confident!)
```

**High Entropy (Uncertain/Random):**
```
Token probabilities: [0.25, 0.25, 0.25, 0.25]
Model says: "I have no idea which token comes next"
Entropy ≈ 1.39 (HIGH - very uncertain!)
```

## The Entropy Formula

```
H(p) = -∑ p(x) × log p(x)
```

Breaking it down:
- `p(x)`: Probability of outcome x
- `log p(x)`: Log-probability (negative since p(x) < 1)
- `p(x) × log p(x)`: Weighted contribution of each outcome
- `∑`: Sum over all possible outcomes
- `-`: Negative sign makes the result positive

## Why Entropy Matters in Language Models

### 1. **Measuring Confidence**
```python
# Low entropy = Model is confident
logits = [10.0, 1.0, 0.5, 0.1]  # One logit dominates
→ entropy ≈ 0.2  # "I'm sure it's token 0"

# High entropy = Model is confused
logits = [2.0, 2.1, 1.9, 2.0]  # All logits similar
→ entropy ≈ 1.38  # "Could be any of these..."
```

### 2. **Detecting Overfitting**
```
Training entropy: 0.01  ← Extremely low!
Validation entropy: 5.2  ← Very high!
→ Model memorized training data but can't generalize
```

### 3. **Reinforcement Learning**
```
Many RL algorithms use entropy as a regularization term:
loss = policy_loss - β × entropy

Why?
- High entropy → More exploration (try different actions)
- Low entropy → More exploitation (stick to known good actions)
- Balance is crucial for learning
```

## Step-by-Step: How the Code Works

### Step 1: Convert Logits to Log-Probabilities

**The Problem:**
```python
# Naive approach (DON'T DO THIS!)
probs = exp(logits) / sum(exp(logits))

# If logits = [1000, 999, 998, ...]:
exp(1000) = INFINITY → NaN → Everything breaks!
```

**The Solution (Log-Sum-Exp Trick):**
```python
log_probs = logits - logsumexp(logits)

# This is numerically stable!
# logsumexp internally does: max + log(sum(exp(logits - max)))
```

**Example:**
```
logits = [2.0, 1.0, 0.5, 0.3]

Step 1a: Compute logsumexp
  logsumexp = log(e^2.0 + e^1.0 + e^0.5 + e^0.3)
            = log(7.39 + 2.72 + 1.65 + 1.35)
            = log(13.11)
            = 2.57

Step 1b: Subtract to get log-probabilities
  log_probs[0] = 2.0 - 2.57 = -0.57
  log_probs[1] = 1.0 - 2.57 = -1.57
  log_probs[2] = 0.5 - 2.57 = -2.07
  log_probs[3] = 0.3 - 2.57 = -2.27
```

### Step 2: Exponentiate to Get Probabilities

```python
probs = exp(log_probs)
```

**Continuing the example:**
```
probs[0] = e^(-0.57) = 0.565
probs[1] = e^(-1.57) = 0.208
probs[2] = e^(-2.07) = 0.126
probs[3] = e^(-2.27) = 0.103

Check: 0.565 + 0.208 + 0.126 + 0.103 = 1.002 ✓
(Close to 1.0, as probabilities should be!)
```

### Step 3: Compute Entropy

```python
entropy = -sum(probs * log_probs)
```

**Continuing the example:**
```
Component-wise multiplication:
  probs[0] * log_probs[0] = 0.565 × (-0.57) = -0.322
  probs[1] * log_probs[1] = 0.208 × (-1.57) = -0.327
  probs[2] * log_probs[2] = 0.126 × (-2.07) = -0.261
  probs[3] * log_probs[3] = 0.103 × (-2.27) = -0.234

Sum: -0.322 + -0.327 + -0.261 + -0.234 = -1.144

Entropy = -(-1.144) = 1.144
```

### Interpretation:
- Max entropy for 4 tokens: log(4) = 1.386
- Our entropy: 1.144
- Percentage: 1.144/1.386 = 82.5% of maximum
- Meaning: Distribution is fairly uncertain but not completely uniform

## Visual Understanding

### Low Entropy Distribution
```
Probability Distribution:
Token 0: ████████████████████ (95%)
Token 1: █ (2%)
Token 2: █ (2%)
Token 3: ▌ (1%)

Entropy ≈ 0.3
Interpretation: "Almost certainly token 0"
```

### Medium Entropy Distribution
```
Probability Distribution:
Token 0: ████████████ (60%)
Token 1: ████ (20%)
Token 2: ██ (10%)
Token 3: ██ (10%)

Entropy ≈ 1.2
Interpretation: "Probably token 0, but others possible"
```

### High Entropy Distribution
```
Probability Distribution:
Token 0: █████ (25%)
Token 1: █████ (25%)
Token 2: █████ (25%)
Token 3: █████ (25%)

Entropy ≈ 1.39 (maximum for 4 tokens)
Interpretation: "No idea - all equally likely"
```

## Why the Negative Sign?

```
log p(x) is always ≤ 0 because:
  - If p(x) = 1.0 → log(1.0) = 0
  - If p(x) = 0.5 → log(0.5) = -0.69
  - If p(x) = 0.1 → log(0.1) = -2.30

Therefore:
  p(x) × log p(x) is always ≤ 0

Sum of negative values is negative:
  sum(p(x) × log p(x)) ≤ 0

We negate to make entropy positive (convention):
  H(p) = -sum(p(x) × log p(x)) ≥ 0
```

## Practical Use Cases

### 1. **Temperature Scaling**
```python
# High temperature → High entropy (more random)
logits_high_temp = logits / 2.0
entropy_high = compute_entropy(logits_high_temp)  # Large

# Low temperature → Low entropy (more deterministic)
logits_low_temp = logits / 0.5
entropy_low = compute_entropy(logits_low_temp)  # Small
```

### 2. **Monitoring Training**
```python
# During training, track entropy
for batch in dataloader:
    logits = model(batch)
    entropy = compute_entropy(logits).mean()
    log(f"Average entropy: {entropy:.3f}")

# Healthy training:
# Epoch 1: entropy ≈ 8.0 (model is uncertain)
# Epoch 10: entropy ≈ 2.0 (model is learning)
# Epoch 20: entropy ≈ 0.5 (model is confident)

# Unhealthy training:
# Epoch 1: entropy ≈ 8.0
# Epoch 10: entropy ≈ 0.001 (TOO low, might be overfitting)
```

### 3. **Entropy Regularization in RL**
```python
# Encourage exploration by penalizing low entropy
policy_loss = -rewards * log_probs
entropy_bonus = β * compute_entropy(logits)
total_loss = policy_loss - entropy_bonus  # Minus = encourage high entropy

# β controls exploration:
# - High β → Explore more (higher entropy preferred)
# - Low β → Exploit more (lower entropy OK)
```

## Shape Transformations

```python
# Input
logits: [batch_size, seq_len, vocab_size]
        [    2,         10,      50000   ]

# After logsumexp with keepdim=True
logsumexp: [batch_size, seq_len,  1  ]
           [    2,         10,     1  ]
           (Broadcasting dimension preserved)

# After subtraction (broadcasting)
log_probs: [batch_size, seq_len, vocab_size]
           [    2,         10,      50000   ]

# After exp
probs:     [batch_size, seq_len, vocab_size]
           [    2,         10,      50000   ]

# After sum(dim=-1) - sum over vocabulary
entropy:   [batch_size, seq_len]
           [    2,         10   ]
           (One entropy value per position)
```

## Common Pitfalls

### ❌ **Pitfall 1: Numerical Instability**
```python
# DON'T DO THIS
probs = torch.exp(logits) / torch.sum(torch.exp(logits), dim=-1, keepdim=True)
# Can overflow if logits are large!

# DO THIS
log_probs = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
probs = torch.exp(log_probs)
# Numerically stable!
```

### ❌ **Pitfall 2: Forgetting the Negative Sign**
```python
# WRONG
entropy = torch.sum(probs * log_probs, dim=-1)
# This gives NEGATIVE entropy!

# CORRECT
entropy = -torch.sum(probs * log_probs, dim=-1)
# Entropy should be positive
```

### ❌ **Pitfall 3: Wrong Dimension**
```python
# If you forget dim=-1
entropy = -torch.sum(probs * log_probs)
# This sums over ALL dimensions, giving a single scalar!

# Should be
entropy = -torch.sum(probs * log_probs, dim=-1)
# Sums only over vocabulary dimension
```

## Summary

**Entropy measures uncertainty:**
- Low entropy (≈0): Model is certain
- High entropy (≈log(vocab_size)): Model is uncertain

**Your implementation is correct and numerically stable:**
1. Uses logsumexp trick to avoid overflow
2. Computes probabilities from log-probabilities
3. Applies entropy formula with proper negative sign
4. Reduces over vocabulary dimension only

**Use cases:**
- Monitoring model confidence during training
- Detecting overfitting (entropy too low)
- Entropy regularization in RL
- Understanding model behavior

The entropy tells you: **"How sure is the model about what comes next?"**
