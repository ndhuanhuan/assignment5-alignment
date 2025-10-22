# Entropy Visual Cheat Sheet

## The Formula in One Line

```
H(p) = -∑ p(x) × log p(x)
       ↑  ↑      ↑
       │  │      └─ How "surprising" is x?
       │  └──────── How likely is x?
       └─────────── Sum over all outcomes, negate at end
```

## The Three-Step Process

```
┌──────────────────────────────────────────────────────┐
│ INPUT: Logits [2.0, 1.0, 0.5, 0.3]                   │
└──────────────────────────────────────────────────────┘
                        ↓
┌──────────────────────────────────────────────────────┐
│ STEP 1: Logits → Log-Probabilities                   │
│                                                       │
│   logsumexp = 2.57                                   │
│   log_probs = [2.0, 1.0, 0.5, 0.3] - 2.57           │
│             = [-0.57, -1.57, -2.07, -2.27]           │
└──────────────────────────────────────────────────────┘
                        ↓
┌──────────────────────────────────────────────────────┐
│ STEP 2: Log-Probabilities → Probabilities            │
│                                                       │
│   probs = exp(log_probs)                             │
│         = [0.565, 0.208, 0.126, 0.103]               │
│         (sum = 1.002 ≈ 1.0 ✓)                        │
└──────────────────────────────────────────────────────┘
                        ↓
┌──────────────────────────────────────────────────────┐
│ STEP 3: Compute Entropy                              │
│                                                       │
│   probs × log_probs = [-0.322, -0.327, -0.261, -0.234]│
│   sum = -1.144                                       │
│   entropy = -(-1.144) = 1.144                        │
└──────────────────────────────────────────────────────┘
                        ↓
┌──────────────────────────────────────────────────────┐
│ OUTPUT: Entropy = 1.144                              │
│ (82.5% of max entropy for 4 tokens)                  │
└──────────────────────────────────────────────────────┘
```

## Entropy Spectrum

```
    0.0                      0.5                      1.0                     1.39
     │━━━━━━━━━━━━━━━━━━━━━━━│━━━━━━━━━━━━━━━━━━━━━━━│━━━━━━━━━━━━━━━━━━━━━━│
     ↑                        ↑                        ↑                       ↑
  CERTAIN              FAIRLY CERTAIN            UNCERTAIN           COMPLETELY RANDOM
  
  [0.99, 0.01, 0.00, 0.00]    [0.70, 0.20, 0.07, 0.03]    [0.40, 0.30, 0.20, 0.10]    [0.25, 0.25, 0.25, 0.25]
  
  "It's definitely 'A'"       "Probably 'A'"              "Maybe 'A' or 'B'?"         "No clue!"
```

## Quick Reference Table

| Scenario | Probabilities | Entropy | Meaning |
|----------|--------------|---------|---------|
| **Completely certain** | [1.0, 0.0, 0.0, 0.0] | 0.0 | "100% sure" |
| **Very confident** | [0.9, 0.05, 0.03, 0.02] | 0.47 | "Pretty sure" |
| **Somewhat confident** | [0.6, 0.2, 0.1, 0.1] | 1.18 | "Leaning toward..." |
| **Uncertain** | [0.4, 0.3, 0.2, 0.1] | 1.28 | "Could be several" |
| **Completely random** | [0.25, 0.25, 0.25, 0.25] | 1.39 | "No idea!" |

*Note: For vocab_size=V, max entropy = log(V)*

## Code Pattern

```python
# Pattern you'll see everywhere in ML code:

# Convert raw outputs to log-probabilities (stable)
log_probs = logits - torch.logsumexp(logits, dim=-1, keepdim=True)

# Get probabilities
probs = torch.exp(log_probs)

# Compute entropy
entropy = -torch.sum(probs * log_probs, dim=-1)
```

## Common Values for Language Models

```
Vocabulary size: 50,000 tokens
Maximum possible entropy: log(50,000) ≈ 10.82

Typical values during generation:
┌────────────────┬──────────┬─────────────────────────┐
│ Scenario       │ Entropy  │ What's happening?       │
├────────────────┼──────────┼─────────────────────────┤
│ Very confident │  0.5-2.0 │ "I know what comes next"│
│ Normal text    │  2.0-5.0 │ "Typical uncertainty"   │
│ Confused       │  5.0-8.0 │ "Many options possible" │
│ Random/broken  │  8.0+    │ "Model is lost"         │
└────────────────┴──────────┴─────────────────────────┘
```

## Temperature and Entropy

```python
# Temperature controls entropy:

# Low temperature (0.5) → Low entropy
logits_cold = logits / 0.5  # Amplify differences
probs = [0.95, 0.03, 0.01, 0.01]  # Sharp distribution
entropy ≈ 0.3  # Very certain

# Normal temperature (1.0)
logits_normal = logits / 1.0
probs = [0.70, 0.15, 0.10, 0.05]  # Moderate distribution
entropy ≈ 1.0  # Moderately uncertain

# High temperature (2.0) → High entropy
logits_hot = logits / 2.0  # Flatten differences
probs = [0.40, 0.30, 0.20, 0.10]  # Flat distribution
entropy ≈ 1.3  # Very uncertain
```

## When to Use Entropy

### ✅ **Use entropy to:**
- Monitor training progress
- Detect overfitting (entropy too low on training set)
- Measure model confidence
- Implement exploration in RL
- Debug weird model behavior

### ❌ **Don't use entropy for:**
- Measuring model accuracy (use cross-entropy loss instead)
- Selecting the best token (use argmax or sampling instead)
- Comparing different models (use perplexity instead)

## The Intuition

Think of entropy as:
- **A thermometer** for model confidence
- **High** = hot = random = uncertain = "model is guessing"
- **Low** = cold = deterministic = certain = "model is sure"

## Why Log-Sum-Exp?

```python
# The problem with naive approach:
exp(1000) = OVERFLOW!  # Infinity
exp(-1000) = UNDERFLOW!  # Zero

# Log-sum-exp trick:
logsumexp([1000, 999, 998]) = 1000 + log(1 + e^(-1) + e^(-2))
                             = 1000 + log(1 + 0.368 + 0.135)
                             = 1000 + 0.411
                             = 1000.411  # No overflow!
```

## Memory Aid

**ENTROPY = UNCERTAINTY**

- 📌 **E**xpected surprise
- 📌 **N**ot knowing what's next
- 📌 **T**otal randomness measure
- 📌 **R**andomness quantified
- 📌 **O**utcome unpredictability
- 📌 **P**robability spread measure
- 📌 **Y**ou don't know!

The more spread out the probabilities, the higher the entropy!
