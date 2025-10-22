# `get_response_log_probs` Visual Cheat Sheet

## The Three-Step Process

```
┌─────────────────────────────────────────────────────────────────┐
│ STEP 1: FORWARD PASS                                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  input_ids: [10, 20, 30, 40]                                   │
│              ↓                                                   │
│         ┌────────┐                                              │
│         │ MODEL  │                                              │
│         └────────┘                                              │
│              ↓                                                   │
│  logits: [batch, seq, vocab_size]                              │
│          [[2.1, -0.5, 1.3, ..., 0.8],  ← 50k values            │
│           [1.8, 0.2, -0.3, ..., 1.5],  ← 50k values            │
│           [0.5, 1.1, 2.3, ..., -0.2],  ← 50k values            │
│           [1.2, -0.8, 0.9, ..., 1.1]]  ← 50k values            │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────────────┐
│ STEP 2: LOG-SOFTMAX                                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  logits → log_softmax(logits, dim=-1)                          │
│                                                                  │
│  Raw scores:     [2.1, -0.5, 1.3, 0.8, ...]                    │
│                   ↓                                             │
│  Softmax:        [0.42, 0.03, 0.19, 0.11, ...]  (sum=1.0)      │
│                   ↓                                             │
│  Log-Softmax:    [-0.87, -3.51, -1.66, -2.20, ...]             │
│                                                                  │
│  Result: [batch, seq, vocab_size]                              │
│          All log-probabilities for all tokens                   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────────────┐
│ STEP 3: GATHER (Extract Label Probabilities)                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  labels: [20, 30, 40, 50]                                      │
│           ↓   ↓   ↓   ↓                                         │
│  Position 0: Extract log_softmax[0, 20] = -0.87               │
│  Position 1: Extract log_softmax[1, 30] = -1.23               │
│  Position 2: Extract log_softmax[2, 40] = -2.15               │
│  Position 3: Extract log_softmax[3, 50] = -0.92               │
│                                                                  │
│  Result: log_probs = [-0.87, -1.23, -2.15, -0.92]             │
│          [batch, seq]                                           │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## torch.gather Visualization

```
What we have:
┌───────────────────────────────────────────────────┐
│ log_softmax: [batch, seq, vocab_size]             │
│                                                    │
│ Position 0: [-0.87, -3.51, -1.66, ..., -1.23]    │
│                                     ↑              │
│                                   index 20         │
│                                                    │
│ Position 1: [-1.45, -2.10, -0.85, ..., -2.31]    │
│                              ↑                     │
│                            index 30                │
│                                                    │
│ Position 2: [-0.95, -1.22, -2.15, ..., -1.88]    │
│                          ↑                         │
│                        index 40                    │
└───────────────────────────────────────────────────┘

What we want:
┌────────────────────────────────────┐
│ labels: [20, 30, 40]               │
│          ↓   ↓   ↓                 │
│ Extract values at these indices    │
└────────────────────────────────────┘

How torch.gather works:
┌─────────────────────────────────────────────────────┐
│ torch.gather(log_softmax, dim=-1, index=labels)     │
│                                                      │
│ From position 0's 50k values, get value at index 20 │
│ From position 1's 50k values, get value at index 30 │
│ From position 2's 50k values, get value at index 40 │
└─────────────────────────────────────────────────────┘

Result:
┌──────────────────────────────────┐
│ log_probs: [-1.23, -0.85, -2.15] │
└──────────────────────────────────┘
```

## Shape Transformations

```
input_ids:           [batch_size, seq_len]
                     [    2,         4    ]
                            ↓
                    ┌───────────────┐
                    │ model.forward │
                    └───────────────┘
                            ↓
logits:              [batch_size, seq_len, vocab_size]
                     [    2,         4,       50000   ]
                            ↓
                    ┌───────────────┐
                    │  log_softmax  │
                    └───────────────┘
                            ↓
log_softmax:         [batch_size, seq_len, vocab_size]
                     [    2,         4,       50000   ]
                            ↓
labels.unsqueeze:    [batch_size, seq_len, 1]
                     [    2,         4,     1]
                            ↓
                    ┌───────────────┐
                    │  torch.gather │
                    └───────────────┘
                            ↓
gathered:            [batch_size, seq_len, 1]
                     [    2,         4,     1]
                            ↓
                    ┌───────────────┐
                    │   .squeeze    │
                    └───────────────┘
                            ↓
log_probs:           [batch_size, seq_len]
                     [    2,         4    ]
```

## Log-Probability Interpretation

```
┌─────────────────┬──────────────┬────────────────┬─────────────────┐
│ log_prob        │ prob         │ percentage     │ interpretation  │
├─────────────────┼──────────────┼────────────────┼─────────────────┤
│   0.0           │ 1.00         │ 100%           │ Perfect!        │
│  -0.1           │ 0.90         │  90%           │ Very confident  │
│  -0.5           │ 0.61         │  61%           │ Confident       │
│  -1.0           │ 0.37         │  37%           │ Moderate        │
│  -2.0           │ 0.14         │  14%           │ Low             │
│  -3.0           │ 0.05         │   5%           │ Very low        │
│  -5.0           │ 0.007        │ 0.7%           │ Extremely low   │
│ -10.0           │ 0.00005      │ 0.005%         │ Nearly zero     │
└─────────────────┴──────────────┴────────────────┴─────────────────┘

Training goal: Make log_probs as close to 0 as possible
               (i.e., make probabilities as close to 1.0 as possible)
```

## Example Walkthrough

```
Sentence: "The cat sat"
Tokenized: [10=The, 20=cat, 30=sat, 40=on]

input_ids: [10,  20,  30]  ← What model sees
labels:    [20,  30,  40]  ← What model should predict
           ─┬─  ─┬─  ─┬─
            │    │    │
            │    │    └─ Given "The cat sat", predict "on"
            │    └────── Given "The cat", predict "sat"
            └─────────── Given "The", predict "cat"

┌──────────────────────────────────────────────────────────┐
│ Position 0: Given [The], predict [cat]                   │
├──────────────────────────────────────────────────────────┤
│ Model outputs 50k logits: [2.1, ..., 1.8, ..., 0.3]     │
│ After log_softmax: [-0.5, ..., -0.8, ..., -2.3]         │
│ Label is 20, so extract: log_softmax[0, 20] = -0.8      │
│ Interpretation: P(cat|The) = exp(-0.8) = 0.45 (45%)     │
└──────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────┐
│ Position 1: Given [The, cat], predict [sat]              │
├──────────────────────────────────────────────────────────┤
│ Model outputs 50k logits: [1.5, ..., 2.3, ..., 0.7]     │
│ After log_softmax: [-1.2, ..., -0.4, ..., -1.9]         │
│ Label is 30, so extract: log_softmax[1, 30] = -0.4      │
│ Interpretation: P(sat|The,cat) = exp(-0.4) = 0.67 (67%) │
└──────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────┐
│ Position 2: Given [The, cat, sat], predict [on]          │
├──────────────────────────────────────────────────────────┤
│ Model outputs 50k logits: [0.8, ..., 1.2, ..., 2.1]     │
│ After log_softmax: [-2.1, ..., -1.7, ..., -0.8]         │
│ Label is 40, so extract: log_softmax[2, 40] = -0.8      │
│ Interpretation: P(on|The,cat,sat) = exp(-0.8) = 0.45    │
└──────────────────────────────────────────────────────────┘

Final output:
log_probs = [-0.8, -0.4, -0.8]

Average: (-0.8 + -0.4 + -0.8) / 3 = -0.67
Loss = -(-0.67) = 0.67
```

## Training Loss Connection

```
┌─────────────────────────────────────────────────────┐
│ TRAINING LOOP                                        │
├─────────────────────────────────────────────────────┤
│                                                      │
│ 1. Get log-probs                                    │
│    outputs = get_response_log_probs(model, ...)     │
│    log_probs = outputs['log_probs']                 │
│    → Shape: [batch, seq]                            │
│                                                      │
│ 2. Apply response mask (only train on output)      │
│    masked_log_probs = log_probs * response_mask     │
│    → Zeros out prompt and padding positions         │
│                                                      │
│ 3. Compute loss (negative log-likelihood)          │
│    loss = -masked_log_probs.sum() / mask.sum()     │
│                                                      │
│ 4. Backprop                                         │
│    loss.backward()                                  │
│    optimizer.step()                                 │
│                                                      │
└─────────────────────────────────────────────────────┘

Goal: Maximize log_probs = Minimize loss
```

## Why Log-Probabilities?

```
┌──────────────────┬─────────────────┬─────────────────────┐
│ Aspect           │ Probabilities   │ Log-Probabilities   │
├──────────────────┼─────────────────┼─────────────────────┤
│ Range            │ [0, 1]          │ [-∞, 0]             │
│ Multiplication   │ p1 × p2 × p3    │ log_p1 + log_p2 + 3 │
│ Numerical        │ Can underflow   │ Stable              │
│ Gradient         │ Can vanish      │ Better behaved      │
│ Loss function    │ Need -log       │ Already in log      │
└──────────────────┴─────────────────┴─────────────────────┘

Example of underflow problem:
  prob = 0.001 × 0.002 × 0.003 = 0.000000006  (very small!)
  log_prob = -6.9 + -6.2 + -5.8 = -18.9  (manageable)
```

## Quick Reference

```python
# Basic usage
outputs = get_response_log_probs(
    model=my_model,
    input_ids=input_ids,      # [batch, seq]
    labels=labels,             # [batch, seq]
    return_token_entropy=False
)

log_probs = outputs['log_probs']  # [batch, seq]

# With entropy
outputs = get_response_log_probs(
    model=my_model,
    input_ids=input_ids,
    labels=labels,
    return_token_entropy=True
)

log_probs = outputs['log_probs']          # [batch, seq]
entropy = outputs['token_entropy']        # [batch, seq]

# Compute loss
loss = -(log_probs * response_mask).sum() / response_mask.sum()
```

## Memory Aid

**GET RESPONSE LOG PROBS**

- **G**et model predictions
- **E**xtract logits for all tokens
- **T**ransform with log-softmax

- **R**etrieve specific token probabilities
- **E**ach position gets one value
- **S**elect using labels as indices
- **P**robabilities in log space
- **O**utput shape: [batch, seq]
- **N**o masking applied
- **S**eparate entropy optional
- **E**ssential for training

**LOG PROBS = How likely were the actual tokens?**
