# Understanding `tokenize_prompt_and_output` for SFT

## What is this function for?

In **Supervised Fine-Tuning (SFT)**, we train a language model to generate specific outputs given specific prompts. This function prepares the data for that training process.

## Core Concept: Next Token Prediction with Masking

Language models work by predicting the next token given previous tokens. But in SFT, we have a special requirement:

- ✅ We WANT to train on generating the **output** tokens
- ❌ We DON'T want to train on generating the **prompt** tokens (the prompt is given!)

## The Four Main Steps

### Step 1: Tokenize Separately

```
Prompt: "What is 2+2?"  →  [10, 20, 30, 40]
Output: "The answer is 4"  →  [50, 60, 70, 80]
```

Why separate?
- We need to remember where the prompt ends and output begins
- This boundary is crucial for creating the response mask

### Step 2: Find Max Length & Initialize Tensors

```
Batch examples:
- Sample 1: 4 prompt tokens + 3 output tokens = 7 total
- Sample 2: 2 prompt tokens + 5 output tokens = 7 total
- Sample 3: 3 prompt tokens + 2 output tokens = 5 total → needs 2 padding tokens

Max length = 7
```

We create tensors of shape `(batch_size, max_length - 1)`:
- Why `-1`? Because of the next-token prediction shift (explained below)

### Step 3: Concatenate and Shift

```
Original sequence:  [10,  20,  30,  40,  50,  60,  70]
                     └──────prompt──────┘  └───output───┘

input_ids:          [10,  20,  30,  40,  50,  60]       ← sequence[:-1]
labels:             [20,  30,  40,  50,  60,  70]       ← sequence[1:]
```

This shift is how language models learn:
- At position 0: Given `[10]`, predict `20` ✓
- At position 1: Given `[10, 20]`, predict `30` ✓
- At position 2: Given `[10, 20, 30]`, predict `40` ✓
- At position 3: Given `[10, 20, 30, 40]`, predict `50` ✓
- And so on...

### Step 4: Create Response Mask

```
Position:        0    1    2    3    4    5
input_ids:      [10,  20,  30,  40,  50,  60]
labels:         [20,  30,  40,  50,  60,  70]
response_mask:  [ F,   F,   F,   T,   T,   T]
                 └─────prompt──┘  └──output──┘
```

**Key insight**: The mask is on the `labels`, not the `input_ids`!

- Position 0-2: Predicting prompt tokens `[20, 30, 40]` → mask = False
- Position 3-5: Predicting output tokens `[50, 60, 70]` → mask = True

Why `prompt_len - 1` as the start index?
- The labels are shifted by 1
- The first output token (`50`) appears at position 3 in the labels
- With prompt_len=4, the first output label is at position 4-1=3 ✓

## Example Walkthrough

```python
prompt = "Hello"  →  [100, 200]       # 2 tokens
output = "World"  →  [300, 400]       # 2 tokens
```

**Step-by-step:**

1. Concatenate: `[100, 200, 300, 400]`
2. Create shifted pairs:
   ```
   input_ids:     [100, 200, 300]
   labels:        [200, 300, 400]
   ```
3. Create mask:
   ```
   response_mask: [False, True, True]
   ```
   
**Why this mask?**
- Position 0: Given `[100]`, predict `200` (prompt token) → False
- Position 1: Given `[100, 200]`, predict `300` (output token!) → True
- Position 2: Given `[100, 200, 300]`, predict `400` (output token!) → True

## Training Loss Computation

During training, the loss is computed like this:

```python
# Model produces predictions for all positions
predictions = model(input_ids)

# Calculate loss for all positions
all_losses = cross_entropy(predictions, labels)

# But only use loss where response_mask is True!
masked_loss = all_losses[response_mask]
final_loss = masked_loss.mean()
```

This ensures:
- The model learns to generate the output
- The model is NOT penalized for "not predicting" the prompt correctly
- Training focuses on the completion task, not memorizing prompts

## Why This Matters

Without the response mask, the model would:
- Waste capacity learning to "predict" the prompt from itself
- Not focus enough on the actual task (generating the output)
- Potentially overfit to prompt patterns instead of learning reasoning

With the response mask:
- Training signal focuses on output generation
- Model learns "given this prompt → generate this specific output"
- More efficient use of model capacity and training data

## Common Pitfalls

1. **Wrong mask range**: Forgetting the shift by 1 leads to incorrect masking
2. **Including padding in mask**: Padding tokens should have mask=False
3. **Not tokenizing separately**: If you tokenize "prompt+output" as one string, you lose the boundary information
4. **Off-by-one errors**: The `-1` in `seq-1` and `prompt_len-1` are crucial!

## Summary

This function does three critical things:
1. **Tokenizes** prompts and outputs while preserving their boundary
2. **Shifts** sequences to create next-token prediction pairs
3. **Masks** to ensure training only on output tokens

All three are essential for effective supervised fine-tuning!
