# Visual Guide to tokenize_prompt_and_output

## The Big Picture: What happens to your data?

```
INPUT:
┌──────────────────────────────────────────────┐
│ prompt_strs = ["What is 2+2?"]               │
│ output_strs = ["The answer is 4"]            │
└──────────────────────────────────────────────┘
                    ↓
        ┌───────────────────────┐
        │  TOKENIZE SEPARATELY  │
        └───────────────────────┘
                    ↓
┌──────────────────────────────────────────────┐
│ prompt_ids = [10, 20, 30, 40]                │
│ output_ids = [50, 60, 70, 80]                │
└──────────────────────────────────────────────┘
                    ↓
        ┌───────────────────────┐
        │  CONCATENATE & PAD    │
        └───────────────────────┘
                    ↓
┌──────────────────────────────────────────────┐
│ sequence = [10, 20, 30, 40, 50, 60, 70, 80]  │
└──────────────────────────────────────────────┘
                    ↓
        ┌───────────────────────┐
        │  CREATE SHIFTED PAIRS │
        └───────────────────────┘
                    ↓
OUTPUT:
┌──────────────────────────────────────────────┐
│ input_ids:     [10, 20, 30, 40, 50, 60, 70]  │
│ labels:        [20, 30, 40, 50, 60, 70, 80]  │
│ response_mask: [ F,  F,  F,  T,  T,  T,  T]  │
└──────────────────────────────────────────────┘
```

## Detailed View: Position-by-Position

```
ORIGINAL TOKENS:
┌────┬────┬────┬────┬────┬────┬────┬────┐
│ 10 │ 20 │ 30 │ 40 │ 50 │ 60 │ 70 │ 80 │
└────┴────┴────┴────┴────┴────┴────┴────┘
  └─────PROMPT──────┘  └─────OUTPUT──────┘

AFTER SHIFTING (for next-token prediction):

Position:     0    1    2    3    4    5    6
         ┌────┬────┬────┬────┬────┬────┬────┐
input_ids│ 10 │ 20 │ 30 │ 40 │ 50 │ 60 │ 70 │
         ├────┼────┼────┼────┼────┼────┼────┤
labels   │ 20 │ 30 │ 40 │ 50 │ 60 │ 70 │ 80 │
         ├────┼────┼────┼────┼────┼────┼────┤
mask     │ F  │ F  │ F  │ T  │ T  │ T  │ T  │
         └────┴────┴────┴────┴────┴────┴────┘
           └──PROMPT──┘  └────OUTPUT────┘
```

## Why the Shift? Understanding Next-Token Prediction

```
The model learns by predicting what comes next:

Position 0:
    Input: [10]
    Predict: 20 ← this is a PROMPT token
    Mask: False (we don't train on this)
    
Position 1:
    Input: [10, 20]
    Predict: 30 ← still a PROMPT token
    Mask: False (we don't train on this)
    
Position 2:
    Input: [10, 20, 30]
    Predict: 40 ← still a PROMPT token
    Mask: False (we don't train on this)
    
Position 3:
    Input: [10, 20, 30, 40]
    Predict: 50 ← first OUTPUT token!
    Mask: True ✓ (START TRAINING HERE)
    
Position 4:
    Input: [10, 20, 30, 40, 50]
    Predict: 60 ← OUTPUT token
    Mask: True ✓ (train on this)
    
Position 5:
    Input: [10, 20, 30, 40, 50, 60]
    Predict: 70 ← OUTPUT token
    Mask: True ✓ (train on this)
    
Position 6:
    Input: [10, 20, 30, 40, 50, 60, 70]
    Predict: 80 ← OUTPUT token
    Mask: True ✓ (train on this)
```

## The Response Mask: Why prompt_len - 1?

This is the trickiest part! Let's see why the mask starts at `prompt_len - 1`:

```
BEFORE SHIFT:
Token:    [10,  20,  30,  40,  50,  60,  70,  80]
Position:   0    1    2    3    4    5    6    7
            └────prompt────┘  └─────output─────┘
            prompt_len = 4    output starts at index 4

AFTER SHIFT (removing first token):
Token:    [20,  30,  40,  50,  60,  70,  80]
Position:   0    1    2    3    4    5    6
            └─prompt──┘  └──────output──────┘
                          first output is now at index 3!

Why index 3?
- Original output started at index 4
- We removed the first token (10)
- So everything shifted left by 1
- New position = 4 - 1 = 3 ✓

That's why: response_mask[i, prompt_len-1 : prompt_len+output_len-1] = True
                                    ↑
                              This -1 accounts for the shift!
```

## Batch Processing with Padding

When you have multiple examples of different lengths:

```
Example 1: prompt(4) + output(3) = 7 tokens
Example 2: prompt(2) + output(5) = 7 tokens
Example 3: prompt(3) + output(2) = 5 tokens

Max length = 7

Example 3 needs padding:
Original:  [A, B, C, D, E]
Padded:    [A, B, C, D, E, PAD, PAD]

Then shift:
input_ids: [A, B, C, D, E, PAD]
labels:    [B, C, D, E, PAD, PAD]
mask:      [F, F, T, T, F, F]
            └prompt┘└output┘└pad┘
```

## Loss Computation During Training

```
┌─────────────────────────────────────────┐
│  1. Model Forward Pass                  │
│     input_ids → model → predictions     │
└─────────────────────────────────────────┘
                ↓
┌─────────────────────────────────────────┐
│  2. Calculate Loss Everywhere           │
│     loss = CrossEntropy(predictions,    │
│                         labels)         │
│                                         │
│     Position 0: loss = 0.5              │
│     Position 1: loss = 0.3              │
│     Position 2: loss = 0.8              │
│     Position 3: loss = 1.2  ←───┐       │
│     Position 4: loss = 0.9  ←───┤       │
│     Position 5: loss = 1.1  ←───┤       │
│     Position 6: loss = 0.7  ←───┘       │
└─────────────────────────────────────────┘
                ↓                  These are
┌─────────────────────────────────────────┐  output tokens
│  3. Apply Response Mask                 │  (mask=True)
│     Only keep losses where mask=True    │
│                                         │
│     Kept losses: [1.2, 0.9, 1.1, 0.7]  │
│     Final loss: mean([1.2, 0.9, ...])  │
│                = 0.975                  │
└─────────────────────────────────────────┘
                ↓
┌─────────────────────────────────────────┐
│  4. Backpropagation                     │
│     Gradients flow back ONLY for        │
│     output token predictions            │
└─────────────────────────────────────────┘
```

## Common Questions

### Q1: Why not just train on everything?
```
If we trained on prompt tokens:
  Given [10] predict [20]  ← "What" → "is"
  Given [10,20] predict [30]  ← "What is" → "2"
  
This teaches the model to predict the prompt from itself!
That's useless - we GIVE the model the prompt.

We only want:
  Given [10,20,30,40] predict [50]  ← "What is 2+2?" → "The"
  Given [10,20,30,40,50] predict [60]  ← "... The" → "answer"
```

### Q2: What if prompt and output have different lengths in the batch?
```
That's fine! The mask handles it:

Sample 1: prompt(2) + output(5)
  mask: [F, T, T, T, T, T]
        └p┘└───output────┘

Sample 2: prompt(4) + output(3)
  mask: [F, F, F, T, T, T]
        └─prompt─┘└output┘

Each sample gets its own correct mask!
```

### Q3: Why dtype=torch.int32 for IDs but torch.bool for mask?
```
input_ids & labels: These are token IDs (integers)
  Example: [100, 200, 300] → actual vocabulary indices
  dtype=torch.int32 (numbers)

response_mask: This is a boolean flag (True/False)
  Example: [False, True, True] → just on/off
  dtype=torch.bool (more memory efficient for binary values)
```

## Summary: The Three Key Outputs

```
input_ids: 
    What the model SEES as context
    Shape: (batch_size, max_seq_len - 1)
    
labels:
    What the model should PREDICT (ground truth)
    Shape: (batch_size, max_seq_len - 1)
    
response_mask:
    Which predictions we CARE ABOUT (only output)
    Shape: (batch_size, max_seq_len - 1)
```

These three tensors work together to train the model to:
**"Given a prompt, generate the correct output"**

Not just: "Predict next token" (too broad)
But specifically: "Predict next token IN THE OUTPUT PART" (focused training!)
