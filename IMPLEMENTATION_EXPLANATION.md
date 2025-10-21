# SFT Dataset Implementation Explanation

## Overview
Your `SftDataset` implements a **packed sequence data loader** for Supervised Fine-Tuning (SFT). This is a common technique in language model training that maximizes GPU utilization by eliminating padding tokens.

## Key Concepts

### 1. **Packing Strategy**
Instead of having variable-length sequences with padding, all documents are:
- Concatenated into one long sequence
- Split into fixed-length chunks
- No padding tokens needed (maximizes GPU throughput)

### 2. **Example Flow**

Given these inputs:
```python
# Input data (sft_sample.jsonl):
{"prompt": "What is 2+2?", "response": "4"}
{"prompt": "Name a color", "response": "Blue"}

# Parameters:
seq_length = 4
shuffle = False
```

**Step-by-step process:**

#### Step 1: Format with Alpaca Template
```
Doc 1: "Below is an instruction...\n### Instruction:\nWhat is 2+2?\n### Response:\n4"
Doc 2: "Below is an instruction...\n### Instruction:\nName a color\n### Response:\nBlue"
```

#### Step 2: Concatenate with Delimiters
```
"<|end_of_text|><|begin_of_text|>" joins documents
Result: "Doc1<|end_of_text|><|begin_of_text|>Doc2"
```

#### Step 3: Tokenize
```
all_tokens = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, ...]
```

#### Step 4: Pack into Fixed-Length Sequences
```
Given seq_length = 4:
- Calculate: length = (len(all_tokens) - 1) // 4
- Create chunks: [[0,1,2,3], [4,5,6,7], [8,9,10,11], ...]
```

#### Step 5: Create Input-Label Pairs
```
input_ids:  [[0, 1, 2, 3], [4, 5, 6, 7], ...]  # Tokens at positions [i:i+seq_length]
labels:     [[1, 2, 3, 4], [5, 6, 7, 8], ...]  # Tokens at positions [i+1:i+seq_length+1]
```

This creates the **next-token prediction** objective:
- Given tokens [0,1,2,3], predict [1,2,3,4]
- The model learns to predict token[i+1] given tokens[0:i+1]

## Code Walkthrough

### `SftDataset.__init__()`

```python
# 1. Load and format documents
docs = []
for line in jsonl_file:
    docs.append(alpaca_template.format(prompt=..., response=...))

# 2. Optionally shuffle (different order each run)
if shuffle:
    random.shuffle(docs)

# 3. Concatenate all documents with delimiters
text = "<|end_of_text|><|begin_of_text|>".join(docs)

# 4. Tokenize entire text
all_tokens = tokenizer.encode(text)

# 5. Calculate how many complete sequences we can create
length = (len(all_tokens) - 1) // seq_length

# 6. Create input_ids: [0:seq_length*length] reshaped
self.input_ids = tensor(all_tokens[0:seq_length * length]).reshape(length, seq_length)

# 7. Create labels: [1:seq_length*length+1] reshaped (shifted by 1)
self.labels = tensor(all_tokens[1:seq_length * length + 1]).reshape(length, seq_length)
```

### Why `(len(all_tokens) - 1) // seq_length`?

- We need pairs: (input, label) where label = input shifted by 1
- For `seq_length` input tokens, we need `seq_length + 1` total tokens
- If we have N tokens, we can make `(N - 1) // seq_length` complete sequences

**Example:**
```
all_tokens = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # 10 tokens
seq_length = 4

length = (10 - 1) // 4 = 9 // 4 = 2

Sequence 1:
  input_ids: [0, 1, 2, 3]
  labels:    [1, 2, 3, 4]

Sequence 2:
  input_ids: [4, 5, 6, 7]
  labels:    [5, 6, 7, 8]

# Token 9 is dropped (incomplete sequence)
```

### `iterate_batches()`

This function wraps the dataset in a PyTorch `DataLoader`:

```python
DataLoader(dataset, batch_size=8, shuffle=True)
```

**What it does:**
1. Groups individual sequences into batches
2. Converts from `(seq_length,)` to `(batch_size, seq_length)`
3. Optionally shuffles the order of sequences
4. Handles the last batch (may be smaller than batch_size)

**Example:**
```python
# Individual sequences from dataset:
seq_0 = {"input_ids": [0,1,2,3], "labels": [1,2,3,4]}
seq_1 = {"input_ids": [4,5,6,7], "labels": [5,6,7,8]}
seq_2 = {"input_ids": [8,9,10,11], "labels": [9,10,11,12]}

# After DataLoader with batch_size=2:
batch_0 = {
    "input_ids": [[0,1,2,3], [4,5,6,7]],  # shape: (2, 4)
    "labels": [[1,2,3,4], [5,6,7,8]]      # shape: (2, 4)
}
batch_1 = {
    "input_ids": [[8,9,10,11]],           # shape: (1, 4) - last batch
    "labels": [[9,10,11,12]]              # shape: (1, 4)
}
```

## Why Packing?

**Traditional approach (with padding):**
```
Doc1: [1, 2, 3, <pad>, <pad>]     # 2 padding tokens
Doc2: [4, 5, <pad>, <pad>, <pad>] # 3 padding tokens
```
- Wasted computation on padding tokens
- Lower GPU utilization

**Packing approach:**
```
All docs: [1, 2, 3, <delimiter>, 4, 5, ...]
Chunks:   [[1, 2, 3, <del>], [4, 5, ...], ...]
```
- No padding tokens
- Maximum GPU utilization
- More efficient training

## Test Results
✅ `test_packed_sft_dataset` - Verifies correct tokenization and packing
✅ `test_iterate_batches` - Verifies correct batching and DataLoader behavior

Your implementation correctly handles all the requirements for packed SFT training!
