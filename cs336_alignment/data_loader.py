import json
import os
import random
import torch

from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizerBase


# Alpaca prompt template: wraps (prompt, response) pairs into a formatted instruction string
propmt_template = \
"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{prompt}

### Response:
{response}"""   

class SftDataset(Dataset):
  """
  PyTorch Dataset for Supervised Fine-Tuning (SFT) with packed sequences.
  
  This dataset:
  1. Loads (prompt, response) pairs from a JSONL file
  2. Formats them using the Alpaca template
  3. Concatenates all documents into a single token sequence with delimiters
  4. Packs the tokens into fixed-length chunks (seq_length)
  5. Creates input_ids and labels for language modeling (labels are input_ids shifted by 1)
  """
  def __init__(self, tokenizer: PreTrainedTokenizerBase, dataset_path: str | os.PathLike, seq_length: int, shuffle: bool):
    # Step 1: Load all (prompt, response) pairs and format them using Alpaca template
    docs = []
    with open(dataset_path, "r", encoding="utf-8") as f:
      for line in f:
        data = json.loads(line)  # Each line is a JSON object with "prompt" and "response" keys
        # Format each pair into an instruction-following string
        docs.append(propmt_template.format(prompt=data["prompt"], response=data["response"]))
    
    # Step 2: Optionally shuffle documents before concatenation
    # This ensures different ordering across training runs when shuffle=True
    if shuffle:
      random.shuffle(docs)

    # Step 3: Concatenate all documents into a single string with delimiters
    # Using <|end_of_text|><|begin_of_text|> as delimiter (Llama 3 convention)
    # This marks boundaries between different instruction-response pairs
    text = "<|end_of_text|><|begin_of_text|>".join(docs)
    
    # Step 4: Tokenize the entire concatenated text into token IDs
    all_tokens = tokenizer.encode(text)

    # Step 5: Pack tokens into fixed-length sequences (packing strategy)
    # Calculate how many complete sequences of length seq_length we can create
    # We need (len-1) because we create labels by shifting tokens by 1 position
    length = (len(all_tokens)-1) // seq_length
    
    # Step 6: Create input_ids by taking the first (seq_length * length) tokens
    # Reshape into (length, seq_length) - each row is one training example
    self.input_ids = torch.tensor(all_tokens[0:seq_length * length], dtype=torch.int64).reshape(length, seq_length)
    
    # Step 7: Create labels by taking tokens starting from position 1 (shifted by 1)
    # This is for next-token prediction: given tokens [0:seq_length], predict tokens [1:seq_length+1]
    # For example: input_ids=[0,1,2,3], labels=[1,2,3,4]
    self.labels = torch.tensor(all_tokens[1:seq_length * length+1], dtype=torch.int64).reshape(length, seq_length)

  def __len__(self):
    """Returns the number of packed sequences in the dataset."""
    return self.input_ids.shape[0]
  
  def __getitem__(self, i):
    """
    Returns the i-th training example as a dictionary.
    
    Returns:
      dict with keys:
        - "input_ids": tensor of shape (seq_length,) - the input token sequence
        - "labels": tensor of shape (seq_length,) - the target tokens (input_ids shifted by 1)
    """
    return {"input_ids":self.input_ids[i], "labels":self.labels[i]}

def iterate_batches(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
):
    """
    Creates a PyTorch DataLoader that yields batches from the dataset.
    
    This function wraps the dataset in a DataLoader which:
    1. Groups examples into batches of size batch_size
    2. Optionally shuffles the order of examples (when shuffle=True)
    3. Returns an iterable that yields one batch at a time
    4. Automatically handles collation (stacking tensors into batch dimension)
    
    Args:
      dataset: PyTorch Dataset to batch
      batch_size: Number of examples per batch (B in the assignment description)
      shuffle: Whether to shuffle examples before batching
    
    Returns:
      DataLoader that yields batches where each batch is a dict with:
        - "input_ids": tensor of shape (batch_size, seq_length)
        - "labels": tensor of shape (batch_size, seq_length)
    
    Note: The last batch may have fewer than batch_size examples if the dataset
          size is not evenly divisible by batch_size.
    """
    data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)
    return data_loader