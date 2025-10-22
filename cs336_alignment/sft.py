import torch
from transformers import PreTrainedTokenizer, PreTrainedModel

def tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: PreTrainedTokenizer,
) -> dict[str, torch.Tensor]:
    """Tokenize the prompt and output strings, and construct a mask that is 1
    for the response tokens and 0 for other tokens (prompt or padding).

    Args:
        prompt_strs: list[str], the prompt strings.
        output_strs: list[str], the output strings.
        tokenizer: PreTrainedTokenizer, the tokenizer to use.

    Returns:
        dict[str, torch.Tensor]:
            "input_ids": torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1):
                the tokenized prompt and output strings, with the final token sliced off.
            "labels": torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1):
                shifted input_ids (i.e., the input_ids without the first token).
            "response_mask": torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1):
                a mask on the response tokens in `labels`.
    """
    # Verify that each prompt has a corresponding output
    assert len(prompt_strs) == len(output_strs)

    # =============================================================================
    # STEP 1: Tokenize prompts and outputs separately
    # =============================================================================
    # In SFT (Supervised Fine-Tuning), we want to train the model to generate
    # the output given the prompt. To do this, we need to:
    # 1. Tokenize the prompt and output separately (not as a single string)
    # 2. Keep them separate so we can create a mask later
    #
    # Why tokenize separately?
    # - We need to know where the prompt ends and the output begins
    # - During training, we only compute loss on the OUTPUT tokens, not the prompt
    # - This is called "causal language modeling with masking"
    
    prompt_and_output_list = []
    for i, (prompt_str, output_str) in enumerate(zip(prompt_strs, output_strs, strict=True)):
        # Tokenize prompt: "Hello, world!" -> ["Hello", ",", " world", "!"] -> [123, 45, 678, 90]
        # We use tokenizer.tokenize() to split text into tokens (subwords)
        # Then convert_tokens_to_ids() converts them to integer IDs the model understands
        prompt_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(prompt_str))
        
        # Tokenize output the same way
        output_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(output_str))
        
        # Store as a tuple (prompt_ids, output_ids) to remember the boundary
        # Example: ([123, 45, 678, 90], [111, 222, 333])
        # Store as a tuple (prompt_ids, output_ids) to remember the boundary
        # Example: ([123, 45, 678, 90], [111, 222, 333])
        prompt_and_output_list.append((prompt_ids, output_ids))

    # =============================================================================
    # STEP 2: Find the maximum sequence length for padding
    # =============================================================================
    # In a batch, different examples have different lengths. We need to pad them
    # to the same length so we can process them together as a tensor.
    #
    # Example batch:
    # - Sample 1: prompt (4 tokens) + output (3 tokens) = 7 tokens total
    # - Sample 2: prompt (2 tokens) + output (5 tokens) = 7 tokens total  
    # - Sample 3: prompt (3 tokens) + output (2 tokens) = 5 tokens total
    # -> max_length = 7, so Sample 3 needs 2 padding tokens
    
    batch_size = len(prompt_and_output_list)
    
    # Find the longest sequence in the batch
    # key=lambda defines how to measure "longest": sum of prompt + output lengths
    max_element = max(prompt_and_output_list, key = lambda x: len(x[0]) + len(x[1]))
    
    # seq is the maximum total length (prompt + output) in this batch
    seq = len(max_element[0]) + len(max_element[1])

    # =============================================================================
    # STEP 3: Initialize tensors for the batch
    # =============================================================================
    # We create three tensors, all with shape (batch_size, seq - 1):
    #
    # Why "seq - 1"?
    # - In language modeling, we predict the NEXT token given previous tokens
    # - If we have [A, B, C, D], inputs are [A, B, C] and labels are [B, C, D]
    # - So we need seq-1 positions for both input and labels
    #
    # Visual example with seq=5:
    # Original:     [A,  B,  C,  D,  E]
    # input_ids:    [A,  B,  C,  D]      <- first 4 tokens (seq-1)
    # labels:       [B,  C,  D,  E]      <- shifted by 1 (also seq-1)
    
    input_ids = torch.zeros(batch_size, seq - 1, dtype=torch.int32)
    labels = torch.zeros(batch_size, seq - 1, dtype=torch.int32)
    
    # response_mask is boolean: True for output tokens, False for prompt/padding tokens
    # This tells the loss function which tokens to compute loss on
    # response_mask is boolean: True for output tokens, False for prompt/padding tokens
    # This tells the loss function which tokens to compute loss on
    response_mask = torch.zeros(batch_size, seq - 1, dtype=torch.bool)

    # =============================================================================
    # STEP 4: Fill in the tensors for each sample in the batch
    # =============================================================================
    for i, prompt_and_output in enumerate(prompt_and_output_list):
        # Extract lengths - we need these to know where the output begins
        prompt_len = len(prompt_and_output[0])
        output_len = len(prompt_and_output[1])
        
        # -------------------------------------------------------------------------
        # Concatenate prompt + output + padding into a single sequence
        # -------------------------------------------------------------------------
        # Example:
        # prompt_ids = [10, 20, 30]          (length 3)
        # output_ids = [40, 50]              (length 2)
        # seq = 7 (max in batch)
        # -> sequence = [10, 20, 30, 40, 50, PAD, PAD]
        #
        # The padding ensures all sequences in the batch have the same length
        sequence = prompt_and_output[0] + prompt_and_output[1] + [tokenizer.pad_token_id] * (seq - prompt_len - output_len)
        
        # -------------------------------------------------------------------------
        # Create input_ids and labels with the "next token prediction" shift
        # -------------------------------------------------------------------------
        # Language models predict the NEXT token given previous context
        #
        # Original sequence:  [10,  20,  30,  40,  50, PAD, PAD]
        # input_ids:          [10,  20,  30,  40,  50, PAD]       <- sequence[:-1] (all but last)
        # labels:             [20,  30,  40,  50, PAD, PAD]       <- sequence[1:]  (all but first)
        #
        # Position 0: Given [10], predict [20] ✓
        # Position 1: Given [10, 20], predict [30] ✓
        # Position 2: Given [10, 20, 30], predict [40] ✓
        # Position 3: Given [10, 20, 30, 40], predict [50] ✓
        # Position 4: Given [10, 20, 30, 40, 50], predict [PAD] ✓
        #
        # This is why we need "seq - 1" length: we lose one position due to the shift
        input_ids[i, :seq-1] = torch.tensor(sequence[:-1])
        labels[i, :seq-1] = torch.tensor(sequence[1:])
        
        # -------------------------------------------------------------------------
        # Create response_mask: True only for OUTPUT tokens in the labels
        # -------------------------------------------------------------------------
        # Why do we need this mask?
        # - In SFT, we want to teach the model to generate the OUTPUT
        # - We DON'T want to compute loss on the PROMPT tokens
        # - The model should learn "given this prompt, generate this output"
        #
        # Example with our sequence:
        # Position:           0    1    2    3    4    5
        # input_ids:         [10,  20,  30,  40,  50, PAD]
        # labels:            [20,  30,  40,  50, PAD, PAD]
        # response_mask:     [ F,   F,   T,   T,   F,   F]
        #                     ^    ^    ^    ^    ^    ^
        #                     |    |    |    |    |    |
        #                   prompt  |  output |  padding
        #
        # Explanation:
        # - Position 0-1: These are predicting prompt tokens [20, 30], so mask=False
        # - Position 2-3: These are predicting output tokens [40, 50], so mask=True
        # - Position 4-5: These are predicting padding, so mask=False
        #
        # Why "prompt_len - 1" as start?
        # - labels are shifted by 1 from input_ids
        # - The first OUTPUT token in labels appears at position (prompt_len - 1)
        # - It continues for output_len positions
        #
        # Let's verify with our example:
        # - prompt_len = 3, output_len = 2
        # - Mask range: [3-1 : 3+2-1] = [2:4]
        # - Positions 2 and 3 are marked True ✓
        response_mask[i, prompt_len-1:prompt_len + output_len - 1 ] = True
    
    # =============================================================================
    # RETURN: Dictionary with input_ids, labels, and response_mask
    # =============================================================================
    # These three tensors work together:
    # - input_ids: What the model sees as input context
    # - labels: What the model should predict (ground truth next tokens)
    # - response_mask: Which predictions we actually care about (only the output)
    #
    # During training:
    # 1. Model gets input_ids, produces predictions for next tokens
    # 2. Compare predictions to labels
    # 3. Only compute loss where response_mask is True
    # 4. Backpropagate only on output tokens, not prompt tokens
    return {"input_ids": input_ids, "labels": labels, "response_mask": response_mask}