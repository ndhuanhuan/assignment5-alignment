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


def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Compute the entropy of a probability distribution given logits.

    Args:
        logits: torch.Tensor of shape (..., vocab_size), the logits output by the model.

    Returns:
        torch.Tensor of shape (...), the entropy of the distribution at each position.
    """
    # =============================================================================
    # ENTROPY: A measure of uncertainty/randomness in a probability distribution
    # =============================================================================
    # Entropy H(p) = -∑ p(x) log p(x)
    #
    # High entropy: Distribution is spread out (uncertain, random)
    #   Example: p = [0.25, 0.25, 0.25, 0.25] → H ≈ 1.39 (very uncertain!)
    #
    # Low entropy: Distribution is peaked (certain, deterministic)
    #   Example: p = [0.97, 0.01, 0.01, 0.01] → H ≈ 0.24 (very certain!)
    #
    # Why compute entropy in language models?
    # - Measures how confident the model is in its predictions
    # - Useful for debugging: extremely low entropy might indicate overfitting
    # - Used in some RL algorithms to encourage exploration
    #
    # Input shape:  [..., vocab_size]  e.g., [batch, seq_len, vocab_size]
    # Output shape: [...]               e.g., [batch, seq_len]
    
    # =============================================================================
    # STEP 1: Compute log-probabilities from logits (numerically stable way)
    # =============================================================================
    # We need to convert logits (raw model outputs) to probabilities
    #
    # The naive way (NUMERICALLY UNSTABLE):
    #   p(x) = exp(logit[x]) / sum(exp(logit[i]))
    #   log p(x) = log(exp(logit[x])) - log(sum(exp(logit[i])))
    #
    # Problem: exp() can overflow for large logits!
    #   If logit = 1000, then exp(1000) = infinity → NaN
    #
    # The stable way (what we use):
    #   log p(x) = logit[x] - logsumexp(logits)
    #   where logsumexp(logits) = log(sum(exp(logit[i])))
    #
    # This is mathematically equivalent but uses the log-sum-exp trick internally
    # to avoid overflow by subtracting the max logit before exponentiating
    
    x = logits  # For clarity: x represents the logits
    
    # Compute log(sum(exp(logits))) along the vocabulary dimension (last dimension)
    # keepdim=True preserves the dimension so we can broadcast during subtraction
    #
    # Example with vocab_size=4:
    #   logits = [2.0, 1.0, 0.5, 0.3]
    #   logsumexp = log(e^2.0 + e^1.0 + e^0.5 + e^0.3) ≈ 2.48
    logsumexp = torch.logsumexp(logits, dim = -1, keepdim = True)
    
    # Compute log-probabilities: log p(x) = logit[x] - logsumexp
    #
    # This is the numerically stable way to compute log(softmax(logits))
    #
    # Example continuing from above:
    #   logpx[0] = 2.0 - 2.48 = -0.48  → p(x) = e^(-0.48) ≈ 0.62
    #   logpx[1] = 1.0 - 2.48 = -1.48  → p(x) = e^(-1.48) ≈ 0.23
    #   logpx[2] = 0.5 - 2.48 = -1.98  → p(x) = e^(-1.98) ≈ 0.14
    #   logpx[3] = 0.3 - 2.48 = -2.18  → p(x) = e^(-2.18) ≈ 0.11
    #   (Note: these sum to ≈1.0 as required for probabilities ✓)
    logpx = logits - logsumexp
    
    # =============================================================================
    # STEP 2: Convert log-probabilities to probabilities
    # =============================================================================
    # We need actual probabilities p(x) to compute entropy
    # p(x) = exp(log p(x))
    #
    # Why go from logits → log-probs → probs instead of just logits → probs?
    # - Numerical stability! The logsumexp trick prevents overflow
    # - We also need log p(x) for the entropy formula anyway
    px = torch.exp(logpx)
    
    # =============================================================================
    # STEP 3: Compute entropy using the formula H(p) = -∑ p(x) log p(x)
    # =============================================================================
    # Entropy formula breakdown:
    # - p(x): Probability of token x
    # - log p(x): Log-probability of token x (already computed as logpx)
    # - p(x) * log p(x): Weighted log-probability
    # - sum over all x: Sum over all tokens in vocabulary
    # - Negative sign: By convention, entropy is positive
    #
    # Example with 4 tokens:
    #   p =     [0.62,  0.23,  0.14,  0.11]
    #   log p = [-0.48, -1.48, -1.98, -2.18]
    #   p*log p = [-0.30, -0.34, -0.28, -0.24]
    #   sum(p*log p) = -1.16
    #   H(p) = -(-1.16) = 1.16  (moderate entropy)
    #
    # Why the negative sign?
    # - log p(x) is always negative (since 0 < p(x) ≤ 1)
    # - p(x) * log p(x) is always negative or zero
    # - sum is negative
    # - We negate to make entropy positive (standard convention)
    #
    # Interpretation:
    # - H ≈ 0: Very certain (one token dominates)
    # - H ≈ log(vocab_size): Very uncertain (uniform distribution)
    # - For vocab_size=50000, max entropy ≈ 10.8
    return -torch.sum(px * logpx, dim = -1) 