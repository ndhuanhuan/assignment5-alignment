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


def get_response_log_probs(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool = False,
) -> dict[str, torch.Tensor]:
    """Get the conditional log-probs of the response given the prompt,
        and optionally the entropy of the next token predictions.

    Args:
        model: PreTrainedModel, the model to score.
        input_ids: torch.Tensor of shape (batch_size, sequence_length):
            the tokenized prompt and output.
        labels: torch.Tensor of shape (batch_size, sequence_length):
            shifted input_ids.
        return_token_entropy: bool, whether to return the entropy of the
            next token predictions.

    Returns:
        dict[str, torch.Tensor]:
            "log_probs": torch.Tensor of shape (batch_size, sequence_length):
                the conditional log-probs of the response given the prompt.
                Note that we have not masked out the token indices corresponding
                to the prompt or padding; that is done in the train loop.
            "token_entropy": Optional[torch.Tensor] of shape (batch_size, sequence_length):
                the entropy of the next token predictions. As with the log-probs,
                we have not masked out the token indices corresponding to the prompt
                or padding; that is done in the train loop.
    """
    # =============================================================================
    # PURPOSE: Compute log-probabilities of the actual tokens that appeared
    # =============================================================================
    # This function answers: "How likely was each token that actually appeared?"
    #
    # In SFT (Supervised Fine-Tuning), we need to know how confident the model
    # was in predicting the correct output tokens. This is used to:
    # 1. Compute the training loss (negative log-likelihood)
    # 2. Evaluate model performance
    # 3. Compare policy models in RL (GRPO)
    #
    # Input shapes:
    #   input_ids: [batch_size, sequence_length]  - What model sees as context
    #   labels:    [batch_size, sequence_length]  - What model should predict
    #
    # Example:
    #   input_ids = [[10, 20, 30]]  - "What is"
    #   labels    = [[20, 30, 40]]  - "is 2+2?"
    #   We want: log P(20|10), log P(30|10,20), log P(40|10,20,30)
    
    # =============================================================================
    # STEP 1: Forward pass through the model to get logits
    # =============================================================================
    # The model outputs raw scores (logits) for each token in the vocabulary
    # at each position in the sequence.
    #
    # input_ids shape: [batch_size, sequence_length]
    # logits shape:    [batch_size, sequence_length, vocab_size]
    #
    # Each logit[b, s, v] represents the model's raw score for predicting
    # vocabulary token v at position s in batch item b.
    #
    # Example with vocab_size=100:
    #   Position 0: model sees [10], produces 100 logits for next token
    #   Position 1: model sees [10, 20], produces 100 logits for next token
    #   Position 2: model sees [10, 20, 30], produces 100 logits for next token
    logits = model(input_ids).logits  # [batch_size, seq_length, vocab_size]
    
    # =============================================================================
    # STEP 2: Convert logits to log-probabilities
    # =============================================================================
    # Logits are raw scores - we need to convert them to probabilities
    # using softmax, then take the log.
    #
    # Why log-probabilities instead of probabilities?
    # 1. Numerical stability (probabilities can be very small, causing underflow)
    # 2. Math convenience (multiplication becomes addition in log space)
    # 3. Cross-entropy loss directly uses log-probabilities
    #
    # log_softmax computes: log(exp(logit[v]) / sum(exp(logit[i])))
    # This is more numerically stable than: log(softmax(logits))
    #
    # Example at one position:
    #   logits = [2.0, 1.0, 0.5, ..., 0.1]  (100 values)
    #   After log_softmax: [-0.48, -1.48, -1.98, ..., -2.38]
    #   These are log P(token_0), log P(token_1), log P(token_2), ...
    log_softmax = torch.nn.functional.log_softmax(logits, dim=-1)
    # Shape: [batch_size, sequence_length, vocab_size]
    
    # =============================================================================
    # STEP 3: Extract log-probabilities of the actual tokens (labels)
    # =============================================================================
    # We have log-probabilities for ALL possible tokens at each position,
    # but we only care about the log-probability of the token that ACTUALLY
    # appeared (i.e., the label).
    #
    # torch.gather extracts specific values from a tensor using indices.
    #
    # Let's break down what happens:
    # 1. labels.unsqueeze(-1): Add a dimension for gather
    #    labels shape: [batch_size, seq_length]
    #    After unsqueeze: [batch_size, seq_length, 1]
    #
    # 2. torch.gather(log_softmax, dim=-1, index=labels.unsqueeze(-1)):
    #    From the vocabulary dimension (dim=-1), select the index specified by labels
    #    Result shape: [batch_size, seq_length, 1]
    #
    # 3. .squeeze(-1): Remove the extra dimension
    #    Final shape: [batch_size, seq_length]
    #
    # Visual example for one sequence:
    #   Position 0: log_softmax has 100 values, labels[0]=20
    #               → Extract log_softmax[0, 20] → log P(token_20)
    #   Position 1: log_softmax has 100 values, labels[1]=30
    #               → Extract log_softmax[1, 30] → log P(token_30)
    #   Position 2: log_softmax has 100 values, labels[2]=40
    #               → Extract log_softmax[2, 40] → log P(token_40)
    #
    # Result: [log P(20), log P(30), log P(40)]
    log_probs = torch.gather(log_softmax, -1, labels.unsqueeze(-1)).squeeze(-1)
    # Shape: [batch_size, sequence_length]
    
    # =============================================================================
    # STEP 4: Package results in a dictionary
    # =============================================================================
    # We always return log_probs, and optionally return token_entropy
    result = {'log_probs': log_probs}
    
    # =============================================================================
    # STEP 5: Optionally compute token entropy
    # =============================================================================
    # If requested, compute the entropy of the probability distribution at
    # each position. This measures how uncertain/confident the model is.
    #
    # High entropy: Model is uncertain (probabilities spread out)
    # Low entropy: Model is confident (probability concentrated on few tokens)
    #
    # This is useful for:
    # - Monitoring model confidence during training
    # - Exploration in RL (encourage high entropy to explore)
    # - Debugging (very low entropy might indicate overfitting)
    if return_token_entropy:
        result['token_entropy'] = compute_entropy(logits)
        # Shape: [batch_size, sequence_length]
    
    # =============================================================================
    # RETURN: Dictionary with log_probs and optionally token_entropy
    # =============================================================================
    # Note: We do NOT mask out prompt or padding tokens here!
    # That masking happens in the training loop using the response_mask.
    #
    # Why not mask here?
    # - This function is more general-purpose (can be used for evaluation too)
    # - Different use cases need different masking strategies
    # - Separating concerns: this computes probabilities, caller decides masking
    return result

def masked_normalize(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: int | None = None,
    normalize_constant: float = 1.0,
) -> torch.Tensor:
    """Sum over a dimension and normalize by a constant,
    considering only the elements with mask value 1.

    Args:
        tensor: torch.Tensor, the tensor to sum and normalize.
        mask: torch.Tensor, the mask. We only consider elements
            with mask value 1.
        dim: int | None, the dimension to sum along before
            normalization. If None, sum over all dimensions.
        normalize_constant: float, the constant to divide by
            for normalization.

    Returns:
        torch.Tensor, the normalized sum, where masked elements
            (mask=0) don't contribute to the sum.
    """
    # =============================================================================
    # PURPOSE: Compute masked sum and normalize by a constant
    # =============================================================================
    # This function is crucial for computing loss in SFT (Supervised Fine-Tuning)
    # where we need to:
    # 1. Sum log-probabilities only for OUTPUT tokens (not prompt or padding)
    # 2. Normalize by the total number of output tokens
    #
    # Why do we need masking?
    # - In SFT, we concatenate prompt + output into a single sequence
    # - But we only want to compute loss on the OUTPUT tokens
    # - The mask indicates which tokens to include (1) and exclude (0)
    #
    # Example use case in SFT:
    #   log_probs = [-0.5, -0.3, -0.8, -0.2, -0.1]  (log-probs for all tokens)
    #   mask      = [   0,    0,    1,    1,    1]  (0=prompt, 1=output)
    #   normalize_constant = 3  (number of output tokens)
    #
    # We want: sum(-0.8, -0.2, -0.1) / 3 = -1.1 / 3 = -0.367
    # This is the average negative log-likelihood of the output tokens
    
    # =============================================================================
    # STEP 1: Apply the mask to zero out unwanted elements
    # =============================================================================
    # We need to multiply the tensor by the mask to set masked-out positions to 0
    #
    # Why multiply instead of indexing?
    # - Indexing would change the tensor shape, making it hard to maintain dimensions
    # - Multiplication keeps the shape intact, just zeros out masked positions
    # - This allows us to use torch.sum() with specific dimensions
    #
    # Example:
    #   tensor = [[1.0, 2.0, 3.0],     mask = [[1, 0, 1],
    #             [4.0, 5.0, 6.0]]              [1, 1, 0]]
    #
    #   masked_tensor = [[1.0, 0.0, 3.0],
    #                    [4.0, 5.0, 0.0]]
    #
    # Note: We convert mask to float because:
    # - mask is boolean (True/False or 1/0)
    # - tensor is float
    # - PyTorch requires matching dtypes for element-wise operations
    masked_tensor = tensor * mask.float()
    
    # =============================================================================
    # STEP 2: Sum along the specified dimension (or all dimensions)
    # =============================================================================
    # The 'dim' parameter controls which dimension to sum over:
    # - dim=None: Sum ALL elements, return a scalar
    # - dim=0: Sum along rows (collapsing the batch dimension)
    # - dim=1: Sum along columns (collapsing the sequence dimension)
    # - dim=-1: Sum along the last dimension
    #
    # Example with our masked_tensor above:
    #   dim=None: sum all → 1.0 + 0.0 + 3.0 + 4.0 + 5.0 + 0.0 = 13.0
    #   dim=0: sum rows → [5.0, 5.0, 3.0]  (each column summed)
    #   dim=1: sum cols → [4.0, 9.0]       (each row summed)
    #
    # Why use torch.sum() instead of .sum()?
    # - Both work, but torch.sum() is more explicit
    # - Makes it clear we're calling a PyTorch function
    # - Consistent with other torch operations in the codebase
    masked_sum = torch.sum(masked_tensor, dim=dim)
    
    # =============================================================================
    # STEP 3: Normalize by dividing by the constant
    # =============================================================================
    # The normalize_constant is typically:
    # - Number of tokens in the output (for average per-token loss)
    # - Gradient accumulation steps (for distributed training)
    # - Batch size (for average per-example loss)
    #
    # Why normalize?
    # - Makes losses comparable across different sequence lengths
    # - Prevents longer sequences from dominating the gradient
    # - Allows fair comparison between different batches
    #
    # Example:
    #   If masked_sum = 13.0 and normalize_constant = 2.0
    #   Result = 13.0 / 2.0 = 6.5
    #
    # In SFT loss computation:
    #   log_probs: [batch, seq_len] with values like [-0.5, -0.3, -0.8, ...]
    #   response_mask: [batch, seq_len] with values [0, 0, 1, 1, 1]
    #   normalize_constant: total number of output tokens
    #
    #   masked_normalize(log_probs, response_mask, dim=None, normalize_constant)
    #   → Average log-probability across all output tokens in the batch
    #   → Negate this to get the cross-entropy loss (NLL loss)
    return masked_sum / normalize_constant

def sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the policy gradient loss and backprop its gradients for a microbatch.
    
    Args:
        policy_log_probs: torch.Tensor of shape (batch_size, sequence_length):
            Per-token log-probabilities from the SFT policy being trained.
        response_mask: torch.Tensor of shape (batch_size, sequence_length):
            1 for response tokens, 0 for prompt/padding tokens.
        gradient_accumulation_steps: int:
            Number of microbatches per optimizer step.
        normalize_constant: float:
            The constant by which to divide the sum. Defaults to 1.0.
    
    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]]:
            - loss: Scalar tensor. The microbatch loss, adjusted for gradient accumulation.
            - metadata: Dict with metadata (can include statistics for logging).
    """
    # =============================================================================
    # COMPUTE SFT LOSS WITH GRADIENT ACCUMULATION
    # =============================================================================
    # This function computes the Supervised Fine-Tuning loss in three steps:
    #
    # 1. masked_normalize: Compute average log-prob per sequence
    #    - For each sequence in the batch, sum log-probs of response tokens
    #    - Divide by normalize_constant (typically 1.0, but could be num_tokens)
    #    - Result shape: (batch_size,)
    #
    # 2. .mean(): Average across the batch
    #    - Reduces (batch_size,) to scalar
    #    - Gives us the mean log-likelihood across all sequences
    #
    # 3. Negate and scale: -(...) / gradient_accumulation_steps
    #    - Negate: maximize log-likelihood = minimize negative log-likelihood
    #    - Scale by gradient_accumulation_steps for proper gradient accumulation
    #
    # Why scale by gradient_accumulation_steps?
    # - When accumulating gradients over G microbatches, each microbatch
    #   contributes to the total gradient
    # - Dividing by G ensures that the accumulated gradient equals the gradient
    #   of the average loss across all G microbatches
    # - This simulates training with a batch size of G × microbatch_size
    #
    # Example with gradient_accumulation_steps=2:
    #   Microbatch 1: loss = -2.0 / 2 = -1.0, gradients scaled by 1/2
    #   Microbatch 2: loss = -3.0 / 2 = -1.5, gradients scaled by 1/2
    #   After accumulation: total gradient = (grad1 + grad2) / 2 ✓
    loss = -masked_normalize(
        tensor=policy_log_probs,
        mask=response_mask,
        dim=-1,  # Sum over sequence dimension for each batch item
        normalize_constant=normalize_constant
    ).mean() / gradient_accumulation_steps
    
    # Perform backward pass to compute and accumulate gradients
    loss.backward()
    
    # Return loss for logging (empty metadata dict for compatibility)
    return (loss, {})