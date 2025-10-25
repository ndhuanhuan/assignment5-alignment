import torch
from transformers import PreTrainedTokenizerBase
from cs336_alignment.sft import get_response_log_probs

# Alpaca prompt template used for formatting instruction-response pairs
# This is the standard template used in instruction-tuning
propmt_template = \
"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{prompt}

### Response:
{response}"""  

def get_logs_prob(lm: torch.nn.Module, prompt: list[int]) -> torch.Tensor:
    """Compute the log-probability of a complete sequence under a language model.
    
    This helper function computes log p(sequence) by:
    1. Splitting sequence into input_ids (all but last token) and labels (all but first)
    2. Getting per-token log-probabilities from the model
    3. Summing them to get the total sequence log-probability
    
    Args:
        lm: Language model to evaluate
        prompt: List of token IDs representing the full sequence (prompt + response + EOS)
    
    Returns:
        Scalar tensor with log p(sequence)
    
    Example:
        tokens = [10, 20, 30, 40]  # "Hello world"
        
        # Model predicts:
        # Given [10], predict 20: log p(20|10) = -0.5
        # Given [10,20], predict 30: log p(30|10,20) = -0.3
        # Given [10,20,30], predict 40: log p(40|10,20,30) = -0.2
        
        # Total: log p(sequence) = -0.5 + -0.3 + -0.2 = -1.0
    """
    # Prepare input_ids and labels for next-token prediction
    # input_ids: [10, 20, 30] (all but last)
    # labels: [20, 30, 40] (all but first)
    # 
    # CRITICAL: Add batch dimension! Model expects (batch_size, seq_len)
    # unsqueeze(0) converts (seq_len,) → (1, seq_len)
    input_ids = torch.tensor(prompt[:-1], dtype=torch.int64).unsqueeze(0)
    labels = torch.tensor(prompt[1:], dtype=torch.int64).unsqueeze(0)
    
    # Get per-token log-probabilities from the model
    # This computes log p(token_t | tokens_<t) for each position
    # Result shape: (1, seq_len) - batch size is 1
    log_probs = get_response_log_probs(lm, input_ids, labels)['log_probs']
    
    # Sum to get total sequence log-probability
    # log p(sequence) = Σ log p(token_t | tokens_<t)
    return torch.sum(log_probs)


def compute_per_instance_dpo_loss(
    lm: torch.nn.Module,
    lm_ref: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    beta: float,
    prompt: str,
    response_chosen: str,
    response_rejected: str,
) -> torch.Tensor:
    """Compute the per-instance DPO (Direct Preference Optimization) loss.
    
    DPO aligns language models using preference data without training a separate reward model.
    The loss for a preference pair (chosen response y_w, rejected response y_l) is:
    
        L = -log σ(β · [log π_θ(y_w|x) - log π_θ(y_l|x) - log π_ref(y_w|x) + log π_ref(y_l|x)])
    
    Where:
        - σ is the sigmoid function
        - β is a temperature parameter controlling deviation from reference model
        - π_θ is the policy being trained
        - π_ref is the frozen reference policy
        - x is the prompt
    
    Key insight: log π(y|x) - log π(y'|x) = log π(x⊕y) - log π(x⊕y')
    because the prompt log-prob cancels out. This simplifies computation.
    
    Args:
        lm: Language model being trained (π_θ)
        lm_ref: Reference language model (π_ref), frozen
        tokenizer: Tokenizer for both models
        beta: DPO temperature parameter (typically 0.1-0.5)
        prompt: The instruction/question
        response_chosen: The preferred response (y_w)
        response_rejected: The rejected response (y_l)
    
    Returns:
        Scalar tensor with the DPO loss for this example
    
    Example:
        prompt = "What is 2+2?"
        response_chosen = "4"  (correct, preferred)
        response_rejected = "5"  (incorrect, rejected)
        
        # Model should learn to increase P(chosen) and decrease P(rejected)
        # relative to the reference model
    """
    # ==========================================================================
    # STEP 1: Format responses using Alpaca template
    # ==========================================================================
    # Apply the same prompt template used during SFT
    # This ensures consistency between training stages
    y_win = propmt_template.format(prompt=prompt, response=response_chosen) 
    y_loss = propmt_template.format(prompt=prompt, response=response_rejected)
    
    # ==========================================================================
    # STEP 2: Tokenize and add EOS token
    # ==========================================================================
    # Convert text to token IDs and append end-of-sequence token
    # EOS token signals the model that the response is complete
    #
    # Example:
    #   response = "The answer is 4"
    #   tokens = [123, 456, 789, 101]  # tokenized
    #   with EOS: [123, 456, 789, 101, 2]  # 2 is EOS token ID
    y_win = tokenizer.encode(y_win) + [tokenizer.eos_token_id]
    y_loss = tokenizer.encode(y_loss) + [tokenizer.eos_token_id]
    
    # ==========================================================================
    # STEP 3: Compute log-probabilities under policy model (π_θ)
    # ==========================================================================
    # Calculate log π_θ(x⊕y_w) and log π_θ(x⊕y_l)
    # These represent how likely the model thinks each complete sequence is
    y_win_logs_prob = get_logs_prob(lm, y_win)      # log π_θ(chosen)
    y_loss_logs_prob = get_logs_prob(lm, y_loss)    # log π_θ(rejected)
    
    # ==========================================================================
    # STEP 4: Compute log-probabilities under reference model (π_ref)
    # ==========================================================================
    # Calculate log π_ref(x⊕y_w) and log π_ref(x⊕y_l)
    # The reference model is frozen and serves as a baseline
    y_ref_win_logs_prob = get_logs_prob(lm_ref, y_win)    # log π_ref(chosen)
    y_ref_loss_logs_prob = get_logs_prob(lm_ref, y_loss)  # log π_ref(rejected)
    
    # ==========================================================================
    # STEP 5: Compute DPO loss
    # ==========================================================================
    # DPO loss = -log σ(β · implicit_reward_diff)
    #
    # Where implicit_reward_diff is:
    #   [log π_θ(y_w|x) - log π_θ(y_l|x)] - [log π_ref(y_w|x) - log π_ref(y_l|x)]
    #
    # Breaking it down:
    # 1. (y_win_logs_prob - y_loss_logs_prob): 
    #    How much more the policy prefers chosen over rejected
    #
    # 2. (y_ref_win_logs_prob - y_ref_loss_logs_prob):
    #    How much more the reference prefers chosen over rejected
    #
    # 3. Difference: How much MORE the policy prefers chosen vs reference
    #    Positive → policy prefers chosen more than reference (good!)
    #    Negative → policy prefers chosen less than reference (bad!)
    #
    # 4. β scales the difference (controls how much we can deviate from reference)
    #
    # 5. logsigmoid converts to probability in log-space
    #    logsigmoid(z) = log(1 / (1 + exp(-z)))
    #
    # 6. Negate to get loss (we minimize loss)
    #
    # Effect:
    # - If policy strongly prefers chosen → loss is low ✓
    # - If policy strongly prefers rejected → loss is high ✗
    # - Bounded loss prevents extreme deviations from reference model
    return -torch.nn.functional.logsigmoid(
        beta * (
            (y_win_logs_prob - y_loss_logs_prob) -      # Policy preference
            (y_ref_win_logs_prob - y_ref_loss_logs_prob) # Reference preference
        )
    )