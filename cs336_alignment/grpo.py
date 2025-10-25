from typing import Literal, Callable
import torch

def compute_group_normalized_rewards(
    reward_fn: Callable[[str, str], dict[str, float]],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    advantage_eps: float,
    normalize_by_std: bool,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """Compute group-normalized rewards (advantages) for GRPO training.
    
    GRPO generates multiple responses per prompt and normalizes rewards within each group.
    This reduces variance and makes training more stable by comparing responses to the
    same prompt rather than across different prompts.
    
    Two normalization modes:
    1. normalize_by_std=True: A(i) = (r(i) - mean) / (std + ε)  [Standard GRPO]
    2. normalize_by_std=False: A(i) = r(i) - mean  [Dr. GRPO - simpler, avoids rewarding low variance]
    
    Args:
        reward_fn: Function that scores (response, ground_truth) → {"reward": float, ...}
        rollout_responses: All generated responses, length = n_prompts × group_size
        repeated_ground_truths: Ground truths repeated group_size times per prompt
        group_size: Number of responses generated per prompt (typically 4-8)
        advantage_eps: Small constant (e.g., 1e-8) to prevent division by zero
        normalize_by_std: If True, use standard GRPO; if False, use Dr. GRPO (simpler)
    
    Returns:
        advantages: (rollout_batch_size,) - Normalized rewards for training
        raw_rewards: (rollout_batch_size,) - Original rewards for logging
        metadata: Dict with reward statistics for monitoring
    
    Example:
        # 2 prompts, 3 responses each (group_size=3)
        rollout_responses = ["ans1", "ans2", "ans3", "ans4", "ans5", "ans6"]
        repeated_ground_truths = ["truth1", "truth1", "truth1", "truth2", "truth2", "truth2"]
        group_size = 3
        
        # Suppose rewards are:
        raw_rewards = [0.8, 0.5, 0.2,   # Group 1 (prompt 1)
                       1.0, 0.7, 0.4]   # Group 2 (prompt 2)
        
        # Group 1: mean=0.5, std=0.3
        # Group 2: mean=0.7, std=0.3
        
        # With normalize_by_std=True:
        advantages = [(0.8-0.5)/0.3, (0.5-0.5)/0.3, (0.2-0.5)/0.3,
                      (1.0-0.7)/0.3, (0.7-0.7)/0.3, (0.4-0.7)/0.3]
                   = [1.0, 0.0, -1.0, 1.0, 0.0, -1.0]
        
        # With normalize_by_std=False (Dr. GRPO):
        advantages = [0.3, 0.0, -0.3, 0.3, 0.0, -0.3]
    """
    # ==========================================================================
    # STEP 1: Validate input dimensions
    # ==========================================================================
    rollout_batch_size = len(rollout_responses)
    assert rollout_batch_size % group_size == 0, (
        f"rollout_batch_size ({rollout_batch_size}) must be divisible by "
        f"group_size ({group_size})"
    )
    
    # Calculate number of unique prompts
    # Example: 12 responses with group_size=4 → 3 prompts
    n_prompts_per_rollout_batch = rollout_batch_size // group_size
    
    # ==========================================================================
    # STEP 2: Compute raw rewards using the reward function
    # ==========================================================================
    # Call reward_fn for each (response, ground_truth) pair
    # reward_fn returns a dict, we extract the "reward" key
    #
    # Example:
    #   rollout_responses = ["42", "41", "43"]
    #   repeated_ground_truths = ["42", "42", "42"]
    #   rewards = [1.0, 0.0, 0.0]  # Only first answer is correct
    rewards = [
        reward_fn(response, truth)['reward'] 
        for response, truth in zip(rollout_responses, repeated_ground_truths)
    ]
    
    # ==========================================================================
    # STEP 3: Reshape rewards into groups
    # ==========================================================================
    # Convert to tensor and reshape into (n_prompts, group_size)
    # This groups responses by their original prompt
    #
    # Example with 6 responses, group_size=3:
    #   Before: [r1, r2, r3, r4, r5, r6]
    #   After:  [[r1, r2, r3],    # Group 1 (prompt 1)
    #            [r4, r5, r6]]    # Group 2 (prompt 2)
    raw_rewards = torch.tensor(rewards, dtype=torch.float32).reshape(
        n_prompts_per_rollout_batch, 
        group_size
    )
    
    # ==========================================================================
    # STEP 4: Compute per-group mean (baseline)
    # ==========================================================================
    # Calculate mean reward within each group
    # Shape: (n_prompts, 1) - keepdim=True preserves dimension for broadcasting
    #
    # Why subtract the mean?
    # - Reduces variance in policy gradient estimates
    # - Makes training more stable
    # - Responses are compared relative to other responses to the same prompt
    #
    # Example:
    #   raw_rewards = [[0.8, 0.5, 0.2],
    #                  [1.0, 0.7, 0.4]]
    #   mean = [[0.5],   # (0.8+0.5+0.2)/3
    #           [0.7]]   # (1.0+0.7+0.4)/3
    mean = torch.mean(raw_rewards, dim=-1, keepdim=True)
    
    # ==========================================================================
    # STEP 5: Compute advantages (centered rewards)
    # ==========================================================================
    # Subtract group mean from each reward
    # Broadcasting: (n_prompts, group_size) - (n_prompts, 1)
    #
    # Positive advantage → response better than average for this prompt
    # Negative advantage → response worse than average for this prompt
    # Zero advantage → response is exactly average
    #
    # Example (continuing from above):
    #   advantages = [[0.8-0.5, 0.5-0.5, 0.2-0.5],
    #                 [1.0-0.7, 0.7-0.7, 0.4-0.7]]
    #              = [[0.3, 0.0, -0.3],
    #                 [0.3, 0.0, -0.3]]
    advantages = raw_rewards - mean
    
    # ==========================================================================
    # STEP 6: Optionally normalize by standard deviation
    # ==========================================================================
    if normalize_by_std:
        # Standard GRPO: Divide by per-group standard deviation
        # This makes advantages scale-invariant (unit variance per group)
        #
        # Why normalize by std?
        # - Makes gradient magnitudes consistent across prompts
        # - Prevents prompts with high reward variance from dominating training
        #
        # Why Dr. GRPO argues against this:
        # - Rewards prompts with low variation (all answers similar quality)
        # - May not be desirable - we want to distinguish good vs bad prompts
        #
        # Example:
        #   std = [[0.3],   # Standard deviation of [0.8, 0.5, 0.2]
        #          [0.3]]   # Standard deviation of [1.0, 0.7, 0.4]
        #   advantages = [[0.3/0.3, 0.0/0.3, -0.3/0.3],
        #                 [0.3/0.3, 0.0/0.3, -0.3/0.3]]
        #              = [[1.0, 0.0, -1.0],
        #                 [1.0, 0.0, -1.0]]
        #
        # Note: advantage_eps prevents division by zero if all rewards are identical
        advantages = advantages / (torch.std(raw_rewards, dim=-1, keepdim=True) + advantage_eps)
    
    # ==========================================================================
    # STEP 7: Flatten and return results
    # ==========================================================================
    # Reshape from (n_prompts, group_size) back to (rollout_batch_size,)
    # This matches the original order of rollout_responses
    #
    # Example:
    #   advantages = [[1.0, 0.0, -1.0],
    #                 [1.0, 0.0, -1.0]]
    #   After reshape: [1.0, 0.0, -1.0, 1.0, 0.0, -1.0]
    return (
        advantages.reshape(-1),    # Flattened advantages for training
        raw_rewards.reshape(-1),   # Unnormalized rewards for each rollout response
        {}                         # Empty metadata dict (can add reward stats if needed)
    )




def compute_naive_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
) -> torch.Tensor:
    """Compute the naive policy gradient loss for reinforcement learning.
    
    This implements the REINFORCE algorithm's core loss function:
        Loss = -A_t · log p_θ(o_t | q, o_<t)
    
    Where:
        - A_t is the advantage (or raw reward) for the entire sequence
        - log p_θ(o_t | q, o_<t) is the log-probability of token o_t given the question q
          and previous output tokens o_<t
    
    Args:
        raw_rewards_or_advantages: torch.Tensor of shape (batch_size,):
            The advantage or raw reward for each sequence in the batch.
            This is a scalar per sequence that indicates how "good" the entire response was.
            Higher values mean better responses that should be reinforced.
        
        policy_log_probs: torch.Tensor of shape (batch_size, sequence_length):
            Per-token log-probabilities from the policy model.
            Each value log p_θ(o_t | q, o_<t) represents how likely the model was
            to generate token o_t given the context.
    
    Returns:
        torch.Tensor of shape (batch_size, sequence_length):
            The per-token policy gradient loss (negative of advantage × log-prob).
            This is NOT a true loss metric and should not be reported for evaluation.
            Instead, track train/validation returns and other RL metrics.
    
    Note:
        This is called "naive" because it applies the same advantage value to ALL tokens
        in the sequence, even though some tokens may have contributed more to the final
        reward than others. More sophisticated methods use per-token advantages.
    """
    # =============================================================================
    # NAIVE POLICY GRADIENT LOSS (REINFORCE Algorithm)
    # =============================================================================
    # This implements the fundamental policy gradient objective from reinforcement learning.
    #
    # The goal of RL fine-tuning is to adjust the policy (your foundation model) so that
    # it generates responses that receive higher rewards from the reward model.
    #
    # INTUITION:
    # - If a response gets a high reward (positive advantage), we want to INCREASE the
    #   probability of generating those tokens again in the future
    # - If a response gets a low reward (negative advantage), we want to DECREASE the
    #   probability of generating those tokens
    #
    # HOW IT WORKS:
    # 1. policy_log_probs: Contains log p_θ(o_t | q, o_<t) for each token
    #    - Shape: (batch_size, sequence_length)
    #    - These are the log-probabilities the policy model assigned to the tokens
    #      that were actually generated
    #
    # 2. raw_rewards_or_advantages: The "goodness" score for each sequence
    #    - Shape: (batch_size,) - one scalar per sequence
    #    - Positive values → good responses we want to reinforce
    #    - Negative values → bad responses we want to discourage
    #
    # 3. Multiplication: raw_rewards_or_advantages * policy_log_probs
    #    - Broadcasting: (batch_size,) × (batch_size, sequence_length)
    #    - Result: (batch_size, sequence_length)
    #    - Each token's log-prob is weighted by the sequence's advantage
    #
    # 4. Negation: We negate because:
    #    - Optimizers MINIMIZE loss
    #    - We want to MAXIMIZE reward
    #    - Minimizing -reward = Maximizing reward
    #
    # MATHEMATICAL DERIVATION:
    # The policy gradient theorem tells us that to maximize expected reward J(θ):
    #   ∇_θ J(θ) = E[∇_θ log p_θ(o|q) · A(q, o)]
    #
    # To maximize J(θ) with gradient descent, we minimize -J(θ):
    #   Loss = -A(q, o) · log p_θ(o|q)
    #        = -A(q, o) · Σ_t log p_θ(o_t | q, o_<t)
    #
    # When we call loss.backward(), PyTorch computes:
    #   ∂Loss/∂θ = -A(q, o) · ∂log p_θ(o|q)/∂θ
    #            = -A(q, o) · ∇_θ log p_θ(o|q)
    #
    # Gradient descent: θ_new = θ_old - α · ∂Loss/∂θ
    #                         = θ_old + α · A(q, o) · ∇_θ log p_θ(o|q)
    #
    # This increases the log-probability of actions with positive advantage!
    #
    # WHY "NAIVE"?
    # This is called "naive" because it uses the same advantage value for ALL tokens
    # in a sequence. In reality:
    # - Early tokens in the response affect later tokens
    # - Some tokens contribute more to the final reward than others
    # - More sophisticated methods (like generalized advantage estimation) compute
    #   per-token advantages or use baselines to reduce variance
    #
    # EXAMPLE:
    # Suppose we have a batch of 2 sequences:
    #   Sequence 1: "The answer is 42" (3 tokens for simplicity)
    #   Sequence 2: "I don't know" (3 tokens)
    #
    # The reward model scores them:
    #   raw_rewards_or_advantages = [2.0, -1.0]  # Seq 1 is good, Seq 2 is bad
    #
    # The policy assigned these log-probs:
    #   policy_log_probs = [[-0.5, -0.3, -0.2],   # log-probs for "The answer is 42"
    #                       [-0.4, -0.6, -0.8]]   # log-probs for "I don't know"
    #
    # After broadcasting multiplication:
    #   2.0 * [-0.5, -0.3, -0.2] = [-1.0, -0.6, -0.4]
    #  -1.0 * [-0.4, -0.6, -0.8] = [0.4, 0.6, 0.8]
    #
    # After negation (the loss):
    #   - raw_rewards_or_advantages * policy_log_probs = 
    #   [[1.0, 0.6, 0.4],     # Sequence 1: positive advantage → negative loss
    #    [-0.4, -0.6, -0.8]]  # Sequence 2: negative advantage → positive loss
    #
    # When we minimize this loss:
    # - Sequence 1 (good): loss is negative, gradients increase log-probs (reinforce)
    # - Sequence 2 (bad): loss is positive, gradients decrease log-probs (discourage)
    #
    # IMPORTANT NOTE:
    # This is NOT a "loss" in the traditional supervised learning sense!
    # - It's a negative expected reward (we minimize it to maximize reward)
    # - DO NOT report this value as an evaluation metric
    # - Instead, track: average reward, success rate, policy entropy, KL divergence, etc.
    
    # policy_log_probs shape: (batch_size, sequence_length)
    # raw_rewards_or_advantages shape: (batch_size,)
    # Broadcasting happens: (batch_size, 1) × (batch_size, sequence_length)
    # Result shape: (batch_size, sequence_length)
    return - raw_rewards_or_advantages * policy_log_probs

def compute_grpo_clip_loss(
    advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the GRPO-Clip (PPO-style) loss for policy optimization.
    
    Implements: -min(ratio * A_t, clip(ratio, 1-ε, 1+ε) * A_t)
    where ratio = π_θ(o_t|q,o_<t) / π_θ_old(o_t|q,o_<t)
    
    Args:
        advantages: Per-token or per-sequence advantages
        policy_log_probs: log π_θ(o_t|q,o_<t) from current policy
        old_log_probs: log π_θ_old(o_t|q,o_<t) from old policy
        cliprange: Clipping parameter ε (typically 0.2)
    
    Returns:
        Tuple of (loss, metadata_dict)
    """
    # Compute probability ratio: π_θ(o_t) / π_θ_old(o_t)
    # 
    # WHY exp(log_prob - old_log_prob)?
    # We have: log π_θ(o_t) and log π_θ_old(o_t)
    # We want: π_θ(o_t) / π_θ_old(o_t)
    #
    # Using log properties:
    #   ratio = π_θ / π_θ_old
    #         = exp(log π_θ) / exp(log π_θ_old)
    #         = exp(log π_θ - log π_θ_old)    [since exp(a)/exp(b) = exp(a-b)]
    #
    # This is numerically stable (avoids underflow from small probabilities)
    ratio = torch.exp(policy_log_probs - old_log_probs)
    
    # Clip the ratio to [1-ε, 1+ε] to prevent large policy updates
    # If ratio > 1+ε: policy assigns much higher probability than old policy (clip it)
    # If ratio < 1-ε: policy assigns much lower probability than old policy (clip it)
    clip_ratio = torch.clip(ratio, min = 1 - cliprange, max = 1 + cliprange)
    
    # Compute the clipped objective: -min(A*ratio, A*clip_ratio)
    # - Unclipped term: advantages * ratio (normal policy gradient)
    # - Clipped term: advantages * clip_ratio (conservative update)
    # - Take minimum (most pessimistic) to prevent destructive updates
    # - Negate to convert maximization into minimization problem
    loss = -torch.min(advantages * ratio, advantages * clip_ratio)
    
    return (loss, {})

def compute_policy_gradient_loss(
  policy_log_probs: torch.Tensor,
  loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
  raw_rewards: torch.Tensor | None= None,
  advantages: torch.Tensor | None= None,
  old_log_probs: torch.Tensor | None= None,
  cliprange: float | None= None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
  if loss_type == 'no_baseline':
    return (compute_naive_policy_gradient_loss(raw_rewards, policy_log_probs), {})
  elif loss_type == 'reinforce_with_baseline':
    return (compute_naive_policy_gradient_loss(advantages, policy_log_probs), {})
  else:
    return compute_grpo_clip_loss(advantages, policy_log_probs, old_log_probs, cliprange)

def masked_mean(tensor: torch.Tensor, mask: torch.Tensor, dim: int | None = None) -> torch.Tensor:
    """Compute mean of tensor only over masked positions (where mask==1).
    
    This computes the average over response tokens only, ignoring prompt/padding tokens.
    Standard in RL for LLMs to normalize by response length.
    
    Args:
        tensor: Values to average (e.g., losses, entropies)
        mask: Binary mask (1=include, 0=ignore)
        dim: Dimension to average over. If None, average over all masked elements.
    
    Returns:
        Mean of masked elements along specified dimension.
    
    Example:
        tensor = [[1.0, 2.0, 3.0, 4.0],
                  [5.0, 6.0, 7.0, 8.0]]
        mask   = [[1,   1,   0,   0],     # First 2 tokens are response
                  [1,   1,   1,   0]]     # First 3 tokens are response
        
        masked_mean(tensor, mask, dim=1):
            → [(1.0+2.0)/2, (5.0+6.0+7.0)/3]
            → [1.5, 6.0]
        
        masked_mean(tensor, mask, dim=None):
            → (1.0+2.0+5.0+6.0+7.0) / 5
            → 4.2
    """
    # Zero out non-response tokens (mask==0)
    masked_tensor = tensor * mask
    
    # Compute mean: sum(masked values) / count(masked positions)
    # This normalizes by response length, not total sequence length
    return torch.sum(masked_tensor, dim = dim) / torch.sum(mask, dim = dim)

def grpo_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Perform one microbatch training step for GRPO (Group Relative Policy Optimization).
    
    This function orchestrates the RL training pipeline:
    1. Compute per-token policy gradient loss
    2. Average over response tokens (masked mean)
    3. Average over batch
    4. Scale for gradient accumulation
    5. Backpropagate
    
    Args:
        policy_log_probs: (batch_size, seq_len) - log-probs from current policy
        response_mask: (batch_size, seq_len) - 1 for response tokens, 0 for prompt/padding
        gradient_accumulation_steps: Number of microbatches before optimizer step
        loss_type: Type of policy gradient loss to use
        raw_rewards: (batch_size,) - raw rewards (for "no_baseline")
        advantages: (batch_size,) or (batch_size, seq_len) - advantages (for other loss types)
        old_log_probs: (batch_size, seq_len) - log-probs from old policy (for "grpo_clip")
        cliprange: Clipping parameter ε (for "grpo_clip")
    
    Returns:
        Tuple of (scalar_loss, metadata_dict)
    """
    # ==========================================================================
    # STEP 1: Compute per-token policy gradient loss
    # ==========================================================================
    # This computes -A_t * log π_θ(o_t) for each token
    # Result shape: (batch_size, sequence_length)
    #
    # Depending on loss_type:
    # - "no_baseline": Uses raw rewards, naive REINFORCE
    # - "reinforce_with_baseline": Uses advantages (rewards - baseline)
    # - "grpo_clip": Uses PPO-style clipped objective with advantages
    loss, _ = compute_policy_gradient_loss(
        policy_log_probs, 
        loss_type, 
        raw_rewards, 
        advantages, 
        old_log_probs, 
        cliprange
    )
    
    # ==========================================================================
    # STEP 2: Aggregate per-token losses to per-sequence losses
    # ==========================================================================
    # Use masked_mean to average loss over response tokens only
    # This normalizes by response length (standard in RL for LLMs)
    #
    # Before: loss shape is (batch_size, sequence_length)
    # After: loss shape is (batch_size,) - one scalar per sequence
    #
    # Example:
    #   loss = [[0.5, 0.3, 0.0, 0.0],   # Only first 2 tokens are response
    #           [0.8, 0.6, 0.4, 0.0]]   # First 3 tokens are response
    #   mask = [[1,   1,   0,   0],
    #           [1,   1,   1,   0]]
    #   
    #   masked_mean(loss, mask) = [(0.5+0.3)/2, (0.8+0.6+0.4)/3]
    #                           = [0.4, 0.6]
    #
    # Note: dim is not specified, so it defaults to None, but masked_mean
    # will sum over sequence dimension and divide by mask count per sequence
    loss = masked_mean(loss, response_mask)
    
    # ==========================================================================
    # STEP 3: Average over batch dimension (implicit in next step)
    # ==========================================================================
    # The loss is now (batch_size,), but when we call .backward(), PyTorch
    # will automatically sum over the batch. To get the mean, we need to
    # divide by batch_size, but this is typically handled by the optimizer
    # or implicitly through gradient accumulation scaling.
    #
    # In this implementation, we're using the .mean() implicitly through
    # the gradient accumulation scaling.
    
    # ==========================================================================
    # STEP 4: Scale for gradient accumulation
    # ==========================================================================
    # When using gradient accumulation with G microbatches:
    # - We process G microbatches sequentially
    # - Gradients accumulate (sum) across microbatches
    # - To get the average gradient, we divide each microbatch's loss by G
    #
    # Why divide by gradient_accumulation_steps?
    # - Microbatch 1: loss₁/G → backward → gradients/G
    # - Microbatch 2: loss₂/G → backward → gradients/G accumulate
    # - ...
    # - After G microbatches: total gradient = (grad₁ + grad₂ + ... + gradG)/G
    # - This equals the gradient of the average loss ✓
    #
    # Example:
    #   If loss = [0.4, 0.6] and gradient_accumulation_steps = 2:
    #   Scaled loss = [0.4, 0.6] / 2 = [0.2, 0.3]
    loss = loss / gradient_accumulation_steps
    
    # ==========================================================================
    # STEP 5: Backpropagate to compute gradients
    # ==========================================================================
    # loss.backward() computes gradients with respect to model parameters
    # These gradients will accumulate in the parameter .grad attributes
    #
    # Important: We do NOT call optimizer.step() here!
    # - The optimizer step happens after processing all microbatches
    # - This allows gradient accumulation to simulate larger batch sizes
    #
    # Training loop pattern:
    #   optimizer.zero_grad()
    #   for microbatch in range(gradient_accumulation_steps):
    #       grpo_microbatch_train_step(...)  # Accumulates gradients
    #   optimizer.step()  # Update parameters with accumulated gradients
    loss.backward()
    
    # ==========================================================================
    # STEP 6: Return the loss for logging
    # ==========================================================================
    # We return the scaled loss (before backward) for monitoring
    # Empty metadata dict for compatibility with other train step functions
    return (loss, {})
