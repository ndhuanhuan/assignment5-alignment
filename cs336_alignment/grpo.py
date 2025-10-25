from typing import Literal
import torch

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