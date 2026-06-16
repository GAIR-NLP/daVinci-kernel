"""
Advantage computation for mixed batches containing policy / selection / summary trajectories.

Each agent type uses its own grouping logic but the same REINFORCE-LOO formula:
  A_i = R_i - mean(R_{j≠i} in the same group)

Policy agent  → TRLOO (multi-turn; delegates to existing core_algos)
Selection     → single-turn GRPO, grouped by original_uid (across k+1 schemes)
Summary       → single-turn GRPO, grouped by summary_group_id (across s parallel summaries)

Entry point: compute_mixed_batch_advantages()

Compatible with kernel_trainer.py adv_estimator == "trloo" branch.
When skill is disabled (no "agent_type" in non_tensor_batch), this module is NOT called.
"""

from collections import defaultdict
from typing import Optional

import numpy as np
import torch
from verl import DataProto

from verl_patch.trainer.code.ppo import core_algos


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _compute_loo_advantages(
    rewards: torch.Tensor,      # shape (N,) – one scalar per trajectory
    group_keys: np.ndarray,     # shape (N,) – group identifier per trajectory
) -> torch.Tensor:
    """
    Standard REINFORCE Leave-One-Out advantage.

    For each sample i in group G:
      A_i = R_i - mean(R_{j in G, j!=i})

    Edge cases:
      N_group == 1 → A_i = R_i  (no baseline available)
      N_group == 2 → std not used; only mean subtracted
    """
    advantages = torch.zeros_like(rewards)
    key_to_indices: dict[str, list] = defaultdict(list)
    for i, k in enumerate(group_keys):
        key_to_indices[str(k)].append(i)

    for _key, indices in key_to_indices.items():
        idx = torch.tensor(indices, dtype=torch.long, device=rewards.device)
        g = rewards[idx]
        n = len(g)
        if n == 1:
            advantages[idx] = g
        else:
            total = g.sum()
            loo_mean = (total - g) / (n - 1)
            advantages[idx] = g - loo_mean

    return advantages


def _last_token_reward(
    token_level_rewards: torch.Tensor,  # (N, max_response_len)
    response_mask: torch.Tensor,        # (N, max_response_len) 0/1
) -> torch.Tensor:
    """
    Extract the scalar reward placed at the last real token position.
    Returns shape (N,).
    """
    valid_len = response_mask.sum(dim=-1).long()  # (N,)
    n = token_level_rewards.shape[0]
    # last valid token index = valid_len - 1, clamp to 0 to avoid -1
    last_idx = (valid_len - 1).clamp(min=0)
    return token_level_rewards[torch.arange(n, device=token_level_rewards.device), last_idx]


def _scalar_to_token_level(
    scalar_adv: torch.Tensor,           # (N,)
    response_mask: torch.Tensor,        # (N, max_response_len)
    loss_mask: torch.Tensor,            # (N,)  1 = this row participates in loss
) -> torch.Tensor:
    """
    Broadcast a per-trajectory scalar advantage to token level.

    For rows where loss_mask == 1: all real-token positions receive scalar_adv.
    For rows where loss_mask == 0: all positions are zero (padding turns / tool turns).

    Shape: (N, max_response_len)
    """
    # scalar_adv: (N,) -> (N, 1)
    adv_2d = scalar_adv.unsqueeze(-1) * response_mask.float()   # (N, max_resp_len)
    loss_mask_2d = loss_mask.unsqueeze(-1).float()               # (N, 1)
    return adv_2d * loss_mask_2d


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def compute_mixed_batch_advantages(
    data: DataProto,
    max_turns: int,
    gamma: float,
    adv_cross_scheme: bool = False,
    selection_weight: float = 1.0,
    summary_weight: float = 1.0,
) -> None:
    """
    Compute and write advantages in-place to data.batch["advantages"] and
    data.batch["returns"].

    Assumes data.batch["token_level_rewards"] already contains the rewards placed
    at last-token positions (set by the rollout engine).

    For policy entries, delegates to core_algos.compute_multi_turn_rloo_outcome_advantage.
    For selection/summary, uses _compute_loo_advantages (single-turn scalar LOO).

    When adv_cross_scheme=True, additionally normalize policy advantages within
    each original_uid group (across all schemes).
    """
    agent_types: np.ndarray = data.non_tensor_batch.get("agent_type", None)

    # -----------------------------------------------------------------------
    # No skill mode: pure policy batch → original TRLOO, no change
    # -----------------------------------------------------------------------
    if agent_types is None:
        advantages, returns = core_algos.compute_multi_turn_rloo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            eos_mask=data.batch["response_mask"],
            loss_mask=data.batch["loss_mask"],
            turn_indices=data.batch["turn_indices"],
            index=data.non_tensor_batch["uid"],
            max_turns=max_turns,
            gamma=gamma,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        return

    # -----------------------------------------------------------------------
    # Mixed batch: separate by agent_type
    # -----------------------------------------------------------------------
    n_total = len(data)
    policy_mask_bool = (agent_types == "policy")
    selection_mask_bool = (agent_types == "selection")
    summary_mask_bool = (agent_types == "summary")

    advantages = torch.zeros(
        n_total,
        data.batch["token_level_rewards"].shape[1],
        device=data.batch["token_level_rewards"].device,
        dtype=torch.float32,
    )
    returns = torch.zeros_like(advantages)

    # -----------------------------------------------------------------------
    # 1. Policy: TRLOO (existing)
    # -----------------------------------------------------------------------
    if policy_mask_bool.any():
        policy_indices = np.where(policy_mask_bool)[0]
        policy_data = data.select_idxs(torch.tensor(policy_indices, dtype=torch.long))

        # Group by original_uid (task level) so that all (k+1)*n rollouts across
        # every scheme for the same task form one LOO group per turn.
        # This gives up to (k+1)*n samples per group instead of n, yielding a
        # lower-variance baseline.  Schemes are treated as parallel rollouts.
        policy_index = policy_data.non_tensor_batch.get(
            "original_uid", policy_data.non_tensor_batch["uid"]
        )

        p_adv, p_ret = core_algos.compute_multi_turn_rloo_outcome_advantage(
            token_level_rewards=policy_data.batch["token_level_rewards"],
            eos_mask=policy_data.batch["response_mask"],
            loss_mask=policy_data.batch["loss_mask"],
            turn_indices=policy_data.batch["turn_indices"],
            index=policy_index,
            max_turns=max_turns,
            gamma=gamma,
        )

        # Optional cross-scheme normalization
        if adv_cross_scheme and "original_uid" in policy_data.non_tensor_batch:
            p_adv = _normalize_across_schemes(
                p_adv,
                policy_data.non_tensor_batch["original_uid"],
                policy_data.batch["response_mask"],
            )

        idx_t = torch.tensor(policy_indices, dtype=torch.long)
        advantages[idx_t] = p_adv
        returns[idx_t] = p_ret

    # -----------------------------------------------------------------------
    # 2. Selection: single-turn GRPO, grouped by original_uid
    # -----------------------------------------------------------------------
    if selection_mask_bool.any():
        sel_indices = np.where(selection_mask_bool)[0]
        sel_data = data.select_idxs(torch.tensor(sel_indices, dtype=torch.long))

        # Reward: placed at last real token position
        sel_rewards = _last_token_reward(
            sel_data.batch["token_level_rewards"],
            sel_data.batch["response_mask"],
        )
        sel_group_keys = sel_data.non_tensor_batch.get(
            "original_uid",
            sel_data.non_tensor_batch["uid"],
        )
        sel_scalar_adv = _compute_loo_advantages(sel_rewards, sel_group_keys)

        sel_token_adv = _scalar_to_token_level(
            sel_scalar_adv * selection_weight,
            sel_data.batch["response_mask"],
            sel_data.batch.get("loss_mask", torch.ones(len(sel_data), dtype=torch.long)),
        )

        idx_t = torch.tensor(sel_indices, dtype=torch.long)
        advantages[idx_t] = sel_token_adv
        returns[idx_t] = sel_token_adv  # for GRPO, returns == advantages

    # -----------------------------------------------------------------------
    # 3. Summary: single-turn GRPO (whole trajectory), grouped by summary_group_id
    # -----------------------------------------------------------------------
    if summary_mask_bool.any():
        sum_indices = np.where(summary_mask_bool)[0]
        sum_data = data.select_idxs(torch.tensor(sum_indices, dtype=torch.long))

        # Summary is single-turn: each row IS a complete trajectory.
        # token_level_rewards.sum(dim=-1) gives the scalar verify_speedup at the last token.
        sum_rewards_per_row = sum_data.batch["token_level_rewards"].sum(dim=-1)  # (N_sum,)

        # Group key for LOO: summary_group_id if available, else original_uid
        sum_group_keys = sum_data.non_tensor_batch.get(
            "summary_group_id",
            sum_data.non_tensor_batch.get("original_uid", sum_data.non_tensor_batch["uid"]),
        )
        sum_scalar_adv = _compute_loo_advantages(sum_rewards_per_row, sum_group_keys)

        sum_token_adv = _scalar_to_token_level(
            sum_scalar_adv * summary_weight,
            sum_data.batch["response_mask"],
            sum_data.batch.get("loss_mask", torch.ones(len(sum_data), dtype=torch.long)),
        )

        idx_t = torch.tensor(sum_indices, dtype=torch.long)
        advantages[idx_t] = sum_token_adv
        returns[idx_t] = sum_token_adv

    data.batch["advantages"] = advantages
    data.batch["returns"] = returns


# ---------------------------------------------------------------------------
# Cross-scheme normalization (optional)
# ---------------------------------------------------------------------------

def _normalize_across_schemes(
    adv: torch.Tensor,            # (N, max_resp_len) token-level advantages
    original_uids: np.ndarray,    # (N,)
    response_mask: torch.Tensor,  # (N, max_resp_len)
) -> torch.Tensor:
    """
    Normalize policy advantages within each original task (across all schemes & rollouts).
    This helps balance the magnitude of advantages from different schemes.
    """
    # Compute mean absolute advantage per row, then per original_uid group
    key_to_indices: dict[str, list] = defaultdict(list)
    for i, k in enumerate(original_uids):
        key_to_indices[str(k)].append(i)

    normalized = adv.clone()
    for _key, indices in key_to_indices.items():
        idx = torch.tensor(indices, dtype=torch.long, device=adv.device)
        group_adv = adv[idx]
        group_mask = response_mask[idx].float()
        # Mean over all valid tokens in the group
        valid_sum = (group_adv.abs() * group_mask).sum()
        valid_count = group_mask.sum()
        if valid_count > 0:
            mean_abs = valid_sum / valid_count
            normalized[idx] = group_adv / (mean_abs + 1e-8)

    return normalized
