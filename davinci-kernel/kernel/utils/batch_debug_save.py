"""
batch_debug_save.py — Temporary debug utility for inspecting DataProto batches.

Saves a DataProto to a JSONL file where each line is one row.
Large tensors are summarised unless a tokenizer is provided, in which case
prompt / response / input_ids are decoded to text.

For selection and summary rows, the full token id arrays and rollout_log_probs
arrays are always stored verbatim (not summarised), so they can be inspected
and verified without re-running inference.

Usage (from kernel_trainer.py):
    from kernel.utils.batch_debug_save import save_batch_to_jsonl
    save_batch_to_jsonl(batch, "/tmp/debug/after_rollout_step{N}.jsonl",
                        tokenizer=self.tokenizer)
"""

import json
import os
from typing import Any, Optional

import numpy as np
import torch


# Fields for which the full array is always stored (trimmed to real tokens via response_mask)
# for selection and summary rows, so downstream analysis can verify logprobs / tokens exactly.
_FULL_ARRAY_KEYS_FOR_SKILL = {"responses", "rollout_log_probs", "token_level_scores", "token_level_rewards"}


def _to_serialisable(v: Any, max_len: int = 20) -> Any:
    """Convert a value to something json.dumps can handle."""
    if isinstance(v, torch.Tensor):
        t = v.detach().cpu()
        if t.numel() == 0:
            return {"__tensor_shape__": list(t.shape), "data": []}
        if t.numel() <= max_len:
            return {"__tensor_shape__": list(t.shape), "data": t.tolist()}
        flat = t.float()
        return {
            "__tensor_shape__": list(t.shape),
            "min": float(flat.min()),
            "max": float(flat.max()),
            "mean": float(flat.mean()),
            "first_5": flat.flatten()[:5].tolist(),
            "last_5": flat.flatten()[-5:].tolist(),
        }
    if isinstance(v, np.ndarray):
        if v.size <= max_len:
            return v.tolist()
        return {"__ndarray_shape__": list(v.shape), "first_5": v.flat[:5].tolist()}
    if isinstance(v, (np.integer, np.floating, np.bool_)):
        return v.item()
    if isinstance(v, (list, tuple)) and len(v) > max_len:
        return list(v[:max_len]) + [f"... ({len(v)} total)"]
    if isinstance(v, dict):
        return {str(k): _to_serialisable(val, max_len) for k, val in v.items()}
    try:
        json.dumps(v)
        return v
    except (TypeError, ValueError):
        return repr(v)


def _tensor_to_trimmed_list(t: torch.Tensor, mask: Optional[torch.Tensor]) -> list:
    """Return t as a Python list, trimmed to real positions by mask (if given)."""
    t = t.detach().cpu()
    if mask is not None:
        mask = mask.detach().cpu().bool()
        return t[mask].tolist()
    return t.tolist()


def _decode_ids(tokenizer, ids: torch.Tensor, mask: Optional[torch.Tensor] = None) -> str:
    """Decode token ids to text, optionally applying a boolean mask first."""
    ids = ids.detach().cpu()
    if mask is not None:
        mask = mask.detach().cpu().bool()
        ids = ids[mask]
    return tokenizer.decode(ids.tolist(), skip_special_tokens=False)


def save_batch_to_jsonl(
    batch,                        # DataProto
    path: str,
    tag: str = "",                # short label printed in each row
    max_rows: int = 0,            # 0 = all rows
    tokenizer=None,               # if provided, decode prompt/response/input_ids
) -> None:
    """
    Save a DataProto to a JSONL file.  Each line = one row with:
      - __meta__: {tag, row_idx, total_rows, agent_type}
      - decoded:  {prompt_text, response_text, trained_tokens_text}  (if tokenizer given)
      - batch:    {field: serialised_value}  (tensor fields summarised or decoded)
      - non_tensor: {field: value}

    For selection and summary rows (agent_type != "policy"), the following fields
    are stored as full trimmed arrays (real tokens only, padding stripped):
      - responses           → response_token_ids (int list)
      - rollout_log_probs   → per-token logprobs (float list, same length)
      - token_level_scores  → sparse reward array (float list)
      - token_level_rewards → same as scores at rollout time (float list)

    Decode logic (when tokenizer provided):
      - prompt_text:        prompts[i] decoded with left-pad stripped (attention_mask applied)
      - response_text:      responses[i] decoded with response_mask applied
      - trained_tokens_text: the tokens the model is trained to predict
                            = input_ids[i, prompt_len : prompt_len+response_len][response_mask[i]]
                            Note: causal LM trains token t+1 given context t, so response_mask[t]
                            covers input_ids positions prompt_len..end; the actual loss target at
                            position p is input_ids[p+1].  We show the response tokens under mask
                            (the inputs, not the shifted labels) to make the training window clear.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    n = len(batch) if hasattr(batch, "__len__") else (
        next(iter(batch.batch.values())).shape[0] if batch.batch else 0
    )
    if max_rows > 0:
        n = min(n, max_rows)

    agent_types = batch.non_tensor_batch.get("agent_type", None)

    # Pre-extract tensors we need for decoding (may be absent in some batch types)
    b = batch.batch
    t_prompts      = b.get("prompts", None)        # [B, prompt_len]
    t_responses    = b.get("responses", None)       # [B, response_len]
    t_input_ids    = b.get("input_ids", None)       # [B, prompt_len + response_len]
    t_attn_mask    = b.get("attention_mask", None)  # [B, prompt_len + response_len]
    t_resp_mask    = b.get("response_mask", None)   # [B, response_len]

    with open(path, "w", encoding="utf-8") as f:
        for i in range(n):
            row: dict = {}

            # ── meta ──────────────────────────────────────────────────────
            agent_type_i = str(agent_types[i]) if agent_types is not None else "policy"
            row["__meta__"] = {
                "tag": tag,
                "row_idx": i,
                "total_rows": n,
                "agent_type": agent_type_i,
            }

            # For selection/summary rows, store full response token arrays so
            # downstream analysis can verify logprobs and token ids exactly.
            is_skill_row = agent_type_i in ("selection", "summary")
            resp_mask_i = t_resp_mask[i] if t_resp_mask is not None else None

            # ── decoded text (only when tokenizer available) ───────────────
            if tokenizer is not None:
                decoded = {}
                try:
                    # prompt: strip left padding via attention_mask's prompt portion
                    if t_prompts is not None and t_attn_mask is not None:
                        prompt_len = t_prompts.shape[1]
                        prompt_attn = t_attn_mask[i, :prompt_len]  # mask for prompt portion
                        decoded["prompt_text"] = _decode_ids(tokenizer, t_prompts[i], prompt_attn)
                    elif t_input_ids is not None and t_attn_mask is not None and t_responses is not None:
                        prompt_len = t_input_ids.shape[1] - t_responses.shape[1]
                        prompt_attn = t_attn_mask[i, :prompt_len]
                        decoded["prompt_text"] = _decode_ids(tokenizer, t_input_ids[i, :prompt_len], prompt_attn)

                    # response: apply response_mask to strip padding
                    if t_responses is not None and t_resp_mask is not None:
                        decoded["response_text"] = _decode_ids(tokenizer, t_responses[i], t_resp_mask[i])
                    elif t_responses is not None:
                        decoded["response_text"] = _decode_ids(tokenizer, t_responses[i])

                    # trained_tokens: the response tokens under response_mask
                    # (these are the INPUT tokens at positions where loss is computed;
                    #  causal LM loss target = next token, so actual label = token at pos+1)
                    if t_resp_mask is not None and t_responses is not None:
                        decoded["trained_tokens_text"] = _decode_ids(
                            tokenizer, t_responses[i], t_resp_mask[i]
                        )
                        decoded["trained_token_count"] = int(t_resp_mask[i].sum().item())

                    # full sequence (prompt+response) with padding stripped
                    if t_input_ids is not None and t_attn_mask is not None:
                        decoded["full_seq_text"] = _decode_ids(tokenizer, t_input_ids[i], t_attn_mask[i])

                except Exception as e:
                    decoded["__decode_error__"] = repr(e)
                row["decoded"] = decoded

            # ── tensor batch fields ────────────────────────────────────────
            row["batch"] = {}
            for key, tensor in b.items():
                if tokenizer is not None and key in (
                    "prompts", "responses", "input_ids",
                    "attention_mask", "response_mask",
                ):
                    # Already captured in decoded; just store shape
                    row["batch"][key] = {"__tensor_shape__": list(tensor[i].shape)}
                elif is_skill_row and key in _FULL_ARRAY_KEYS_FOR_SKILL:
                    # Store full trimmed array for selection/summary so logprobs and
                    # token ids can be inspected exactly (padding stripped by response_mask).
                    # For all 4 fields, trimming by response_mask is correct:
                    #   - responses / rollout_log_probs: trim to real response tokens
                    #   - token_level_scores / token_level_rewards: reward is at the last
                    #     real token (inside response_mask), so it is preserved after trim.
                    row["batch"][key] = {
                        "__full_array__": True,
                        "__tensor_shape__": list(tensor[i].shape),
                        "data": _tensor_to_trimmed_list(tensor[i], resp_mask_i),
                        "data_trimmed_by_response_mask": True,
                    }
                else:
                    row["batch"][key] = _to_serialisable(tensor[i])

            # ── non-tensor batch fields ────────────────────────────────────
            row["non_tensor"] = {}
            for key, arr in batch.non_tensor_batch.items():
                val = arr[i] if hasattr(arr, "__getitem__") else arr
                row["non_tensor"][key] = _to_serialisable(val)

            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"[batch_debug_save] Saved {n} rows → {path}  (tag={tag!r})")
