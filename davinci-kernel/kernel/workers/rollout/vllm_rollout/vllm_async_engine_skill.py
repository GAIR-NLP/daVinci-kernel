"""
SkillAwareMultiIterAsyncvLLMEngine

Extends MultiIterAsyncvLLMEngine with three skill-aware agents:
  1. Selection Agent  – choose k skill schemes; single-turn; logprobs recorded
  2. Policy Agent     – inherits parent multi-turn rollout with injected skill context
  3. Summary Agent    – single-turn agent; calls update_skill_library once; triggered conditionally

When skill.enable=false the parent's generate_sequences() is called unmodified.

All three agents use the latest model weights via the shared vLLM engine.
Logprob extraction is identical to _process_single_turn L1891-1897 in the parent.

DataProto returned by generate_sequences() contains all three agent types,
marked by non_tensor_batch["agent_type"] in {"policy", "selection", "summary"}.
The kernel_trainer.py calls compute_mixed_batch_advantages() on the combined batch.
"""

import asyncio
import json
import logging
import os
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

import numpy as np
import ray
import torch
from omegaconf import DictConfig
from tensordict import TensorDict
from verl import DataProto
from vllm import SamplingParams
from vllm.inputs import TokensPrompt

from kernel.workers.rollout.vllm_rollout.vllm_async_engine_multi_iter import (
    MultiIterAsyncvLLMEngine as _MultiIterActorClass,
)
# Ray does not allow subclassing @ray.remote-decorated classes.
# __ray_actor_class__ exposes the underlying plain Python class,
# which can be subclassed normally; @ray.remote is then applied
# only to the derived class.
_MultiIterBase = _MultiIterActorClass.__ray_actor_class__
from kernel.skill.skill_library import NewSkill, SkillLibrary
from kernel.skill.skill_prompt_builder import (
    get_selection_tools,
    get_summary_tools,
    build_selection_messages,
    build_summary_initial_messages,
    build_summary_verify_messages,
    extract_task_description,
    inject_skill_into_messages,
)


# ---------------------------------------------------------------------------
# Internal data classes
# ---------------------------------------------------------------------------

@dataclass
class SelectionTrajectory:
    """
    One independent selection agent call for ONE task: selects ONE scheme.

    We run k independent calls per task → k SelectionTrajectory objects.
    Plus one implicit null-scheme entry (no inference, reward set directly).
    Total k+1 trajectories per task → LOO advantage across k+1 gives valid gradients.
    """
    original_uid: str
    scheme_idx: int                      # 0..k-1 for real inferences; k for null
    prompt_ids: List[int]                # token ids of the selection prompt
    response_token_ids: List[int]        # token ids of the response
    logprobs: List[float]                # per-token log-prob, same length as response_token_ids
    selected_paths: List[str]            # skill paths this trajectory selected (parsed from response)
    skill_context: Optional[str]         # joined full content of selected skills (None = no skills)
    reward: float = 0.0                  # filled after policy rollout: max policy return for this scheme


@dataclass
class SummaryTrajectory:
    """
    Stores a single-turn Summary agent trajectory.
    Aligned with run_skill_inference.py: one inference, one update_skill_library call.
    """
    summary_group_id: str               # "{original_uid}_summary"
    original_uid: str
    prompt_ids: List[int]               # token ids of the prompt
    response_token_ids: List[int]       # token ids of the response
    logprobs: List[float]               # per-token log-prob
    new_skills: List[NewSkill] = field(default_factory=list)
    verify_speedup: float = 0.0         # speedup of the best verified new skill


# ---------------------------------------------------------------------------
# SkillAwareMultiIterAsyncvLLMEngine
# ---------------------------------------------------------------------------

@ray.remote(num_cpus=1)
class SkillAwareMultiIterAsyncvLLMEngine(_MultiIterBase):
    """
    Skill-aware rollout engine.

    Inherits all policy rollout logic from MultiIterAsyncvLLMEngine.
    Adds:
      - Skill selection (runs BEFORE policy rollout; expands prompts)
      - Skill summary / verification (runs AFTER policy rollout; conditional)
      - Packs all agent trajectories into one DataProto for unified training

    Compatibility: when skill.enable=False the parent class is used unchanged.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        cfg = self.config
        skill_cfg = cfg.get("skill", None)

        self._skill_enabled = (
            skill_cfg is not None and skill_cfg.get("enable", False)
        )
        # Log so we can confirm skill config was received on the remote actor
        print(
            f"[Skill] __init__  skill_cfg_found={skill_cfg is not None}"
            f"  enable={self._skill_enabled}"
            f"  k={skill_cfg.get('k', '?') if skill_cfg else '?'}"
        )
        if not self._skill_enabled:
            return

        self._scfg = skill_cfg
        self.skill_library = SkillLibrary(
            library_root=skill_cfg.library_root,
            global_step_prefix=skill_cfg.get("global_step_prefix", "global_step_"),
        )
        # If a specific snapshot file is given (eval mode), load it immediately.
        # Sets _initialized=True so generate_sequences() skips load_for_step().
        _library_file = skill_cfg.get("library_file", None)
        if _library_file:
            self.skill_library.load_from_file(_library_file)
            print(f"[Skill] engine init: loaded from file: {_library_file}  ({len(self.skill_library._skills)} skills)")
        self._k = int(skill_cfg.get("k", 3))
        # BM25 hyperparams for selection (v5 additions)
        self._top_bm25 = int(skill_cfg.get("top_bm25", 20))          # BM25 recall count for selection
        self._top_k_select = int(skill_cfg.get("top_k_select", self._k))  # LLM select count; falls back to k
        self._summary_parallel_s = int(skill_cfg.get("summary_parallel_s", 4))
        self._speedup_improve_thresh = float(skill_cfg.get("speedup_improve_thresh", 1.2))
        self._speedup_vs_baseline_thresh = float(skill_cfg.get("speedup_vs_baseline_thresh", 1.44))
        self._skill_verify_speedup_thresh = float(skill_cfg.get("skill_verify_speedup_thresh", 1.2))
        self._skill_verify_min_abs_speedup = float(skill_cfg.get("verify_min_absolute_speedup", 1.1))
        self._summary_require_turn1_correct = bool(skill_cfg.get("summary_require_turn1_correct", True))
        self._max_new_skills = int(skill_cfg.get("max_new_skills_per_step", 2))
        self._selection_temperature = float(skill_cfg.get("selection_temperature", 0.8))
        self._summary_temperature = float(skill_cfg.get("summary_temperature", 0.9))
        self._selection_max_tokens = int(skill_cfg.get("selection_max_tokens", 1024))
        self._summary_max_tokens = int(skill_cfg.get("summary_max_tokens", 2048))
        # Skill library flush frequency: write JSONL every N training steps.
        # Must be a divisor of trainer.save_freq (model checkpoint frequency) so
        # that every model checkpoint always has a corresponding skill snapshot.
        # Default 1 = flush every step.
        self._skill_flush_freq = int(skill_cfg.get("save_freq", 1))
        self._use_bm25_direct = bool(skill_cfg.get("use_bm25_direct", False))
        if self._use_bm25_direct and bool(skill_cfg.get("train_selection", True)):
            raise ValueError(
                "[Skill] use_bm25_direct=True but train_selection=True: "
                "BM25-direct mode produces no selection trajectories, so training the "
                "selection model is meaningless. Set skill.train_selection=False "
                "(SKILL_TRAIN_SELECTION=False in the launch script)."
            )
        # _ckpt_save_freq is validated lazily on the first generate_sequences call
        # when the trainer injects meta_info["ckpt_save_freq"].
        self._ckpt_save_freq: Optional[int] = None

    # ===========================================================================
    # Public entry point
    # ===========================================================================

    def drain_staged_skills(self) -> list:
        """Return and clear this worker's staged skill list."""
        if not self._skill_enabled:
            return []
        drained = list(self.skill_library._staged)
        self.skill_library._staged = []
        return drained

    def set_skill_library(self, skills: list, snapshot_step: int) -> None:
        """Broadcast: overwrite this worker's in-memory skill library.

        Called on all non-rank-0 workers after rank-0 flushes, so every worker
        has the same skill set without reading from disk.
        """
        if not self._skill_enabled:
            return
        self.skill_library.set_skills(skills, snapshot_step)

    def flush_skill_library(self, global_step: int, start_step: int = 0,
                             extra_staged: list | None = None) -> "dict | tuple[dict, list]":
        """
        Called by the trainer after a successful update_actor() to persist staged
        skills to disk.  Only effective when skill.enable=True; no-op otherwise.
        This is a synchronous method so the trainer can call it via Ray remote.

        Rules:
          - On the very first call (global_step == start_step), always writes a
            snapshot even if empty, so there is always a baseline at start_step.
          - Only flushes when (global_step - start_step) % skill.save_freq == 0
          - Always writes a snapshot (even if no new skills staged), so that
            every skill save step has a complete on-disk record.
        """
        if not self._skill_enabled:
            return {"skill/library_size": 0, "skill/new_this_step": 0, "skill/flush_skipped": True}
        if global_step < start_step:
            return {"skill/library_size": 0, "skill/new_this_step": 0, "skill/flush_skipped": True}
        if (global_step - start_step) % self._skill_flush_freq != 0:
            print(f"[SkillLibrary] flush_skill_library: skip (step={global_step}  "
                  f"start_step={start_step}  flush_freq={self._skill_flush_freq}  "
                  f"staged={len(self.skill_library._staged)})")
            return {"skill/library_size": len(self.skill_library._skills), "skill/new_this_step": 0, "skill/flush_skipped": True}
        # Merge extra_staged (collected from other DP workers) into local staged
        if extra_staged:
            self.skill_library._staged.extend(extra_staged)
        metrics = self.skill_library.flush_staged_skills(global_step)
        # Return (metrics, skills) so async_server can broadcast the skill list
        # to other DP workers without a separate round-trip.
        return metrics, list(self.skill_library._skills)

    def get_skill_library_size(self) -> int:
        """Return the number of skills currently in the library (for trainer gating)."""
        if not self._skill_enabled:
            return 0
        return len(self.skill_library._skills)

    async def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        if not self._skill_enabled:
            return await super().generate_sequences(prompts, **kwargs)

        global_step = prompts.meta_info.get("global_step", 0)
        is_validate = prompts.meta_info.get("validate", False)
        skill_validate = prompts.meta_info.get("skill_validate", False)
        # start_step caps which skill snapshots are valid for this run
        # (avoids loading snapshots written by a previous run from a later checkpoint)
        start_step = int(prompts.meta_info.get("skill_start_step", 0))

        B = len(prompts)
        # On first call or checkpoint resume, load from disk.
        # During normal training, skill list is kept up-to-date via broadcast
        # after each flush, so no disk read is needed.
        if not self.skill_library._initialized:
            self.skill_library.load_for_step(global_step, start_step=start_step)
        n_skills = len(self.skill_library.list_rel_paths())
        print(f"[Skill] step={global_step}  B={B}  library_size={n_skills}  validate={is_validate}  skill_validate={skill_validate}")

        # Lazily validate skill.save_freq divides trainer.save_freq once the
        # trainer has injected ckpt_save_freq into meta_info.
        ckpt_save_freq = prompts.meta_info.get("ckpt_save_freq", None)
        if ckpt_save_freq is not None and self._ckpt_save_freq is None:
            self._ckpt_save_freq = int(ckpt_save_freq)
            if self._ckpt_save_freq % self._skill_flush_freq != 0:
                raise ValueError(
                    f"[SkillLibrary] skill.save_freq={self._skill_flush_freq} must divide "
                    f"trainer.save_freq={self._ckpt_save_freq}. "
                    f"Please set SKILL_SAVE_FREQ to a divisor of {self._ckpt_save_freq}."
                )

        if is_validate and skill_validate:
            print(f"[Skill] → skill-injected validation  B={B}")
            return await self._validate_with_skill(prompts, **kwargs)

        if is_validate:
            print(f"[Skill] → plain validation (no skill injection)  B={B}")
            return await super().generate_sequences(prompts, **kwargs)

        if self._use_bm25_direct:
            print(f"[Skill] → bm25_direct mode  B={B}  k={self._k}  top_k_select={self._top_k_select}")
            return await self._generate_bm25_direct(prompts, **kwargs)

        # ── Step 1: Selection ─────────────────────────────────────────────
        print(f"[Skill] Step1 selection  B={B}  k={self._k}  top_k_select={self._top_k_select}  top_bm25={self._top_bm25}  total_inferences={B * self._k}")
        import time as _time
        _t0 = _time.time()
        skill_contexts_per_task, selection_trajs_per_task = \
            await self._run_selection_batch(prompts)
        _sel_time = _time.time() - _t0
        # Count how many tasks got at least one non-null scheme
        n_with_skill = sum(
            any(ctx is not None for ctx in ctxs)
            for ctxs in skill_contexts_per_task
        )
        print(f"[Skill] Step1 done  elapsed={_sel_time:.1f}s  tasks_with_skill={n_with_skill}/{B}")

        # ── Step 2: Expand prompts (B → B*(k+1)) ─────────────────────────
        expanded_prompts = self._expand_prompts_with_schemes(prompts, skill_contexts_per_task)
        print(f"[Skill] Step2 expanded  {B} → {len(expanded_prompts)} prompts  (k+1={self._k + 1})")

        # ── Step 3: Policy rollout (parent, all schemes) ──────────────────
        print(f"[Skill] Step3 policy rollout start  n_prompts={len(expanded_prompts)}")
        _t0 = _time.time()
        policy_output: DataProto = await super().generate_sequences(expanded_prompts, **kwargs)
        _pol_time = _time.time() - _t0
        if policy_output is None or len(policy_output) == 0:
            print("[Skill] Step3 policy rollout returned empty, skipping")
            return policy_output
        print(f"[Skill] Step3 policy rollout done  elapsed={_pol_time:.1f}s  output_rows={len(policy_output)}")

        # ── Step 4: Assign selection rewards ─────────────────────────────
        selection_trajs_per_task = self._assign_selection_rewards(
            selection_trajs_per_task, policy_output,
        )
        # ── Print per-task selection reward summary ────────────────────────
        for b_log in range(B):
            trajs = selection_trajs_per_task[b_log]
            uid_short = str(prompts.non_tensor_batch["uid"][b_log])[:20]
            print(f"[Skill] Step4 rewards  task[{b_log}] uid={uid_short}")
            for t in trajs:
                if not t.response_token_ids:
                    print(f"[Skill]   scheme={t.scheme_idx}  (null)  reward={t.reward:.4f}")
                else:
                    names = [
                        (self.skill_library.get_skill_meta(p).name
                         if self.skill_library.get_skill_meta(p) else p)
                        for p in t.selected_paths
                    ]
                    print(f"[Skill]   scheme={t.scheme_idx}  skills={names}  reward={t.reward:.4f}")

        # ── Step 5: Conditional summary ───────────────────────────────────
        print(f"[Skill] Step5 checking summary trigger  thresholds={self._speedup_improve_thresh}x/turn1  {self._speedup_vs_baseline_thresh}x/baseline")
        summary_output: Optional[DataProto] = None
        triggered, best_traj_list = self._collect_trigger_trajectories(
            policy_output, prompts,
            skill_contexts_per_task=skill_contexts_per_task,
        )
        if triggered and best_traj_list:
            print(f"[Skill] Step5 summary triggered  n_tasks={len(best_traj_list)}  s={self._summary_parallel_s}  total_trajs={len(best_traj_list) * self._summary_parallel_s}")
            for bt in best_traj_list:
                print(f"[Skill]   trigger: uid={bt['original_uid'][:20]}  turn1={bt['turn1_speedup']:.2f}x → best={bt['best_turn_speedup']:.2f}x  injected={'yes' if bt.get('injected_skill_content') else 'null'}")
            _t0 = _time.time()
            summary_output = await self._run_summary_batch(best_traj_list)
            _sum_time = _time.time() - _t0
            n_sum_rows = len(summary_output) if summary_output is not None else 0
            print(f"[Skill] Step5 summary done  elapsed={_sum_time:.1f}s  output_rows={n_sum_rows}")
        else:
            print(f"[Skill] Step5 no summary triggered")

        # ── Step 6: Pack and return ───────────────────────────────────────
        all_selection_trajs = [t for task_trajs in selection_trajs_per_task for t in task_trajs]
        result = self._pack_all(policy_output, all_selection_trajs, summary_output)
        n_pol = sum(1 for at in result.non_tensor_batch.get("agent_type", []) if at == "policy")
        n_sel = sum(1 for at in result.non_tensor_batch.get("agent_type", []) if at == "selection")
        n_sum = sum(1 for at in result.non_tensor_batch.get("agent_type", []) if at == "summary")
        print(f"[Skill] Step6 packed  total={len(result)}  policy={n_pol}  selection={n_sel}  summary={n_sum}")

        # Write skill usage stats into meta_info for trainer to pick up and log to wandb
        n_policy_with_skill = sum(
            1 for at, si in zip(
                result.non_tensor_batch.get("agent_type", []),
                result.non_tensor_batch.get("skill_scheme_idx", []),
            )
            if at == "policy" and int(si) > 0
        )
        n_staged = len(self.skill_library._staged)
        result.meta_info["skill_step_stats"] = {
            "skill/library_size":      len(self.skill_library._skills),
            "skill/tasks_with_skill":  n_with_skill,
            "skill/tasks_total":       B,
            "skill/policy_rows_with_skill": n_policy_with_skill,
            "skill/policy_rows_total": n_pol,
            "skill/selection_rows":    n_sel,
            "skill/summary_rows":      n_sum,
            "skill/summary_triggered": int(triggered),
            "skill/staged_pending":    n_staged,
        }
        return result

    async def _validate_with_skill(self, prompts: DataProto, **kwargs) -> DataProto:
        """
        Validation pass with the best greedy skill scheme injected.
        """
        if not self.skill_library.list_rel_paths():
            print(f"[Skill] _validate_with_skill: library empty, falling back to plain validation")
            return await super().generate_sequences(prompts, **kwargs)

        raw_prompts = prompts.non_tensor_batch["raw_prompt"]
        uids = prompts.non_tensor_batch["uid"]
        B = len(raw_prompts)
        print(f"[Skill] _validate_with_skill: B={B}  running greedy selection (T=0)")

        sel_tasks = []
        for b in range(B):
            task_msgs = (
                list(raw_prompts[b])
                if not isinstance(raw_prompts[b], list) else raw_prompts[b]
            )
            sel_tasks.append(
                self._select_skills_for_one(
                    original_uid=str(uids[b]),
                    task_messages=task_msgs,
                    scheme_idx=0,
                )
            )

        orig_temp = self._selection_temperature
        self._selection_temperature = 0.0
        try:
            sel_trajs: List[SelectionTrajectory] = await asyncio.gather(*sel_tasks)
        finally:
            self._selection_temperature = orig_temp

        skill_contexts_single: List[Optional[str]] = [t.skill_context for t in sel_trajs]
        n_injected = sum(1 for ctx in skill_contexts_single if ctx is not None)
        print(f"[Skill] _validate_with_skill: greedy selection done  injected={n_injected}/{B}")
        for b in range(min(3, B)):
            paths = sel_trajs[b].selected_paths
            print(f"[Skill]   task[{b}] selected_paths={paths}")

        # Save selection trajectories if a path is provided in meta_info
        _sel_save_path = prompts.meta_info.get("selection_save_path", None)
        if _sel_save_path:
            import json as _json, os as _os
            _os.makedirs(_os.path.dirname(_os.path.abspath(_sel_save_path)), exist_ok=True)
            _tok = getattr(self, "tokenizer", None)
            with open(_sel_save_path, "a", encoding="utf-8") as _f:
                for b, t in enumerate(sel_trajs):
                    # Decode prompt and response text if tokenizer available
                    _prompt_text = (
                        _tok.decode(t.prompt_ids, skip_special_tokens=False)
                        if _tok and t.prompt_ids else None
                    )
                    _response_text = (
                        _tok.decode(t.response_token_ids, skip_special_tokens=False)
                        if _tok and t.response_token_ids else None
                    )
                    # Raw task messages (what the policy sees as input, before skill injection)
                    _task_messages = None
                    if b < len(raw_prompts):
                        _rp = raw_prompts[b]
                        _task_messages = list(_rp) if not isinstance(_rp, list) else _rp
                    _rec = {
                        "uid": t.original_uid,
                        "task_messages": _task_messages,
                        "prompt_text": _prompt_text,
                        "response_text": _response_text,
                        "selected_skills": t.selected_paths,
                        "skill_context": t.skill_context,
                        "injected": t.skill_context is not None,
                        "logprobs": t.logprobs,
                    }
                    _f.write(_json.dumps(_rec, ensure_ascii=False) + "\n")
            print(f"[Skill] _validate_with_skill: selection data saved to {_sel_save_path}  ({len(sel_trajs)} records)")

        from copy import deepcopy
        injected_prompts_list = []
        for b in range(B):
            p = prompts[b:b+1]
            p = DataProto(
                batch=p.batch.clone(),
                non_tensor_batch={k: deepcopy(v) for k, v in p.non_tensor_batch.items()},
                meta_info=dict(p.meta_info),
            )
            ctx = skill_contexts_single[b]
            if ctx is not None:
                raw = p.non_tensor_batch.get("raw_prompt", None)
                if raw is not None and len(raw) > 0:
                    original_msgs = raw[0] if isinstance(raw[0], list) else list(raw[0])
                    injected_msgs = inject_skill_into_messages(
                        original_msgs, ctx, skill_config=self._scfg
                    )
                    arr = np.empty(1, dtype=object)
                    arr[0] = injected_msgs
                    p.non_tensor_batch["raw_prompt"] = arr
            # Normalize raw_prompt to 1D object array so DataProto.concat
            # doesn't fail when mixing injected (always 1D) and non-injected
            # samples (may inherit 2D shape from the original batch slice).
            # 2D shape (1, n_turns) means each element is a message dict;
            # raw[0] retrieves the full first-row message array → list of dicts.
            raw = p.non_tensor_batch.get("raw_prompt", None)
            if raw is not None and raw.ndim != 1:
                arr_1d = np.empty(1, dtype=object)
                arr_1d[0] = list(raw[0])
                p.non_tensor_batch["raw_prompt"] = arr_1d
            injected_prompts_list.append(p)

        injected_prompts = DataProto.concat(injected_prompts_list)
        print(f"[Skill] _validate_with_skill: running parent validation with injected prompts")
        return await super().generate_sequences(injected_prompts, **kwargs)

    # ===========================================================================
    # BM25-direct mode helpers
    # ===========================================================================

    async def _get_bm25_skill_contexts(self, prompts: DataProto) -> List[Optional[str]]:
        """
        BM25 retrieval only — no LLM inference.
        Returns one skill_context string per task (None when library is empty or no hit).
        """
        raw_prompts = prompts.non_tensor_batch["raw_prompt"]
        B = len(raw_prompts)
        skill_contexts: List[Optional[str]] = []
        for b in range(B):
            task_msgs = (
                list(raw_prompts[b]) if not isinstance(raw_prompts[b], list) else raw_prompts[b]
            )
            task_description = extract_task_description(task_msgs)
            skill_context: Optional[str] = None
            if self.skill_library.list_rel_paths() and self._top_bm25 > 0:
                candidates = self.skill_library.retrieve_bm25(
                    task_description, top_k=self._top_bm25
                )
                candidates = (candidates or [])[:self._top_k_select]
                if candidates:
                    parts = []
                    for p in candidates:
                        try:
                            parts.append(self.skill_library.get_skill_content(p))
                        except FileNotFoundError:
                            pass
                    skill_context = "\n\n".join(parts) if parts else None
            skill_contexts.append(skill_context)
        return skill_contexts

    async def _generate_bm25_direct(self, prompts: DataProto, **kwargs) -> DataProto:
        """
        BM25-direct rollout path:
          1. BM25 retrieval for each task (no LLM selection inference).
          2. Expand B → B*(k+1) prompts, all with identical BM25-retrieved skill injection.
             Uses the same _scheme{j} uid convention so _collect_trigger_trajectories works.
          3. Policy rollout via parent.
          4. Optionally trigger summary agent.
          5. Pack: strip _scheme{j} suffix from uid so all n*(k+1) rollouts of the same
             task form one unified advantage group.  No selection trajectories are produced.
        """
        import time as _time
        global_step = prompts.meta_info.get("global_step", 0)
        B = len(prompts)

        # Step 1 – BM25 retrieval
        _t0 = _time.time()
        skill_contexts = await self._get_bm25_skill_contexts(prompts)
        n_with_skill = sum(1 for c in skill_contexts if c is not None)
        print(
            f"[Skill/BM25Direct] step={global_step}  B={B}"
            f"  tasks_with_skill={n_with_skill}  bm25_elapsed={_time.time()-_t0:.2f}s"
        )

        # Step 2 – Expand B → B*(k+1), all copies get the same BM25 skill context.
        # skill_contexts_per_task[b][j] = same ctx for all j so that
        # _collect_trigger_trajectories can look up the injected skill by scheme index.
        skill_contexts_per_task: List[List[Optional[str]]] = [
            [ctx] * (self._k + 1) for ctx in skill_contexts
        ]
        expanded_prompts = self._expand_prompts_with_schemes(prompts, skill_contexts_per_task)
        print(f"[Skill/BM25Direct] expanded {B} → {len(expanded_prompts)} (×{self._k+1})")

        # Step 3 – Policy rollout
        _t0 = _time.time()
        policy_output: DataProto = await super().generate_sequences(expanded_prompts, **kwargs)
        print(
            f"[Skill/BM25Direct] policy done  elapsed={_time.time()-_t0:.1f}s"
            f"  rows={len(policy_output) if policy_output is not None else 0}"
        )
        if policy_output is None or len(policy_output) == 0:
            return policy_output

        # Step 4 – Optional summary trigger
        summary_output: Optional[DataProto] = None
        triggered, best_traj_list = self._collect_trigger_trajectories(
            policy_output, prompts,
            skill_contexts_per_task=skill_contexts_per_task,
        )
        if triggered and best_traj_list:
            print(f"[Skill/BM25Direct] summary triggered  n_tasks={len(best_traj_list)}")
            summary_output = await self._run_summary_batch(best_traj_list)

        # Step 5 – Pack: collapse _scheme{j} → original uid (one advantage group per task)
        result = self._pack_bm25_direct(policy_output, summary_output)

        n_pol = sum(1 for at in result.non_tensor_batch.get("agent_type", []) if at == "policy")
        n_sum = sum(1 for at in result.non_tensor_batch.get("agent_type", []) if at == "summary")
        print(f"[Skill/BM25Direct] packed  total={len(result)}  policy={n_pol}  summary={n_sum}")

        result.meta_info["skill_step_stats"] = {
            "skill/library_size":           len(self.skill_library._skills),
            "skill/tasks_with_skill":       n_with_skill,
            "skill/tasks_total":            B,
            "skill/policy_rows_with_skill": n_pol,
            "skill/policy_rows_total":      n_pol,
            "skill/selection_rows":         0,
            "skill/summary_rows":           n_sum,
            "skill/summary_triggered":      int(triggered),
            "skill/staged_pending":         len(self.skill_library._staged),
        }
        return result

    def _pack_bm25_direct(
        self,
        policy_output: DataProto,
        summary_output: Optional[DataProto],
    ) -> DataProto:
        """
        Tag policy rows for bm25_direct mode.  uid keeps its _scheme{j} suffix so the
        trainer's skill_active expansion block (B → B*(k+1)) and subsequent batch.union()
        work exactly as in standard skill mode.

        With adv_cross_scheme=True, compute_mixed_batch_advantages normalises advantages
        across all k+1 schemes per task together — equivalent to one n*(k+1) rollout group.

        No selection trajectory rows are added (bm25_direct produces none).
        """
        n_policy = len(policy_output)

        # Tag as policy (mirrors _pack_all, uid intentionally NOT stripped)
        policy_output.non_tensor_batch["agent_type"] = np.full(n_policy, "policy", dtype=object)
        if "original_uid" not in policy_output.non_tensor_batch:
            policy_output.non_tensor_batch["original_uid"] = np.array([
                str(uid).rsplit("_scheme", 1)[0] if "_scheme" in str(uid) else str(uid)
                for uid in policy_output.non_tensor_batch["uid"]
            ], dtype=object)
        if "skill_scheme_idx" not in policy_output.non_tensor_batch:
            policy_output.non_tensor_batch["skill_scheme_idx"] = np.array([
                int(str(uid).rsplit("_scheme", 1)[1]) if "_scheme" in str(uid) else 0
                for uid in policy_output.non_tensor_batch["uid"]
            ], dtype=np.int32)
        policy_output.non_tensor_batch["summary_group_id"] = np.full(n_policy, "", dtype=object)
        policy_output.non_tensor_batch["skill_staged"] = np.zeros(n_policy, dtype=bool)

        if summary_output is None:
            return policy_output

        parts = [policy_output, summary_output]
        all_keys: set = set()
        for p in parts:
            all_keys.update(p.batch.keys())
        for p in parts:
            for key in all_keys:
                if key not in p.batch:
                    ref = next(q.batch[key] for q in parts if key in q.batch)
                    n = len(p)
                    if ref.dim() == 1:
                        p.batch[key] = torch.zeros(n, dtype=ref.dtype, device=ref.device)
                    else:
                        p.batch[key] = torch.zeros(
                            n, *ref.shape[1:], dtype=ref.dtype, device=ref.device
                        )
        return DataProto.concat(parts)

    # ===========================================================================
    # Step 1: Selection
    # ===========================================================================

    async def _run_selection_batch(
        self, prompts: DataProto
    ) -> Tuple[List[List[Optional[str]]], List[List[SelectionTrajectory]]]:
        """
        Run k independent selection inferences per task, plus one null scheme.

        Returns:
          skill_contexts_per_task[b][j]: str|None  — injected skill content for scheme j
            j in 0..k-1: from independent selection inference
            j == k:       None  (null scheme, no skill injection)
          selection_trajs_per_task[b]: List[SelectionTrajectory]  length k+1
            first k entries: real inferences
            last entry: null scheme (empty response, reward set later)
        """
        raw_prompts = prompts.non_tensor_batch["raw_prompt"]
        uids = prompts.non_tensor_batch["uid"]
        B = len(raw_prompts)

        # Run k inferences per task, all async in parallel
        tasks = []
        task_meta = []  # (b, scheme_idx)
        for b in range(B):
            orig_uid = str(uids[b])
            task_msgs = (
                list(raw_prompts[b])
                if not isinstance(raw_prompts[b], list) else raw_prompts[b]
            )
            for j in range(self._k):
                tasks.append(self._select_skills_for_one(orig_uid, task_msgs, scheme_idx=j))
                task_meta.append((b, j))

        results = await asyncio.gather(*tasks)

        # Assemble per-task output
        skill_contexts_per_task: List[List[Optional[str]]] = [
            [None] * (self._k + 1) for _ in range(B)
        ]
        selection_trajs_per_task: List[List[SelectionTrajectory]] = [
            [] for _ in range(B)
        ]

        for (b, j), traj in zip(task_meta, results):
            skill_contexts_per_task[b][j] = traj.skill_context
            selection_trajs_per_task[b].append(traj)

        # Add null scheme (j == k) for each task: no inference needed
        uids_arr = prompts.non_tensor_batch["uid"]
        for b in range(B):
            orig_uid = str(uids_arr[b])
            selection_trajs_per_task[b].append(SelectionTrajectory(
                original_uid=orig_uid,
                scheme_idx=self._k,
                prompt_ids=[],          # null scheme has no inference
                response_token_ids=[],
                logprobs=[],
                selected_paths=[],
                skill_context=None,     # no skills injected
                reward=0.0,             # filled in _assign_selection_rewards
            ))
            # skill_contexts_per_task[b][k] is already None

        return skill_contexts_per_task, selection_trajs_per_task

    async def _select_skills_for_one(
        self,
        original_uid: str,
        task_messages: List[dict],
        scheme_idx: int,
    ) -> SelectionTrajectory:
        """
        One independent selection inference: outputs ONE skill scheme.
        Prompt asks the model to select the best combination for this task.
        k independent calls per task produce k diverse selections via temperature sampling.
        """
        if not self.skill_library.list_rel_paths():
            return SelectionTrajectory(
                original_uid=original_uid,
                scheme_idx=scheme_idx,
                prompt_ids=[],
                response_token_ids=[],
                logprobs=[],
                selected_paths=[],
                skill_context=None,
                reward=0.0,
            )

        # BM25 two-stage retrieval: first recall top_bm25 candidates, then let the
        # LLM select top_k_select from those candidates (matches data-collection logic).
        task_description = extract_task_description(task_messages)
        bm25_candidates: Optional[List[str]] = None
        if self._top_bm25 > 0 and self.skill_library.list_rel_paths():
            bm25_candidates = self.skill_library.retrieve_bm25(
                task_description, top_k=self._top_bm25
            )
            # Fall back to full library when BM25 returns nothing
            if not bm25_candidates:
                bm25_candidates = None

        messages = build_selection_messages(
            task_messages, self.skill_library, self._top_k_select,
            skill_config=self._scfg,
            bm25_candidate_paths=bm25_candidates,
            seed=scheme_idx,   # deterministic sub-sample per scheme for reproducibility
        )
        prompt_ids = self.tokenizer.apply_chat_template(
            messages,
            tools=get_selection_tools(self._top_k_select),
            add_generation_prompt=True,
            tokenize=True,
        )
        # Log selection input — print full messages so we can verify prompt quality
        n_candidates = len(bm25_candidates) if bm25_candidates else len(self.skill_library.list_rel_paths())
        print(f"[Skill] selection_input  uid={original_uid[:16]}  j={scheme_idx}  "
              f"prompt_tokens={len(prompt_ids)}  bm25_candidates={n_candidates}")
        for _msg in messages:
            print(f"[selection_agent] [{_msg.get('role', '?')}]\n{_msg.get('content', '')}")
        max_tokens = min(
            max(1, self.max_model_len - len(prompt_ids)),
            self._selection_max_tokens,
        )
        sampling_params = SamplingParams(
            temperature=self._selection_temperature,
            max_tokens=max_tokens,
            logprobs=1,   # identical to _process_single_turn L1789
        )
        request_id = f"sel_{original_uid}_j{scheme_idx}_{uuid4().hex[:8]}"
        results = None
        async for res in self.engine.generate(
            prompt=TokensPrompt(prompt_token_ids=prompt_ids),
            sampling_params=sampling_params,
            request_id=request_id,
        ):
            results = res

        if results is None or not results.outputs:
            print(f"[Skill] _select_skills_for_one: generation failed  uid={original_uid[:16]}  j={scheme_idx}")
            return SelectionTrajectory(
                original_uid=original_uid,
                scheme_idx=scheme_idx,
                prompt_ids=prompt_ids,
                response_token_ids=[],
                logprobs=[],
                selected_paths=[],
                skill_context=None,
                reward=0.0,
            )

        output = results.outputs[0]
        print(f"[selection_agent] [assistant]\n{output.text}")
        response_token_ids = list(output.token_ids)

        # Logprob extraction — identical to _process_single_turn L1891-L1897
        model_logprobs: List[float] = []
        if output.logprobs:
            for i in range(len(response_token_ids)):
                token_id = response_token_ids[i]
                model_logprobs.append(output.logprobs[i].get(token_id).logprob)
        assert len(response_token_ids) == len(model_logprobs)

        # Parse the response: select_skills(think, selected_skills: [{name, reason}])
        # Aligned with selected_skill_data.py — model returns skill names, not rel-paths.
        # Resolve name -> rel_path via SkillLibrary.get_rel_path_by_name().
        from kernel.skill.skill_summary_env import _parse_tool_call as _ptc
        tool_call = _ptc(output.text)
        think = ""
        selected_paths: List[str] = []
        if tool_call and tool_call.get("name") == "select_skills":
            think = tool_call.get("args", {}).get("think", "")
            raw_selected = tool_call.get("args", {}).get("selected_skills", [])
            if not isinstance(raw_selected, list):
                raw_selected = []
            for item in raw_selected:
                name = item.get("name", "") if isinstance(item, dict) else str(item)
                rel_path = self.skill_library.get_rel_path_by_name(name)
                if rel_path is not None:
                    selected_paths.append(rel_path)

        # Load skill content for injection; collect meta for logging
        skill_context = None
        if selected_paths:
            parts = []
            for p in selected_paths:
                try:
                    parts.append(self.skill_library.get_skill_content(p))
                except FileNotFoundError:
                    pass
            skill_context = "\n\n".join(parts) if parts else None

        # ── Print full selection result ─────────────────────────────────────
        print(f"[Skill] selection_result  uid={original_uid[:16]}  scheme={scheme_idx}  "
              f"resp_tokens={len(response_token_ids)}  "
              f"bm25_candidates={len(bm25_candidates) if bm25_candidates else 'all'}  "
              f"selected={len(selected_paths)}/{self._top_k_select}")
        if selected_paths:
            # Rebuild name→reason map from parsed tool call for logging
            raw_sel = tool_call.get("args", {}).get("selected_skills", []) if tool_call else []
            name_to_reason = {
                item.get("name", ""): item.get("reason", "")
                for item in raw_sel if isinstance(item, dict)
            }
            for sp in selected_paths:
                meta = self.skill_library.get_skill_meta(sp)
                name = meta.name if meta else sp
                reason = name_to_reason.get(name, "")
                print(f"[Skill]   -> {sp}  name={name!r}  reason={reason[:80]!r}")
        else:
            print(f"[Skill]   -> (no valid skills selected)  raw_text={output.text[:120]!r}")
        if think:
            print(f"[Skill]   think={think[:150]!r}")

        return SelectionTrajectory(
            original_uid=original_uid,
            scheme_idx=scheme_idx,
            prompt_ids=prompt_ids,
            response_token_ids=response_token_ids,
            logprobs=model_logprobs,
            selected_paths=selected_paths,
            skill_context=skill_context,
            reward=0.0,
        )

    # ===========================================================================
    # Step 2: Expand prompts
    # ===========================================================================

    def _expand_prompts_with_schemes(
        self,
        prompts: DataProto,
        skill_contexts_per_task: List[List[Optional[str]]],
    ) -> DataProto:
        """
        Expand B prompts to exactly B×(k+1) prompts (fixed, no dedup).
        Each scheme j in 0..k gets its own policy rollout.

        uid format: "{orig_uid}_scheme{j}"
        Duplicate skill sets across j's are allowed; reward merging is done
        AFTER rollout in _assign_selection_rewards (not at rollout time).
        """
        from copy import deepcopy

        expanded_list: List[DataProto] = []
        for b in range(len(prompts)):
            orig_uid = str(prompts.non_tensor_batch["uid"][b])
            contexts = skill_contexts_per_task[b]  # length k+1

            for j, ctx in enumerate(contexts):
                p = prompts[b:b+1]
                p = DataProto(
                    batch=p.batch.clone(),
                    non_tensor_batch={k: deepcopy(v) for k, v in p.non_tensor_batch.items()},
                    meta_info=dict(p.meta_info),
                )
                p.non_tensor_batch["uid"] = np.array([f"{orig_uid}_scheme{j}"], dtype=object)
                p.non_tensor_batch["original_uid"] = np.array([orig_uid], dtype=object)
                p.non_tensor_batch["skill_scheme_idx"] = np.array([j], dtype=np.int32)

                if ctx is not None:
                    raw = p.non_tensor_batch.get("raw_prompt", None)
                    if raw is not None and len(raw) > 0:
                        original_msgs = raw[0] if isinstance(raw[0], list) else list(raw[0])
                        injected_msgs = inject_skill_into_messages(
                            original_msgs, ctx, skill_config=self._scfg
                        )
                        # Store as 1D object array (shape=(1,)) to match the
                        # null-scheme raw_prompt shape, so DataProto.concat works.
                        arr = np.empty(1, dtype=object)
                        arr[0] = injected_msgs
                        p.non_tensor_batch["raw_prompt"] = arr

                expanded_list.append(p)

        if not expanded_list:
            return prompts
        return DataProto.concat(expanded_list)

    # ===========================================================================
    # Step 4: Assign selection rewards
    # ===========================================================================

    def _assign_selection_rewards(
        self,
        selection_trajs_per_task: List[List[SelectionTrajectory]],
        policy_output: DataProto,
        skill_contexts_per_task: Optional[List[List[Optional[str]]]] = None,
    ) -> List[List[SelectionTrajectory]]:
        """
        Assign aggregated policy reward to each SelectionTrajectory.

        Per scheme j: collect ALL n rollout returns from policy_output for
        scheme uid "{orig}_scheme{j}".  (Each turn row shares the same uid;
        we sum token_level_scores over the response to get a scalar return.)

        Same-recipe merging: selection trajs for task b that selected the same
        frozenset(selected_paths) share their n rollout pools.
        The merged pool has n×merge_size returns.

        Aggregation (controlled by skill.selection_reward_aggregation):
          "mean" (default): mean of all pooled returns
          "max":            max  of all pooled returns
        """
        uids = policy_output.non_tensor_batch["uid"]
        turn_indices = policy_output.batch.get("turn_indices", None)
        scores = policy_output.batch.get(
            "token_level_scores",
            policy_output.batch.get("token_level_rewards", None),
        )
        if scores is None:
            return selection_trajs_per_task

        aggregation = getattr(self._scfg, "selection_reward_aggregation", "mean")
        turn_mode = getattr(self._scfg, "selection_turn_reward_mode", "max")

        # Per-row reward: sum of token-level scores for that row (sparse: non-zero
        # only at the last real token of each turn row).
        row_rewards = scores.sum(dim=-1)  # (N,)

        # Collect per-trajectory reward for each (orig_uid, scheme_j).
        # Multi-turn: each trajectory has max_turns rows with per-turn rewards
        # r_1, r_2, ..., r_T (the speedup at each turn).  We reduce these to one
        # scalar per trajectory using selection_turn_reward_mode:
        #   "max"   – best turn reward (default)
        #   "mean"  – average across turns
        #   "first" – turn-0 reward only
        from collections import defaultdict as _dd
        scheme_returns: Dict = _dd(list)

        if turn_indices is not None:
            # uid → list of (turn_idx, reward) across all rows for that trajectory
            uid_turn_rewards: Dict[str, list] = _dd(list)
            for i, uid_str in enumerate(uids):
                uid_str = str(uid_str)
                t = int(turn_indices[i].item())
                if t == -1:
                    continue
                if "_scheme" not in uid_str:
                    continue
                uid_turn_rewards[uid_str].append((t, float(row_rewards[i].item())))

            for uid_str, turn_reward_pairs in uid_turn_rewards.items():
                orig_uid, _, scheme_str = uid_str.rpartition("_scheme")
                try:
                    j = int(scheme_str)
                except ValueError:
                    continue
                rewards_only = [r for _, r in turn_reward_pairs]
                if turn_mode == "first":
                    # sort by turn index, take turn-0
                    turn_reward_pairs.sort(key=lambda x: x[0])
                    traj_reward = turn_reward_pairs[0][1]
                elif turn_mode == "mean":
                    traj_reward = sum(rewards_only) / len(rewards_only)
                else:  # "max" (default)
                    traj_reward = max(rewards_only)
                scheme_returns[(orig_uid, j)].append(traj_reward)
        else:
            # Single-turn (summary verify or no turn_indices): reward = return
            for i, uid_str in enumerate(uids):
                uid_str = str(uid_str)
                if "_scheme" not in uid_str:
                    continue
                orig_uid, _, scheme_str = uid_str.rpartition("_scheme")
                try:
                    j = int(scheme_str)
                except ValueError:
                    continue
                scheme_returns[(orig_uid, j)].append(float(row_rewards[i].item()))

        # For each task, group selection trajs by frozenset(selected_paths)
        # to merge duplicate recipes, then compute the aggregated reward
        for task_trajs in selection_trajs_per_task:
            # recipe_key → list of scheme_j indices with identical selected_paths
            recipe_to_js: Dict = _dd(list)
            for traj in task_trajs:
                if not traj.response_token_ids:
                    # null scheme or failed inference: no selection output
                    recipe_key = "__null__"
                else:
                    recipe_key = frozenset(traj.selected_paths)
                recipe_to_js[recipe_key].append(traj.scheme_idx)

            # For each recipe group: pool all rollout returns, aggregate
            for recipe_key, js_in_group in recipe_to_js.items():
                orig_uid = task_trajs[0].original_uid if task_trajs else ""
                # Pool: n returns per scheme × merge_size schemes
                pooled: List[float] = []
                for j in js_in_group:
                    pooled.extend(scheme_returns.get((orig_uid, j), []))

                if not pooled:
                    aggregated = 0.0
                elif aggregation == "max":
                    aggregated = max(pooled)
                else:  # "mean" (default)
                    aggregated = sum(pooled) / len(pooled)

                # Assign to all trajs in this recipe group
                for traj in task_trajs:
                    if traj.scheme_idx in js_in_group:
                        traj.reward = aggregated

        return selection_trajs_per_task

    # ===========================================================================
    # Step 5: Trigger detection + summary
    # ===========================================================================

    def _collect_trigger_trajectories(
        self,
        policy_output: DataProto,
        original_prompts: DataProto,
        skill_contexts_per_task: Optional[List[List[Optional[str]]]] = None,
    ) -> Tuple[bool, List[dict]]:
        """
        Check whether any rollout satisfies the trigger condition:
          speedup(turn_t) >= speedup(turn_1) * speedup_improve_thresh  AND
          speedup(turn_t) >= speedup_vs_baseline_thresh

        If triggered, collect the per-task "best trajectory" dicts for summary.
        best_traj includes "injected_skill_content" — the actual skill text that
        was given to the policy agent for that scheme (None for null scheme).
        Returns (triggered, best_traj_list).
        """
        reward_extra_info = policy_output.non_tensor_batch.get("reward_extra_info", None)
        if reward_extra_info is None:
            return False, []

        uids = policy_output.non_tensor_batch["uid"]
        turn_indices = policy_output.batch.get("turn_indices", None)

        # Per-sample, per-turn speedup data
        # uid_str -> {turn_idx -> speedup}
        uid_turn_speedup: Dict[str, Dict[int, float]] = defaultdict(dict)
        uid_turn_code: Dict[str, Dict[int, str]] = defaultdict(dict)

        # Pre-decode all response tokens for code extraction
        # reward_extra_info has no "code" field; decode from responses tensor
        responses_t = policy_output.batch.get("responses", None)
        response_mask_t = policy_output.batch.get("response_mask", None)

        for i, uid_str in enumerate(uids):
            uid_str = str(uid_str)
            t_idx = int(turn_indices[i].item()) if turn_indices is not None else 1
            if t_idx == -1:
                continue
            info = reward_extra_info[i]
            if not isinstance(info, dict):
                continue
            speedup = float(info.get("performance", 0.0) or 0.0)
            # Decode response tokens then extract the Python code block.
            # reward_extra_info has no "code" field; the code lives in the response.
            code = ""
            if responses_t is not None and response_mask_t is not None:
                mask = response_mask_t[i].bool()
                valid_tokens = responses_t[i][mask].tolist()
                response_text = self.tokenizer.decode(valid_tokens, skip_special_tokens=True)
                from kernel.rewards.kernel_reward import extract_kernel_code
                code = extract_kernel_code(response_text)
            uid_turn_speedup[uid_str][t_idx] = speedup
            uid_turn_code[uid_str][t_idx] = code

        # Build original_uid -> list of per-rollout dicts
        orig_uid_to_rollouts: Dict[str, List[dict]] = defaultdict(list)
        for uid_str, turn_speedups in uid_turn_speedup.items():
            if "_scheme" not in uid_str:
                continue
            orig_uid, _, scheme_str = uid_str.rpartition("_scheme")
            if not scheme_str.isdigit():
                continue

            turns = sorted(turn_speedups.keys())
            if not turns:
                continue
            turn1_speedup = turn_speedups.get(min(turns), 0.0)
            best_turn = max(turns, key=lambda t: turn_speedups[t])
            best_speedup = turn_speedups[best_turn]

            if (
                best_speedup >= turn1_speedup * self._speedup_improve_thresh
                and best_speedup >= self._speedup_vs_baseline_thresh
                and (not self._summary_require_turn1_correct or turn1_speedup > 0)
            ):
                orig_uid_to_rollouts[orig_uid].append({
                    "uid_str": uid_str,
                    "orig_uid": orig_uid,
                    "turn1_speedup": turn1_speedup,
                    "best_turn_speedup": best_speedup,
                    "turn1_code": uid_turn_code[uid_str].get(min(turns), ""),
                    "best_turn_code": uid_turn_code[uid_str].get(best_turn, ""),
                })

        if not orig_uid_to_rollouts:
            return False, []

        # Build reverse map: original_uid → task index b (for skill context lookup)
        orig_uid_to_b = {
            str(original_prompts.non_tensor_batch["uid"][b]): b
            for b in range(len(original_prompts))
        }

        raw_prompts_map = {
            str(original_prompts.non_tensor_batch["uid"][b]): (
                list(original_prompts.non_tensor_batch["raw_prompt"][b])
                if not isinstance(original_prompts.non_tensor_batch["raw_prompt"][b], list)
                else original_prompts.non_tensor_batch["raw_prompt"][b]
            )
            for b in range(len(original_prompts))
        }
        reward_model_map = {
            str(original_prompts.non_tensor_batch["uid"][b]):
                original_prompts.non_tensor_batch.get("reward_model", [{}] * len(original_prompts))[b]
            for b in range(len(original_prompts))
        }
        extra_info_map = {
            str(original_prompts.non_tensor_batch["uid"][b]):
                original_prompts.non_tensor_batch.get("extra_info", [{}] * len(original_prompts))[b]
            for b in range(len(original_prompts))
        }

        best_traj_list: List[dict] = []
        for orig_uid, rollouts in orig_uid_to_rollouts.items():
            best = max(rollouts, key=lambda r: r["best_turn_speedup"])
            task_messages = raw_prompts_map.get(orig_uid, [])
            # Debug: log task_messages structure
            _tm_info = [(m.get('role','?'), len(m.get('content','')), repr(m.get('content','')[:80])) for m in task_messages] if isinstance(task_messages, list) and task_messages and isinstance(task_messages[0], dict) else repr(type(task_messages))
            print(f"[Skill] _collect: orig_uid={orig_uid[:24]}  task_messages={_tm_info}")
            reward_model = reward_model_map.get(orig_uid, {})
            if isinstance(reward_model, str):
                try:
                    reward_model = json.loads(reward_model)
                except Exception:
                    reward_model = {}
            extra_info = extra_info_map.get(orig_uid, {})
            if isinstance(extra_info, str):
                try:
                    extra_info = json.loads(extra_info)
                except Exception:
                    extra_info = {}
            ground_truth = reward_model.get("ground_truth", "")
            # entry_point: prefer reward_model, then extra_info, fallback "Model"
            entry_point = (
                reward_model.get("entry_point")
                or extra_info.get("entry_point")
                or "Model"
            )

            # Retrieve the skill content that was actually injected for the winning scheme.
            # skill_contexts_per_task[b][j] is str|None;
            # None means null scheme (no skills were given to policy).
            injected_skill_content: Optional[str] = None
            if skill_contexts_per_task is not None:
                b_idx = orig_uid_to_b.get(orig_uid)
                best_scheme_str = best.get("uid_str", "").rsplit("_scheme", 1)
                if len(best_scheme_str) == 2 and best_scheme_str[1].isdigit():
                    j = int(best_scheme_str[1])
                    if b_idx is not None and j < len(skill_contexts_per_task[b_idx]):
                        injected_skill_content = skill_contexts_per_task[b_idx][j]

            best_traj_list.append({
                "original_uid": orig_uid,
                "task_description": extract_task_description(task_messages),
                "task_messages": task_messages,
                "turn1_speedup": best["turn1_speedup"],
                "turn1_code": best["turn1_code"],
                "best_turn_speedup": best["best_turn_speedup"],
                "best_turn_code": best["best_turn_code"],
                "ground_truth": ground_truth,
                "entry_point": entry_point,
                "injected_skill_content": injected_skill_content,  # str|None
            })

        return True, best_traj_list

    async def _run_summary_batch(self, best_traj_list: List[dict]) -> Optional[DataProto]:
        """
        For each triggered task, run s parallel Summary inferences (single-turn each).
        Aligned with run_skill_inference.py: one inference, one update_skill_library call.
        Verify each proposed skill. Stage accepted skills in skill_library.
        """
        tasks = []
        for best_traj in best_traj_list:
            for s_idx in range(self._summary_parallel_s):
                tasks.append(self._run_one_summary(best_traj, s_idx))

        summary_trajs: List[SummaryTrajectory] = await asyncio.gather(*tasks)

        # Verify proposed skills in parallel
        verify_tasks = []
        verify_meta = []
        for straj in summary_trajs:
            orig_uid = straj.original_uid
            bt = next((b for b in best_traj_list if b["original_uid"] == orig_uid), None)
            if bt is None:
                for skill in straj.new_skills:
                    async def _null_verify():
                        return (False, 0.0)
                    verify_tasks.append(_null_verify())
                    verify_meta.append((straj, skill, 0.0))
                continue
            for skill in straj.new_skills:
                verify_tasks.append(self._verify_skill(skill, bt))
                verify_meta.append((straj, skill, bt["turn1_speedup"]))

        if verify_tasks:
            verify_results = await asyncio.gather(*verify_tasks)
        else:
            verify_results = []

        # Assign verify results
        for (straj, skill, turn1_speedup), (passed, speedup) in zip(verify_meta, verify_results):
            print(f"[Skill] verify  skill={skill.name!r}  uid={straj.original_uid[:16]}  "
                  f"passed={passed}  speedup={speedup:.3f}x  "
                  f"(turn1_baseline={turn1_speedup:.3f}x)")
            if passed:
                skill.verify_speedup = speedup
            else:
                skill.verify_speedup = 0.0
                straj.new_skills = [s for s in straj.new_skills if s is not skill]

        # Per original_uid: stage best verified skill
        uid_to_best_skill: Dict[str, Optional[NewSkill]] = {}
        for straj in summary_trajs:
            for skill in straj.new_skills:
                if skill.verify_speedup > 0:
                    prev = uid_to_best_skill.get(straj.original_uid)
                    if prev is None or skill.verify_speedup > prev.verify_speedup:
                        uid_to_best_skill[straj.original_uid] = skill

        for skill in uid_to_best_skill.values():
            if skill is not None:
                self.skill_library.stage_new_skill(skill)
                print(f"[Skill] staged  name={skill.name!r}  scope={skill.scope}  "
                      f"verify_speedup={skill.verify_speedup:.3f}x")

        n_staged = sum(1 for s in uid_to_best_skill.values() if s is not None)
        print(f"[Skill] _run_summary_batch done  staged={n_staged}  "
              f"total_trajs={len(summary_trajs)}")

        for straj in summary_trajs:
            straj.verify_speedup = max(
                (s.verify_speedup for s in straj.new_skills if s.verify_speedup > 0),
                default=0.0,
            )

        return self._pack_summary_trajs(summary_trajs)

    async def _run_one_summary(
        self, best_traj: dict, s_idx: int
    ) -> SummaryTrajectory:
        """
        Single-turn summary inference.
        Aligned with run_skill_inference.py: one call to update_skill_library(think, new_skills).
        No ReAct, no read_skill_files.
        Logprob extraction identical to _process_single_turn L1891-L1897.
        """
        from kernel.skill.skill_summary_env import parse_summary_response

        messages = build_summary_initial_messages(
            best_traj, self._max_new_skills,
            skill_config=self._scfg,
        )
        orig_uid = best_traj["original_uid"]
        summary_group_id = f"{orig_uid}_summary"

        _injected_preview = (best_traj.get("injected_skill_content") or "(null)")[:120].replace("\n", " | ")
        print(
            f"[Skill] summary_input  uid={orig_uid[:16]}  s={s_idx}"
            f"  turn1={best_traj.get('turn1_speedup', 0):.3f}x"
            f"  best={best_traj.get('best_turn_speedup', 0):.3f}x"
            f"  injected={_injected_preview!r}"
        )
        for _msg in messages:
            print(f"[summary_agent] [{_msg.get('role','?')}]\n{_msg.get('content','')}")

        prompt_ids = self.tokenizer.apply_chat_template(
            messages,
            tools=get_summary_tools(self._max_new_skills),
            add_generation_prompt=True,
            tokenize=True,
        )
        max_tokens = min(
            max(1, self.max_model_len - len(prompt_ids)),
            self._summary_max_tokens,
        )
        sampling_params = SamplingParams(
            temperature=self._summary_temperature,
            max_tokens=max_tokens,
            logprobs=1,  # identical to _process_single_turn L1789
        )
        request_id = f"sum{s_idx}_{orig_uid[:8]}_{uuid4().hex[:6]}"
        results = None
        try:
            async for res in self.engine.generate(
                prompt=TokensPrompt(prompt_token_ids=prompt_ids),
                sampling_params=sampling_params,
                request_id=request_id,
            ):
                results = res
        except Exception as e:
            logging.warning(f"[Summary] Generation failed: {e}")
            return SummaryTrajectory(
                summary_group_id=summary_group_id,
                original_uid=orig_uid,
                prompt_ids=prompt_ids,
                response_token_ids=[],
                logprobs=[],
            )

        if results is None or not results.outputs:
            print(f"[Skill] summary  uid={orig_uid[:16]}  s={s_idx}  generation returned None")
            return SummaryTrajectory(
                summary_group_id=summary_group_id,
                original_uid=orig_uid,
                prompt_ids=prompt_ids,
                response_token_ids=[],
                logprobs=[],
            )

        output = results.outputs[0]
        resp_token_ids = list(output.token_ids)

        # Logprob extraction — identical to _process_single_turn L1891-L1897
        logprobs: List[float] = []
        if output.logprobs:
            for i in range(len(resp_token_ids)):
                token_id = resp_token_ids[i]
                logprobs.append(output.logprobs[i].get(token_id).logprob)
        assert len(resp_token_ids) == len(logprobs)

        print(f"[summary_agent] [assistant]\n{output.text}")
        print(f"[Skill] summary  uid={orig_uid[:16]}  s={s_idx}  resp_tokens={len(resp_token_ids)}")

        new_skills = parse_summary_response(output.text, self._max_new_skills)

        return SummaryTrajectory(
            summary_group_id=summary_group_id,
            original_uid=orig_uid,
            prompt_ids=prompt_ids,
            response_token_ids=resp_token_ids,
            logprobs=logprobs,
            new_skills=new_skills,
            verify_speedup=0.0,
        )

    async def _verify_skill(
        self, new_skill: NewSkill, best_traj: dict
    ) -> Tuple[bool, float]:
        """
        Verify new_skill by running policy agent turn-1 with the augmented prompt.

        Inject content = original_injected_skills (what policy already had)
                       + new_skill content (the proposed addition)

        The baseline for comparison is best_traj["turn1_speedup"]:
        the speedup the policy achieved on turn-1 WITH the original skills.
        A new skill is accepted if the augmented prompt lifts this further
        by at least skill_verify_speedup_thresh.
        """
        task_messages = best_traj.get("task_messages", [])
        if not task_messages:
            print(f"[Skill] _verify_skill: no task_messages, skip  skill={new_skill.name!r}")
            return False, 0.0

        print(f"[Skill] _verify_skill start  skill={new_skill.name!r}  turn1_baseline={best_traj.get('turn1_speedup', 0.0):.3f}x")

        # Build the combined skill text: existing skills + new skill
        new_skill_text = (
            f"---\nname: {new_skill.name}\ndescription: {new_skill.description}\n"
            f"scope: {new_skill.scope}\ntags: {new_skill.tags}\n---\n\n{new_skill.content}"
        )
        original_content = best_traj.get("injected_skill_content")  # str | None
        if original_content:
            combined_content = original_content + "\n\n---\n\n" + new_skill_text
        else:
            combined_content = new_skill_text
        verify_messages = build_summary_verify_messages(
            task_messages, combined_content, skill_config=self._scfg
        )
        for _msg in verify_messages:
            c = _msg.get('content', '')
            print(f"[verify_agent] [{_msg.get('role', '?')}] len={len(c)}  head={repr(c[:120])}\n{c}")

        prompt_ids = self.tokenizer.apply_chat_template(
            verify_messages, add_generation_prompt=True, tokenize=True
        )
        max_tokens = min(
            max(1, self.max_model_len - len(prompt_ids)),
            self.config.rollout.response_length,
        )
        sampling_params = SamplingParams(
            temperature=0.0,   # greedy for reproducibility
            max_tokens=max_tokens,
            logprobs=1,
        )
        request_id = f"verify_{uuid4().hex[:8]}"
        results = None
        try:
            async for res in self.engine.generate(
                prompt=TokensPrompt(prompt_token_ids=prompt_ids),
                sampling_params=sampling_params,
                request_id=request_id,
            ):
                results = res
        except Exception as e:
            logging.warning(f"[Verify] Generation failed: {e}")
            return False, 0.0

        if results is None or not results.outputs:
            return False, 0.0

        output = results.outputs[0]
        print(f"[verify_agent] [assistant] (skill={new_skill.name!r})\n{output.text}")
        ground_truth = best_traj.get("ground_truth", "")
        entry_point = best_traj.get("entry_point", "")
        turn1_speedup = best_traj.get("turn1_speedup", 0.0)
        verify_uuid = f"verify_{uuid4().hex}"

        loop = asyncio.get_running_loop()
        try:
            env_result = await loop.run_in_executor(
                None,
                lambda: self.reward_fn(
                    list(output.token_ids),
                    output.text,
                    ground_truth,
                    entry_point,
                    verify_uuid,
                    return_full_state=True,
                ),
            )
        except Exception as e:
            logging.warning(f"[Verify] reward_fn failed: {e}")
            return False, 0.0

        env_state = env_result.get("env_state", {}) if isinstance(env_result, dict) else {}
        # env_state is the raw server result dict; key is "speedup", not "performance"
        speedup = float(env_state.get("speedup", 0.0) or 0.0)
        correctness = bool(env_state.get("correctness", False))
        passed = (
            correctness
            and speedup > 0
            and turn1_speedup > 0
            and speedup >= self._skill_verify_min_abs_speedup
            and speedup >= turn1_speedup * self._skill_verify_speedup_thresh
        )
        print(
            f"[Skill] verify_reward  skill={new_skill.name!r}"
            f"  correctness={correctness}  speedup={speedup:.3f}x"
            f"  turn1_baseline={turn1_speedup:.3f}x  thresh={self._skill_verify_speedup_thresh}x"
            f"  passed={passed}"
        )
        return passed, speedup

    # ===========================================================================
    # Step 6: Pack all trajectories into unified DataProto
    # ===========================================================================

    # Canonical non_tensor_batch key set shared by ALL three agent types.
    # Every key must exist in every row; missing keys get a neutral default.
    _NTB_KEYS = (
        "uid",              # str   — row identifier (includes agent prefix)
        "original_uid",     # str   — task-level uid (no scheme/sel/sum suffix)
        "agent_type",       # str   — "policy" | "selection" | "summary"
        "skill_scheme_idx", # int32 — scheme index; -1 for summary
        "summary_group_id", # str   — group key for summary LOO; "" for non-summary
        "skill_staged",     # bool  — True if this summary row produced a verified staged skill
        "num_turns",        # int32
        "contain_void_turn",# int32
        "finish_reasons",   # str
        "multiturn_messages",# object
        "global_turn_indices",# int32
        "reward_extra_info",# object dict
    )

    def _pack_all(
        self,
        policy_output: DataProto,
        selection_trajs: List[SelectionTrajectory],
        summary_output: Optional[DataProto],
    ) -> DataProto:
        """
        Tag policy rows, pack selection and summary, align all non_tensor keys,
        then concatenate into one DataProto.
        """
        n_policy = len(policy_output)

        # ── Tag policy rows ────────────────────────────────────────────────
        policy_output.non_tensor_batch["agent_type"] = np.full(
            n_policy, "policy", dtype=object
        )
        if "original_uid" not in policy_output.non_tensor_batch:
            policy_output.non_tensor_batch["original_uid"] = np.array([
                str(uid).rsplit("_scheme", 1)[0]
                for uid in policy_output.non_tensor_batch["uid"]
            ], dtype=object)
        # Fill neutral defaults for fields only selection/summary have.
        # skill_scheme_idx was already set per-scheme in _expand_prompts_with_schemes;
        # only fall back to zeros when it is genuinely absent (should not happen).
        if "skill_scheme_idx" not in policy_output.non_tensor_batch:
            policy_output.non_tensor_batch["skill_scheme_idx"] = np.array([
                int(str(uid).rsplit("_scheme", 1)[1]) if "_scheme" in str(uid) else 0
                for uid in policy_output.non_tensor_batch["uid"]
            ], dtype=np.int32)
        policy_output.non_tensor_batch["summary_group_id"] = np.full(
            n_policy, "", dtype=object
        )
        policy_output.non_tensor_batch["skill_staged"] = np.zeros(n_policy, dtype=bool)

        parts = [policy_output]

        sel_dp = self._pack_selection_trajs(selection_trajs, policy_output)
        if sel_dp is not None:
            parts.append(sel_dp)

        if summary_output is not None:
            parts.append(summary_output)

        if len(parts) == 1:
            return parts[0]

        # ── Align tensor keys before concat ───────────────────────────────
        # Collect the union of all tensor keys across parts
        all_keys: set = set()
        for p in parts:
            all_keys.update(p.batch.keys())

        for p in parts:
            for key in all_keys:
                if key not in p.batch:
                    # Fill missing key with zeros matching the shape of that key
                    # from whichever part has it
                    ref = next(q.batch[key] for q in parts if key in q.batch)
                    n = len(p)
                    if ref.dim() == 1:
                        p.batch[key] = torch.zeros(n, dtype=ref.dtype, device=ref.device)
                    else:
                        p.batch[key] = torch.zeros(
                            n, *ref.shape[1:], dtype=ref.dtype, device=ref.device
                        )

        return DataProto.concat(parts)

    def _pack_selection_trajs(
        self,
        selection_trajs: List[SelectionTrajectory],  # flat list: B*(k+1) entries
        policy_output: DataProto,
    ) -> Optional[DataProto]:
        """
        Convert selection trajectories to DataProto rows (one row per trajectory).

        Each SelectionTrajectory is an independent inference → one unique row.
        Null-scheme trajectories (scheme_idx == k, empty response) are skipped
        since there is no generated text to train on.

        After this function, the B*(k+1) per-task groups are fully preserved in
        non_tensor_batch["original_uid"], enabling correct LOO grouping in skill_adv.py:
          - k real inference rows   (agent_type="selection", scheme_idx 0..k-1)
          - 0 null rows             (null scheme has no trainable response)
        The null scheme's reward is still used in LOO via the policy rows for scheme k.
        """
        max_prompt_length = policy_output.batch["prompts"].shape[1]
        max_response_length = policy_output.batch["responses"].shape[1]
        pad_token_id = self.tokenizer.pad_token_id or 0
        device = policy_output.batch["prompts"].device

        all_prompts, all_responses, all_logprobs = [], [], []
        all_rewards, all_uids, all_orig_uids, all_scheme_idx = [], [], [], []

        for traj in selection_trajs:
            # Skip null scheme (no inference) and failed inferences
            if not traj.response_token_ids:
                continue
            all_prompts.append(traj.prompt_ids)
            all_responses.append(traj.response_token_ids)
            all_logprobs.append(traj.logprobs)
            all_rewards.append(float(traj.reward))
            all_uids.append(f"{traj.original_uid}_sel_scheme{traj.scheme_idx}")
            all_orig_uids.append(traj.original_uid)
            all_scheme_idx.append(traj.scheme_idx)

        if not all_prompts:
            return None

        padded_prompts = []
        for p in all_prompts:
            p = p[-max_prompt_length:]
            padded_prompts.append([pad_token_id] * (max_prompt_length - len(p)) + p)

        padded_responses, padded_logprobs, response_masks = [], [], []
        for r, lp in zip(all_responses, all_logprobs):
            r = r[:max_response_length];  lp = lp[:max_response_length]
            pad_len = max_response_length - len(r)
            padded_responses.append(r + [pad_token_id] * pad_len)
            padded_logprobs.append(lp + [-1.0] * pad_len)
            response_masks.append([1] * len(r) + [0] * pad_len)

        prompts_t   = torch.tensor(padded_prompts,   dtype=torch.long,    device=device)
        responses_t = torch.tensor(padded_responses, dtype=torch.long,    device=device)
        resp_mask_t = torch.tensor(response_masks,   dtype=torch.long,    device=device)
        logprobs_t  = torch.tensor(padded_logprobs,  dtype=torch.float32, device=device)
        input_ids_t = torch.cat([prompts_t, responses_t], dim=1)
        attn_mask_t = (input_ids_t != pad_token_id).long()
        pos_ids_t   = (attn_mask_t.cumsum(dim=1) - 1) * attn_mask_t

        valid_len = resp_mask_t.sum(dim=-1).long()
        token_rewards_t = torch.zeros(len(all_responses), max_response_length, device=device)
        for i, (vl, rw) in enumerate(zip(valid_len, all_rewards)):
            token_rewards_t[i, max(0, vl.item() - 1)] = rw

        n = len(all_prompts)
        turn_indices_t = torch.ones(n, dtype=torch.long, device=device)
        loss_mask_t    = torch.ones(n, dtype=torch.long, device=device)

        return DataProto(
            batch=TensorDict({
                "prompts":             prompts_t,
                "responses":           responses_t,
                "response_mask":       resp_mask_t,
                "input_ids":           input_ids_t,
                "attention_mask":      attn_mask_t,
                "position_ids":        pos_ids_t,
                "rollout_log_probs":   logprobs_t,
                "turn_indices":        turn_indices_t,
                "loss_mask":           loss_mask_t,
                "token_level_scores":  token_rewards_t,
                "token_level_rewards": token_rewards_t,
                "sample_indices":      torch.arange(n, dtype=torch.long, device=device),
            }, batch_size=n),
            non_tensor_batch={
                "uid":                  np.array(all_uids,      dtype=object),
                "original_uid":         np.array(all_orig_uids, dtype=object),
                "agent_type":           np.full(n, "selection", dtype=object),
                "skill_scheme_idx":     np.array(all_scheme_idx, dtype=np.int32),
                "summary_group_id":     np.full(n, "",          dtype=object),
                "skill_staged":         np.zeros(n,             dtype=bool),
                "num_turns":            np.ones(n,              dtype=np.int32),
                "contain_void_turn":    np.zeros(n,             dtype=np.int32),
                "finish_reasons":       np.full(n, "stop",      dtype=object),
                "multiturn_messages":   np.full(n, None,        dtype=object),
                "global_turn_indices":  np.ones(n,              dtype=np.int32),
                "reward_extra_info":    np.array([{}] * n,      dtype=object),
            },
        )

    def _pack_summary_trajs(
        self, summary_trajs: List[SummaryTrajectory]
    ) -> Optional[DataProto]:
        """
        Convert single-turn summary trajectories to DataProto.
        One row per trajectory (single inference, single turn_index=1).
        reward = verify_speedup placed at last real token position.
        """
        if not summary_trajs:
            return None

        max_prompt_length   = self.config.rollout.prompt_length
        max_response_length = self.config.rollout.response_length
        pad_token_id = self.tokenizer.pad_token_id or 0

        all_prompts, all_responses, all_logprobs = [], [], []
        all_rewards, all_uids, all_orig_uids, all_sum_group_ids = [], [], [], []
        all_skill_staged = []

        for idx, straj in enumerate(summary_trajs):
            if not straj.response_token_ids:
                continue  # failed inference — skip
            all_prompts.append(straj.prompt_ids)
            all_responses.append(straj.response_token_ids)
            all_logprobs.append(straj.logprobs)
            all_rewards.append(straj.verify_speedup)
            all_uids.append(f"{straj.summary_group_id}_{idx}")
            all_orig_uids.append(straj.original_uid)
            all_sum_group_ids.append(straj.summary_group_id)
            all_skill_staged.append(straj.verify_speedup > 0)

        if not all_prompts:
            return None

        device = torch.device("cpu")

        padded_prompts = []
        for p in all_prompts:
            p = p[-max_prompt_length:]
            padded_prompts.append([pad_token_id] * (max_prompt_length - len(p)) + p)

        padded_responses, padded_logprobs, response_masks = [], [], []
        for r, lp in zip(all_responses, all_logprobs):
            r  = r[:max_response_length]
            lp = lp[:max_response_length]
            pad_len = max_response_length - len(r)
            padded_responses.append(r  + [pad_token_id] * pad_len)
            padded_logprobs.append(lp  + [-1.0]         * pad_len)
            response_masks.append([1]  * len(r) + [0]   * pad_len)

        prompts_t   = torch.tensor(padded_prompts,   dtype=torch.long,    device=device)
        responses_t = torch.tensor(padded_responses, dtype=torch.long,    device=device)
        resp_mask_t = torch.tensor(response_masks,   dtype=torch.long,    device=device)
        logprobs_t  = torch.tensor(padded_logprobs,  dtype=torch.float32, device=device)
        input_ids_t = torch.cat([prompts_t, responses_t], dim=1)
        attn_mask_t = (input_ids_t != pad_token_id).long()
        pos_ids_t   = (attn_mask_t.cumsum(dim=1) - 1) * attn_mask_t

        valid_len = resp_mask_t.sum(dim=-1).long()
        n = len(all_prompts)
        token_rewards_t = torch.zeros(n, max_response_length, device=device)
        for i, (vl, rw) in enumerate(zip(valid_len, all_rewards)):
            if rw != 0.0:
                token_rewards_t[i, max(0, vl.item() - 1)] = rw

        turn_indices_t   = torch.ones(n,  dtype=torch.long, device=device)   # single turn = 1
        loss_mask_t      = torch.ones(n,  dtype=torch.long, device=device)   # all participate
        sample_indices_t = torch.arange(n, dtype=torch.long, device=device)

        return DataProto(
            batch=TensorDict({
                "prompts":             prompts_t,
                "responses":           responses_t,
                "response_mask":       resp_mask_t,
                "input_ids":           input_ids_t,
                "attention_mask":      attn_mask_t,
                "position_ids":        pos_ids_t,
                "rollout_log_probs":   logprobs_t,
                "turn_indices":        turn_indices_t,
                "loss_mask":           loss_mask_t,
                "token_level_scores":  token_rewards_t,
                "token_level_rewards": token_rewards_t,
                "sample_indices":      sample_indices_t,
            }, batch_size=n),
            non_tensor_batch={
                "uid":                np.array(all_uids,          dtype=object),
                "original_uid":       np.array(all_orig_uids,     dtype=object),
                "agent_type":         np.full(n, "summary",       dtype=object),
                "skill_scheme_idx":   np.full(n, -1,              dtype=np.int32),
                "summary_group_id":   np.array(all_sum_group_ids, dtype=object),
                "skill_staged":       np.array(all_skill_staged,  dtype=bool),
                "num_turns":          np.ones(n,                  dtype=np.int32),
                "contain_void_turn":  np.zeros(n,                 dtype=np.int32),
                "finish_reasons":     np.full(n, "stop",          dtype=object),
                "multiturn_messages": np.full(n, None,            dtype=object),
                "global_turn_indices":np.ones(n,                  dtype=np.int32),
                "reward_extra_info":  np.array([{}] * n,          dtype=object),
            },
        )

