#!/usr/bin/env python3
"""
generate_skill_data.py — Offline skill data generation for KernelGYM.

Phase 1 (--phase1):
  - Load N_PHASE1 (default 400) high-speedup trajectories from the coldstart parquet.
  - Run Summary ReAct agent (with Qwen3 tool-call chat template) on each.
  - Add proposed skills directly to the skill library (no verification).

Phase 2 (--phase2):
  - Load N_PHASE2 (default 4000) fresh examples.
  - Run Selection agent on each (k=3 independent calls → 4000 selection trajectories).
  - Run Summary agent on triggered examples (best_speedup >= thresholds).
  - Output: {output_dir}/selection_trajectories.jsonl + summary_trajectories.jsonl

Both phases use the OpenAI messages format (role/content dicts) throughout.
Inference is via vLLM AsyncLLMEngine (local model).

Usage:
    python generate_skill_data.py --phase1 [options]
    python generate_skill_data.py --phase2 [options]
    python generate_skill_data.py --phase1 --phase2 [options]  # run both
"""

import argparse
import asyncio
import json
import os
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from uuid import uuid4

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths / defaults
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent / "davinci-kernel"
sys.path.insert(0, str(_REPO_ROOT))

DEFAULT_MODEL_PATH = ""  # set via --model_path
DEFAULT_PARQUET = "hkust-nlp/drkernel-coldstart-8k"  # or local parquet path
DEFAULT_SKILL_ROOT = "./skill_library"
DEFAULT_OUTPUT_DIR = "./skill_data_output"

# ---------------------------------------------------------------------------
# Lazy imports (vLLM not available at import time on CPU-only machines)
# ---------------------------------------------------------------------------
def _import_vllm():
    from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams
    from vllm.inputs import TokensPrompt
    return AsyncEngineArgs, AsyncLLMEngine, SamplingParams, TokensPrompt


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def parse_turn_speedup(feedback_content: str) -> float:
    """Extract speedup float from a server-feedback user message."""
    m = re.search(r'"speedup"\s*:\s*([0-9.eE+\-]+)', feedback_content)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    return 0.0


def parse_turn_code(assistant_content: str) -> str:
    """Extract the last ```python ... ``` code block from an assistant message."""
    blocks = re.findall(r"```(?:python)?\s*\n?(.*?)```", assistant_content, re.DOTALL)
    if blocks:
        for _block in reversed(blocks):
            if "ModelNew" in _block:
                return _block.strip()
        return blocks[-1].strip()
    return assistant_content.strip()


@dataclass
class TrajectoryExample:
    """One multi-turn trajectory extracted from the coldstart parquet."""
    uid: str
    task_messages: List[dict]          # [system/user] turn-0 prompt only
    original_python_code: str
    entry_point: str
    # Per-turn data (turn index starts at 1)
    turn_speedups: Dict[int, float]    # turn_idx -> speedup
    turn_codes: Dict[int, str]         # turn_idx -> assistant code text
    final_speedup: float
    best_round: int

    @property
    def turn1_speedup(self) -> float:
        return self.turn_speedups.get(1, 0.0)

    @property
    def best_speedup(self) -> float:
        if not self.turn_speedups:
            return 0.0
        return max(self.turn_speedups.values())

    @property
    def best_turn_code(self) -> str:
        if not self.turn_speedups:
            return ""
        best_t = max(self.turn_speedups, key=lambda t: self.turn_speedups[t])
        return self.turn_codes.get(best_t, "")

    @property
    def turn1_code(self) -> str:
        return self.turn_codes.get(1, "")


def load_trajectories(
    parquet_path: str,
    n: int,
    min_speedup: float = 1.1,
    shuffle: bool = True,
    seed: int = 42,
    skip_uids: Optional[set] = None,
) -> List[TrajectoryExample]:
    """
    Load up to `n` trajectories with final_speedup >= min_speedup.
    Messages format: role/content dicts (OpenAI style).
    """
    df = pd.read_parquet(parquet_path)
    df = df[df["final_speedup"] >= min_speedup].copy()
    if shuffle:
        df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    results = []
    for _, row in df.iterrows():
        if len(results) >= n:
            break
        uid = str(row["uuid"])
        if skip_uids and uid in skip_uids:
            continue

        msgs = list(row["messages"])  # already role/content dicts
        # Extract turn-0 prompt: first user message only (system prompt is baked in)
        task_messages = []
        for m in msgs:
            role = m.get("role", "")
            if role in ("system", "user"):
                task_messages.append({"role": role, "content": str(m.get("content", ""))})
                if role == "user":
                    break  # only the initial user message

        # Parse per-turn speedup / code from the full conversation
        turn_speedups: Dict[int, float] = {}
        turn_codes: Dict[int, str] = {}
        turn_idx = 0
        for i, m in enumerate(msgs):
            role = m.get("role", "")
            content = str(m.get("content", ""))
            if role == "assistant":
                turn_idx += 1
                turn_codes[turn_idx] = parse_turn_code(content)
            elif role == "user" and turn_idx > 0:
                sp = parse_turn_speedup(content)
                if sp > 0:
                    turn_speedups[turn_idx] = sp

        if not turn_speedups:
            continue

        results.append(TrajectoryExample(
            uid=uid,
            task_messages=task_messages,
            original_python_code=str(row.get("original_python_code", "")),
            entry_point=str(row.get("entry_point", "Model")),
            turn_speedups=turn_speedups,
            turn_codes=turn_codes,
            final_speedup=float(row["final_speedup"]),
            best_round=int(row.get("best_round", 1)),
        ))

    print(f"[Data] Loaded {len(results)} trajectories (min_speedup={min_speedup})")
    return results


# ---------------------------------------------------------------------------
# Skill library (thin wrapper for offline use)
# ---------------------------------------------------------------------------

class OfflineSkillLibrary:
    """Minimal skill library for offline data generation."""

    def __init__(self, root: str, step: int = 0):
        self.root = Path(root)
        self.step = step
        self._skills: Dict[str, str] = {}  # rel_path -> content
        self._load()

    def _step_dir(self) -> Path:
        d = self.root / f"global_step_{self.step}"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _load(self):
        """Load skills from the current step directory."""
        d = self.root / f"global_step_{self.step}"
        if not d.exists():
            return
        for md in d.rglob("*.md"):
            rel = str(md.relative_to(d))
            self._skills[rel] = md.read_text(encoding="utf-8")

    def list_rel_paths(self) -> List[str]:
        return sorted(self._skills.keys())

    def get_file_tree(self, max_skills: Optional[int] = None, seed: Optional[int] = None) -> str:
        """Return a compact file tree with frontmatter for selection agent.

        max_skills: randomly sample this many skills (5-100 recommended).
        seed: for reproducible sub-sampling.
        """
        if not self._skills:
            return "(empty — no skills yet)"
        import random as _random
        items = sorted(self._skills.items())
        if max_skills is not None and max_skills < len(items):
            rng = _random.Random(seed)
            items = sorted(rng.sample(items, max_skills))
        lines = []
        for rel, content in items:
            # Extract description from frontmatter
            desc = ""
            for line in content.splitlines():
                line = line.strip()
                if line.startswith("description:"):
                    desc = line[len("description:"):].strip().strip('"')
                    break
            lines.append(f"  {rel}  # {desc}" if desc else f"  {rel}")
        return "\n".join(lines)

    def get_content(self, rel_path: str) -> str:
        return self._skills.get(rel_path, f"[not found: {rel_path}]")

    def add_skill(self, name: str, description: str, scope: str,
                  tags: List[str], content: str):
        """Add skill directly (no verification)."""
        safe_name = re.sub(r"[^a-z0-9_]", "_", name.lower())
        if scope.startswith("task_specific/"):
            sub = scope[len("task_specific/"):]
            safe_sub = re.sub(r"[^a-z0-9_]", "_", sub.lower())
            rel = f"task_specific/{safe_sub}/{safe_name}.md"
        else:
            rel = f"general/{safe_name}.md"

        tag_str = ", ".join(tags) if tags else ""
        md_content = (
            f"---\n"
            f"name: {name}\n"
            f"description: {description}\n"
            f"scope: {scope}\n"
            f"tags: [{tag_str}]\n"
            f"---\n\n"
            f"{content.strip()}\n"
        )
        dest = self._step_dir() / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(md_content, encoding="utf-8")
        self._skills[rel] = md_content
        print(f"[Skill] Added: {rel}")
        return rel

    def save_and_advance(self):
        """Bump step so next load sees new skills."""
        self.step += 1
        # Copy all skills to new step dir
        d = self._step_dir()
        for rel, content in self._skills.items():
            dest = d / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(content, encoding="utf-8")
        print(f"[Skill] Library advanced to step {self.step}: {len(self._skills)} skills")


# ---------------------------------------------------------------------------
# Tool call parser (handles Qwen3 XML format)
# ---------------------------------------------------------------------------

_QWEN3_TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*<function=([^>]+)>(.*?)</function>\s*</tool_call>",
    re.DOTALL,
)
_QWEN3_PARAM_RE = re.compile(r"<parameter=([^>]+)>(.*?)</parameter>", re.DOTALL)
_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def parse_tool_call(text: str) -> Optional[dict]:
    """Parse tool call from Qwen3 response. Returns {name, args} or None."""
    # 1. Qwen3 XML format (primary)
    m = _QWEN3_TOOL_CALL_RE.search(text)
    if m:
        func_name = m.group(1).strip()
        args = {}
        for pm in _QWEN3_PARAM_RE.finditer(m.group(2)):
            key = pm.group(1).strip()
            val_str = pm.group(2).strip()
            try:
                args[key] = json.loads(val_str)
            except json.JSONDecodeError:
                args[key] = val_str
        return {"name": func_name, "args": args}

    # 2. Fallback: JSON code block
    for fm in _JSON_BLOCK_RE.finditer(text):
        try:
            obj = json.loads(fm.group(1))
            if isinstance(obj, dict) and "name" in obj:
                if "arguments" in obj and "args" not in obj:
                    obj["args"] = obj.pop("arguments")
                return obj
        except json.JSONDecodeError:
            continue
    return None


# ---------------------------------------------------------------------------
# Prompt builders (inline, OpenAI messages format)
# ---------------------------------------------------------------------------

def build_summary_messages(
    traj: TrajectoryExample,
    skill_library: OfflineSkillLibrary,
    max_skills: int,
) -> List[dict]:
    """Build the initial [system, user] messages for the Summary agent."""
    system_content = (
        f"You are an expert CUDA/Triton kernel optimization engineer.\n"
        f"You have observed a successful multi-turn optimization trajectory.\n"
        f"Your task: extract GENERAL, REUSABLE optimization principles as skills.\n\n"
        f"Workflow:\n"
        f"1. Optionally call read_skill_files to inspect existing skills and avoid duplicates.\n"
        f"2. Call update_skill_library with at most {max_skills} new skill(s).\n"
        f"   Each skill body must contain: ## Motivation, ## Key Idea, ## Example (with code).\n"
        f"3. Your turn ends automatically after update_skill_library is called.\n\n"
        f"Rules:\n"
        f"- Skills must be GENERAL (applicable beyond this specific task).\n"
        f"- Do NOT add task-specific hacks or solutions.\n"
        f"- Respond in English only."
    )

    # Format task messages (system+user from turn 0)
    task_formatted = "\n\n".join(
        f"[{m['role'].upper()}]\n{m['content'][:2000]}"
        for m in traj.task_messages
    )

    turn1_speedup_str = f"{traj.turn1_speedup:.3f}"
    best_speedup_str = f"{traj.best_speedup:.3f}"
    turn1_code = traj.turn1_code or "(not available)"
    best_code = traj.best_turn_code or "(not available)"

    user_content = (
        f"## Task\n\n{task_formatted}\n\n"
        f"## Existing Skills in Library\n\n"
        f"```\n{skill_library.get_file_tree()}\n```\n\n"
        f"## Turn 1 Result (speedup: {turn1_speedup_str}x)\n\n"
        f"```python\n{turn1_code[:2000]}\n```\n\n"
        f"## Best Turn Result (speedup: {best_speedup_str}x)\n\n"
        f"```python\n{best_code[:3000]}\n```\n\n"
        f"The policy agent improved from turn 1 to its best turn.\n"
        f"Your goal: propose at most {max_skills} new skill(s) that capture the key "
        f"optimization insights from this trajectory.\n"
        f"Call read_skill_files first if needed, then call update_skill_library."
    )

    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]


def build_selection_messages(
    traj: TrajectoryExample,
    skill_library: OfflineSkillLibrary,
    k: int,
    max_skills_shown: int = 50,
    seed: Optional[int] = None,
) -> List[dict]:
    """Build [system, user] for the Selection agent.

    Uses SELECTION_TOOLS via apply_chat_template — no JSON format in the prompt.
    max_skills_shown: randomly sample this many skills (5-100 recommended).
    seed: per-scheme seed so each of the k calls sees a different random subset.
    """
    task_desc = ""
    for m in traj.task_messages:
        if m["role"] == "user":
            task_desc = m["content"][:1000]
            break

    system_content = (
        f"You are an expert CUDA kernel optimization engineer.\n"
        f"Given a programming task and a skill library, select the most relevant "
        f"optimization techniques by calling the select_skills function.\n\n"
        f"Your goal: choose the SINGLE BEST combination of skills (1-5 skills) most "
        f"likely to help solve this specific task.\n"
        f"You will be called {k} times per task — each call should make an independent "
        f"selection to explore different strategies.\n"
        f"Call select_skills exactly once. Respond in English only."
    )

    file_tree = skill_library.get_file_tree(max_skills=max_skills_shown, seed=seed)
    user_content = (
        f"## Task Description\n\n{task_desc}\n\n"
        f"## Available Skills\n\n```\n{file_tree}\n```\n\n"
        f"Select the best skill combination for this task."
    )

    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]


def parse_selection_response(text: str, valid_paths: set) -> List[str]:
    """Extract selected skill paths from selection agent response using tool call parser."""
    tool_call = parse_tool_call(text)
    if tool_call and tool_call.get("name") == "select_skills":
        raw = tool_call.get("args", {}).get("skills", [])
        return [p for p in raw if isinstance(p, str) and p in valid_paths]
    return []


# ---------------------------------------------------------------------------
# Async inference engine wrapper
# ---------------------------------------------------------------------------

class AsyncInferenceEngine:
    def __init__(self, model_path: str, gpu_memory_utilization: float = 0.85,
                 tensor_parallel_size: int = 1, max_model_len: int = 16384):
        AsyncEngineArgs, AsyncLLMEngine, _, _ = _import_vllm()
        engine_args = AsyncEngineArgs(
            model=model_path,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            enforce_eager=False,
            disable_log_requests=True,
        )
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)
        self.max_model_len = max_model_len
        print(f"[Engine] Loaded model: {model_path}")

    async def generate(
        self,
        prompt_ids: List[int],
        temperature: float = 0.8,
        max_tokens: int = 2048,
        request_id: Optional[str] = None,
    ) -> Optional[str]:
        _, _, SamplingParams, TokensPrompt = _import_vllm()
        if request_id is None:
            request_id = uuid4().hex
        sp = SamplingParams(
            temperature=temperature,
            max_tokens=min(max_tokens, self.max_model_len - len(prompt_ids)),
            top_p=0.9 if temperature > 0 else 1.0,
        )
        result = None
        async for res in self.engine.generate(
            prompt=TokensPrompt(prompt_token_ids=prompt_ids),
            sampling_params=sp,
            request_id=request_id,
        ):
            result = res
        if result and result.outputs:
            return result.outputs[0].text
        return None


# ---------------------------------------------------------------------------
# Phase 1: Summary generation → skill library bootstrap
# ---------------------------------------------------------------------------

@dataclass
class SummaryTraj:
    uid: str
    messages: List[dict]           # full conversation (system/user/assistant/tool_response)
    proposed_skills: List[dict]    # raw skill dicts from update_skill_library
    turns: int
    added_to_library: bool = False


async def run_summary_react(
    traj: TrajectoryExample,
    skill_library: OfflineSkillLibrary,
    engine: AsyncInferenceEngine,
    tokenizer,
    max_skills: int,
    max_turns: int,
    temperature: float,
    max_tokens: int,
    tools: List[dict],
    s_idx: int = 0,
) -> SummaryTraj:
    """Run one Summary ReAct trajectory."""
    messages = build_summary_messages(traj, skill_library, max_skills)
    all_messages = list(messages)  # track full conversation for output
    proposed_skills: List[dict] = []
    _max_tool_calls = 4

    print(f"  [Summary] uid={traj.uid[:16]}  s={s_idx}  turn1={traj.turn1_speedup:.2f}x → best={traj.best_speedup:.2f}x")

    for turn_idx in range(max_turns):
        prompt_ids = tokenizer.apply_chat_template(
            messages,
            tools=tools,
            add_generation_prompt=True,
            tokenize=True,
        )
        response_text = await engine.generate(
            prompt_ids,
            temperature=temperature,
            max_tokens=max_tokens,
            request_id=f"sum_{traj.uid[:8]}_{s_idx}_{turn_idx}_{uuid4().hex[:6]}",
        )
        if response_text is None:
            print(f"  [Summary] uid={traj.uid[:16]}  generation returned None, stopping")
            break

        messages.append({"role": "assistant", "content": response_text})
        all_messages.append({"role": "assistant", "content": response_text})

        tool_call = parse_tool_call(response_text)
        if tool_call is None:
            print(f"  [Summary] uid={traj.uid[:16]}  turn={turn_idx}  no tool call, done")
            break

        name = tool_call.get("name", "")
        args = tool_call.get("args", {})

        if name == "read_skill_files":
            files = args.get("files", [])[:5]
            result = {f: skill_library.get_content(f) for f in files}
            tool_resp = json.dumps(result, ensure_ascii=False, indent=2)
            tool_msg = {"role": "tool", "content": f"<tool_response>\n{tool_resp}\n</tool_response>"}
            messages.append(tool_msg)
            all_messages.append(tool_msg)
            print(f"  [Summary] uid={traj.uid[:16]}  turn={turn_idx}  read_skill_files: {files}")

        elif name == "update_skill_library":
            raw_skills = args.get("new_skills", [])[:max_skills]
            for d in raw_skills:
                if isinstance(d, dict) and all(k in d for k in ("name", "description", "scope", "content")):
                    proposed_skills.append(d)
            tool_resp = json.dumps({"status": "accepted", "count": len(proposed_skills)})
            tool_msg = {"role": "tool", "content": f"<tool_response>\n{tool_resp}\n</tool_response>"}
            messages.append(tool_msg)
            all_messages.append(tool_msg)
            print(f"  [Summary] uid={traj.uid[:16]}  turn={turn_idx}  update_skill_library: {[d.get('name') for d in proposed_skills]}")
            break  # done after submitting skills
        else:
            tool_msg = {"role": "tool", "content": f'<tool_response>{{"error": "unknown tool: {name}"}}</tool_response>'}
            messages.append(tool_msg)
            all_messages.append(tool_msg)

        if turn_idx >= _max_tool_calls - 1:
            break

    return SummaryTraj(
        uid=traj.uid,
        messages=all_messages,
        proposed_skills=proposed_skills,
        turns=len([m for m in all_messages if m["role"] == "assistant"]),
    )


async def run_phase1(
    args_ns,
    engine: AsyncInferenceEngine,
    tokenizer,
    skill_library: OfflineSkillLibrary,
    tools: List[dict],
) -> List[SummaryTraj]:
    """
    Phase 1: generate N_PHASE1 summary trajectories and add skills to library.
    Returns list of SummaryTraj for saving.
    """
    print(f"\n{'='*60}")
    print(f"PHASE 1: Generating {args_ns.n_phase1} summary trajectories")
    print(f"{'='*60}\n")

    trajs = load_trajectories(
        args_ns.parquet,
        n=args_ns.n_phase1,
        min_speedup=args_ns.min_speedup_phase1,
        seed=args_ns.seed,
    )

    summary_trajs: List[SummaryTraj] = []
    t0 = time.time()

    # Process sequentially (or in small async batches)
    batch_size = args_ns.summary_concurrency
    for i in range(0, len(trajs), batch_size):
        batch = trajs[i:i + batch_size]
        tasks = []
        for j, traj in enumerate(batch):
            for s in range(args_ns.summary_parallel_s):
                tasks.append(run_summary_react(
                    traj, skill_library, engine, tokenizer,
                    max_skills=args_ns.max_new_skills,
                    max_turns=args_ns.summary_max_turns,
                    temperature=args_ns.summary_temperature,
                    max_tokens=args_ns.summary_max_tokens,
                    tools=tools,
                    s_idx=s,
                ))
        results = await asyncio.gather(*tasks)
        for straj in results:
            summary_trajs.append(straj)
            # Add proposed skills directly (no verification)
            for skill in straj.proposed_skills:
                try:
                    skill_library.add_skill(
                        name=str(skill.get("name", "unnamed")),
                        description=str(skill.get("description", "")),
                        scope=str(skill.get("scope", "general")),
                        tags=[str(t) for t in skill.get("tags", [])] if isinstance(skill.get("tags"), list) else [],
                        content=str(skill.get("content", "")),
                    )
                    straj.added_to_library = True
                except Exception as e:
                    print(f"  [Skill] Failed to add skill: {e}")

        elapsed = time.time() - t0
        print(f"[Phase1] {min(i+batch_size, len(trajs))}/{len(trajs)} done  "
              f"elapsed={elapsed:.0f}s  library_size={len(skill_library.list_rel_paths())}")

    # Persist skills to new step
    skill_library.save_and_advance()
    print(f"\n[Phase1] Done. {len(skill_library.list_rel_paths())} skills in library.")
    return summary_trajs


# ---------------------------------------------------------------------------
# Phase 2: Selection + Summary rollouts for training data
# ---------------------------------------------------------------------------

@dataclass
class SelectionTraj:
    uid: str                        # "{orig_uid}_sel_scheme{j}"
    original_uid: str
    scheme_idx: int
    messages: List[dict]            # [system, user, assistant]
    selected_paths: List[str]
    reward: float = 0.0             # filled later (placeholder for offline generation)


async def run_selection(
    traj: TrajectoryExample,
    skill_library: OfflineSkillLibrary,
    engine: AsyncInferenceEngine,
    tokenizer,
    k: int,
    temperature: float,
    max_tokens: int,
    selection_tools: list,
    max_skills_shown: int = 50,
) -> List[SelectionTraj]:
    """Run k independent selection inferences for one task.

    Each of the k calls uses a different random sub-sample of the skill library
    (max_skills_shown, seeded by scheme_idx) to promote diverse selection.
    """
    valid_paths = set(skill_library.list_rel_paths())
    tasks = []
    for j in range(k):
        # Each scheme gets a different random subset (seed=j) for diversity
        messages = build_selection_messages(
            traj, skill_library, k,
            max_skills_shown=max_skills_shown,
            seed=j,
        )
        prompt_ids = tokenizer.apply_chat_template(
            messages,
            tools=selection_tools,
            add_generation_prompt=True,
            tokenize=True,
        )
        tasks.append((j, messages, prompt_ids))

    gen_tasks = [
        engine.generate(
            p_ids, temperature=temperature, max_tokens=max_tokens,
            request_id=f"sel_{traj.uid[:8]}_j{j}_{uuid4().hex[:6]}",
        )
        for j, _, p_ids in tasks
    ]
    responses = await asyncio.gather(*gen_tasks)

    results = []
    for (j, messages, _), response_text in zip(tasks, responses):
        if response_text is None:
            selected = []
        else:
            selected = parse_selection_response(response_text, valid_paths)
        full_msgs = messages + [{"role": "assistant", "content": response_text or ""}]
        results.append(SelectionTraj(
            uid=f"{traj.uid}_sel_scheme{j}",
            original_uid=traj.uid,
            scheme_idx=j,
            messages=full_msgs,
            selected_paths=selected,
        ))
    return results


async def run_phase2(
    args_ns,
    engine: AsyncInferenceEngine,
    tokenizer,
    skill_library: OfflineSkillLibrary,
    summary_tools: List[dict],
    selection_tools: List[dict],
    phase1_uids: Optional[set] = None,
) -> Tuple[List[SelectionTraj], List[SummaryTraj]]:
    """
    Phase 2: run selection (k per task) and summary on triggered tasks.
    Returns (selection_trajs, summary_trajs).
    """
    print(f"\n{'='*60}")
    print(f"PHASE 2: Generating selection + summary data ({args_ns.n_phase2} tasks)")
    print(f"{'='*60}\n")

    trajs = load_trajectories(
        args_ns.parquet,
        n=args_ns.n_phase2,
        min_speedup=1.0,  # use all examples for phase2
        seed=args_ns.seed + 1000,
        skip_uids=phase1_uids,
    )

    all_selection: List[SelectionTraj] = []
    all_summary: List[SummaryTraj] = []
    batch_size = args_ns.selection_concurrency
    t0 = time.time()

    for i in range(0, len(trajs), batch_size):
        batch = trajs[i:i + batch_size]

        # --- Selection ---
        sel_tasks = [
            run_selection(
                traj, skill_library, engine, tokenizer,
                k=args_ns.k,
                temperature=args_ns.selection_temperature,
                max_tokens=args_ns.selection_max_tokens,
                selection_tools=selection_tools,
                max_skills_shown=args_ns.max_skills_shown,
            )
            for traj in batch
        ]
        sel_results_batch = await asyncio.gather(*sel_tasks)
        for sel_list in sel_results_batch:
            all_selection.extend(sel_list)

        # --- Summary for triggered tasks ---
        triggered = [
            traj for traj in batch
            if (traj.best_speedup >= traj.turn1_speedup * args_ns.speedup_improve_thresh
                and traj.best_speedup >= args_ns.speedup_vs_baseline_thresh
                and traj.turn1_speedup > 0)
        ]
        if triggered:
            sum_tasks = []
            for traj in triggered:
                for s in range(args_ns.summary_parallel_s):
                    sum_tasks.append(run_summary_react(
                        traj, skill_library, engine, tokenizer,
                        max_skills=args_ns.max_new_skills,
                        max_turns=args_ns.summary_max_turns,
                        temperature=args_ns.summary_temperature,
                        max_tokens=args_ns.summary_max_tokens,
                        tools=summary_tools,
                        s_idx=s,
                    ))
            sum_results = await asyncio.gather(*sum_tasks)
            all_summary.extend(sum_results)

        elapsed = time.time() - t0
        print(
            f"[Phase2] {min(i+batch_size, len(trajs))}/{len(trajs)} done  "
            f"elapsed={elapsed:.0f}s  selection={len(all_selection)}  summary={len(all_summary)}"
        )

    print(f"\n[Phase2] Done. selection={len(all_selection)}  summary={len(all_summary)}")
    return all_selection, all_summary


# ---------------------------------------------------------------------------
# Output serialisation
# ---------------------------------------------------------------------------

def save_selection_jsonl(trajs: List[SelectionTraj], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for t in trajs:
            f.write(json.dumps({
                "uid": t.uid,
                "original_uid": t.original_uid,
                "scheme_idx": t.scheme_idx,
                "messages": t.messages,
                "selected_paths": t.selected_paths,
                "reward": t.reward,
            }, ensure_ascii=False) + "\n")
    print(f"[Output] Saved {len(trajs)} selection trajectories → {path}")


def save_summary_jsonl(trajs: List[SummaryTraj], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for t in trajs:
            f.write(json.dumps({
                "uid": t.uid,
                "messages": t.messages,
                "proposed_skills": t.proposed_skills,
                "turns": t.turns,
                "added_to_library": t.added_to_library,
            }, ensure_ascii=False) + "\n")
    print(f"[Output] Saved {len(trajs)} summary trajectories → {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Generate skill training data for KernelGYM")
    p.add_argument("--phase1", action="store_true", help="Run Phase 1 (summary bootstrap)")
    p.add_argument("--phase2", action="store_true", help="Run Phase 2 (selection + summary)")
    p.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    p.add_argument("--parquet", default=DEFAULT_PARQUET)
    p.add_argument("--skill-root", default=DEFAULT_SKILL_ROOT)
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--skill-step", type=int, default=0,
                   help="Initial skill library global step to load from")
    # Phase 1
    p.add_argument("--n-phase1", type=int, default=400)
    p.add_argument("--min-speedup-phase1", type=float, default=1.2,
                   help="Min speedup for Phase 1 examples")
    # Phase 2
    p.add_argument("--n-phase2", type=int, default=4000)
    p.add_argument("--k", type=int, default=3, help="Selection schemes per task")
    # Summary
    p.add_argument("--summary-parallel-s", type=int, default=2,
                   help="Parallel summary trajectories per triggered task")
    p.add_argument("--summary-max-turns", type=int, default=3)
    p.add_argument("--summary-temperature", type=float, default=0.8)
    p.add_argument("--summary-max-tokens", type=int, default=2048)
    p.add_argument("--max-new-skills", type=int, default=2)
    # Selection
    p.add_argument("--selection-temperature", type=float, default=0.8)
    p.add_argument("--selection-max-tokens", type=int, default=1024)
    p.add_argument("--max-skills-shown", type=int, default=50,
                   help="Random sub-sample of skills shown per selection call (5-100)."
                        " Each of the k calls uses a different random subset.")
    # Trigger thresholds
    p.add_argument("--speedup-improve-thresh", type=float, default=1.1)
    p.add_argument("--speedup-vs-baseline-thresh", type=float, default=1.1)
    # Inference
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    p.add_argument("--tensor-parallel-size", type=int, default=1)
    p.add_argument("--max-model-len", type=int, default=16384)
    p.add_argument("--summary-concurrency", type=int, default=8,
                   help="Summary tasks processed concurrently")
    p.add_argument("--selection-concurrency", type=int, default=32,
                   help="Selection tasks processed concurrently")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


async def async_main(args_ns):
    if not args_ns.phase1 and not args_ns.phase2:
        print("Specify --phase1 and/or --phase2")
        sys.exit(1)

    # Load tools from skill_prompt_builder
    try:
        from kernel.skill.skill_prompt_builder import SUMMARY_TOOLS, SELECTION_TOOLS
    except ImportError:
        # Inline fallback if running outside davinci-kernel package
        from generate_skill_data import _SUMMARY_TOOLS_FALLBACK as SUMMARY_TOOLS  # type: ignore
        SELECTION_TOOLS = []  # type: ignore

    # Tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args_ns.model_path, trust_remote_code=True)
    print(f"[Tokenizer] Loaded from {args_ns.model_path}")

    # Skill library
    skill_library = OfflineSkillLibrary(args_ns.skill_root, step=args_ns.skill_step)
    print(f"[Library] Root={args_ns.skill_root}  step={args_ns.skill_step}  "
          f"skills={len(skill_library.list_rel_paths())}")

    # Inference engine
    engine = AsyncInferenceEngine(
        model_path=args_ns.model_path,
        gpu_memory_utilization=args_ns.gpu_memory_utilization,
        tensor_parallel_size=args_ns.tensor_parallel_size,
        max_model_len=args_ns.max_model_len,
    )

    phase1_uids: set = set()
    summary_trajs_phase1: List[SummaryTraj] = []
    selection_trajs: List[SelectionTraj] = []
    summary_trajs_phase2: List[SummaryTraj] = []

    # ── Phase 1 ──────────────────────────────────────────────────────────────
    if args_ns.phase1:
        summary_trajs_phase1 = await run_phase1(
            args_ns, engine, tokenizer, skill_library, SUMMARY_TOOLS
        )
        phase1_uids = {t.uid for t in summary_trajs_phase1}
        save_summary_jsonl(
            summary_trajs_phase1,
            os.path.join(args_ns.output_dir, "phase1_summary_trajectories.jsonl"),
        )

    # ── Phase 2 ──────────────────────────────────────────────────────────────
    if args_ns.phase2:
        selection_trajs, summary_trajs_phase2 = await run_phase2(
            args_ns, engine, tokenizer, skill_library,
            summary_tools=SUMMARY_TOOLS,
            selection_tools=SELECTION_TOOLS,
            phase1_uids=phase1_uids if args_ns.phase1 else None,
        )
        save_selection_jsonl(
            selection_trajs,
            os.path.join(args_ns.output_dir, "selection_trajectories.jsonl"),
        )
        save_summary_jsonl(
            summary_trajs_phase2,
            os.path.join(args_ns.output_dir, "phase2_summary_trajectories.jsonl"),
        )

    print(f"\n{'='*60}")
    print(f"DONE")
    print(f"  Phase1 summary:    {len(summary_trajs_phase1)}")
    print(f"  Phase2 selection:  {len(selection_trajs)}")
    print(f"  Phase2 summary:    {len(summary_trajs_phase2)}")
    print(f"  Skill library:     {len(skill_library.list_rel_paths())} skills")
    print(f"  Output dir:        {args_ns.output_dir}")
    print(f"{'='*60}\n")


def main():
    args_ns = parse_args()
    asyncio.run(async_main(args_ns))


if __name__ == "__main__":
    main()
