"""
Prompt construction for all three Skill-aware agents.

All prompt text AND tool schemas live in kernel/config/skill_prompts.yaml.
Placeholders use __varname__ style to avoid conflicts with JSON example braces.
Substitution is done with str.replace(), NOT str.format().

Public API:
  get_selection_tools(top_k)              -> List[dict]  (from yaml, maxItems patched)
  get_summary_tools(max_skills)           -> List[dict]  (from yaml, maxItems patched)
  inject_skill_into_messages(messages, skill_content, skill_config)
  extract_task_description(messages)      -> str
  build_selection_messages(task_messages, skill_library, top_k_select, skill_config,
                           bm25_candidate_paths, ...) -> List[dict]
  build_summary_initial_messages(best_traj, max_skills,
                                 skill_config) -> List[dict]
  build_summary_verify_messages(task_messages, new_skill_content, skill_config)

Note on tool format:
  Selection: select_skills(think, selected_skills: [{name, reason}])
    — name is skill name (snake_case), resolved to rel_path by SkillLibrary.
    — Aligned with selected_skill_data.py make_tools().
  Summary:   update_skill_library(think, new_skills: [{name, description, scope, tags, content}])
    — Single-turn: model calls this exactly once. No ReAct, no read_skill_files.
    — Aligned with run_skill_inference.py TOOLS.
"""

import copy
import os
from copy import deepcopy
from functools import lru_cache
from typing import List, Optional

from kernel.skill.skill_library import SkillLibrary


# ---------------------------------------------------------------------------
# YAML loading
# ---------------------------------------------------------------------------

_YAML_PATH = os.path.normpath(os.path.join(
    os.path.dirname(__file__),       # .../kernel/skill/
    "..", "config", "skill_prompts.yaml"
))


@lru_cache(maxsize=1)
def _load_yaml() -> dict:
    """Load skill_prompts.yaml once and cache the full document."""
    try:
        import yaml
        with open(_YAML_PATH, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return data or {}
    except Exception as e:
        import warnings
        warnings.warn(f"[skill_prompt_builder] Could not load {_YAML_PATH}: {e}")
        return {}


def _get_prompt(key: str, skill_config=None) -> str:
    """
    Return prompt string for `key`.

    Priority:
      1. skill_config.prompts.<key>  (OmegaConf runtime override)
      2. skill_prompts.yaml prompts.<key>
      3. empty string
    """
    if skill_config is not None:
        node = getattr(skill_config, "prompts", None)
        if node is not None:
            val = None
            try:
                val = node.get(key, None)
            except Exception:
                try:
                    val = getattr(node, key, None)
                except Exception:
                    pass
            if val is not None:
                return str(val)

    prompts = _load_yaml().get("prompts", {})
    return str(prompts.get(key, ""))


def _render(template: str, **kwargs) -> str:
    """Replace __varname__ placeholders with str()-converted values."""
    result = template
    for key, value in kwargs.items():
        result = result.replace(f"__{key}__", str(value))
    return result


# ---------------------------------------------------------------------------
# Tool schema loaders  (from YAML, dynamic fields patched at call time)
# ---------------------------------------------------------------------------

def get_selection_tools(top_k: int) -> List[dict]:
    """
    Return the selection agent tool list with maxItems patched to top_k.
    Aligned with selected_skill_data.py make_tools(top_select).
    """
    tools = copy.deepcopy(_load_yaml().get("tools", {}).get("selection", []))
    for tool in tools:
        params = tool.get("function", {}).get("parameters", {})
        props = params.get("properties", {})
        if "selected_skills" in props:
            props["selected_skills"]["maxItems"] = top_k
            props["selected_skills"]["description"] = (
                f"Ordered list of selected skills (at most {top_k})."
            )
        desc = tool.get("function", {}).get("description", "")
        if desc:
            tool["function"]["description"] = (
                f"Select the top {top_k} most relevant skills from the "
                f"candidate list for this task. Call exactly once."
            )
    return tools


def get_summary_tools(max_skills: int) -> List[dict]:
    """
    Return the summary agent tool list with maxItems patched to max_skills.
    Contains only update_skill_library (single-turn, no ReAct read_skill_files).
    Aligned with run_skill_inference.py TOOLS.
    """
    tools = copy.deepcopy(_load_yaml().get("tools", {}).get("summary", []))
    for tool in tools:
        if tool.get("function", {}).get("name") == "update_skill_library":
            params = tool.get("function", {}).get("parameters", {})
            props = params.get("properties", {})
            if "new_skills" in props:
                props["new_skills"]["maxItems"] = max_skills
                props["new_skills"]["description"] = (
                    f"List of new skills to add (at most {max_skills})."
                )
            tool["function"]["description"] = (
                f"Propose new skills to add to the skill library. "
                f"Call this exactly once with at most {max_skills} skills."
            )
    return tools


# ---------------------------------------------------------------------------
# Skill injection (into policy agent messages)
# ---------------------------------------------------------------------------

# Sentinel: the final task instruction at the end of every KernelBench prompt.
# Skills are inserted immediately before this so the model sees them just before
# being asked to act, while the actionable instruction stays last.
_FINAL_INSTRUCTION_PREFIX = "Optimize the architecture named Model"


def _insert_skill_into_text(text: str, skill_block: str) -> str:
    """Insert skill_block before the final 'Optimize the architecture…' sentence.
    Falls back to appending at the end if the sentinel is not found."""
    idx = text.rfind(_FINAL_INSTRUCTION_PREFIX)
    if idx >= 0:
        return text[:idx] + skill_block + "\n\n" + text[idx:]
    return text + skill_block


def inject_skill_into_messages(
    messages: List[dict],
    skill_content: str,
    skill_config=None,
) -> List[dict]:
    """
    Inject skill_content into messages.

    Context layout:
      - If a system turn exists: skill block appended to system (role context preserved).
      - If no system turn (typical KernelBench format — task lives in user):
        skill block inserted into the first user message, immediately BEFORE the
        final "Optimize the architecture…" instruction, so the model reads:
            [user]  <task description + code>
                    <skill block>
                    Optimize the architecture named Model… ModelNew… step by step.
        This keeps the actionable instruction last while making skills visible
        just before the model is asked to generate.
    """
    messages = deepcopy(messages)
    header = _get_prompt("skill_injection_header", skill_config) or "\n\n---\n## [Skill Library] Potentially Relevant Optimization Techniques\n\n"
    footer = _get_prompt("skill_injection_footer", skill_config) or "\n\n---"
    skill_block = header + skill_content.strip() + footer

    for msg in messages:
        if msg.get("role") == "system":
            msg["content"] = msg["content"] + skill_block
            return messages

    # No system turn: inject into first user message
    for msg in messages:
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        part["text"] = _insert_skill_into_text(part["text"], skill_block)
                        return messages
                content.append({"type": "text", "text": skill_block.strip()})
            else:
                msg["content"] = _insert_skill_into_text(content, skill_block)
            return messages

    # No user turn (shouldn't happen): last-resort system
    messages.insert(0, {"role": "system", "content": skill_block.strip()})
    return messages


# ---------------------------------------------------------------------------
# Message content helpers
# ---------------------------------------------------------------------------

def _format_task_messages(messages: List[dict], max_chars: int = 400000) -> str:
    """
    Format policy-agent messages for the summary agent.
    Only system and user turns; truncates to max_chars total.
    """
    parts = []
    total = 0
    for msg in messages:
        role = msg.get("role", "")
        if role not in ("system", "user"):
            continue
        content = msg.get("content", "")
        if isinstance(content, list):
            content = " ".join(
                p.get("text", "") for p in content
                if isinstance(p, dict) and p.get("type") == "text"
            )
        text = f"[{role.upper()}]\n{content}"
        remaining = max_chars - total
        if remaining <= 0:
            break
        if len(text) > remaining:
            text = text[:remaining] + "\n... (truncated)"
            total = max_chars
        else:
            total += len(text)
        parts.append(text)
    return "\n\n".join(parts) if parts else "(no task context available)"


def extract_task_description(messages: List[dict], max_chars: int = 100000) -> str:
    """
    Return the first user message content (truncated) as a short task description.
    Used by selection agent prompt.
    """
    for msg in messages:
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        return str(part.get("text", ""))[:max_chars]
                return ""
            return str(content)[:max_chars]
    return ""


def _format_file_tree_for_selection(
    skill_library: SkillLibrary,
    bm25_candidate_paths: Optional[List[str]],
    max_skills_shown: int,
    seed: Optional[int],
) -> str:
    """
    Build the numbered skill list shown to the selection agent.
    Matches selected_skill_data.py _format_candidates() format:
      N. **name**  [scope=...]  [tags: ...]
         description

    When bm25_candidate_paths is given, only those skills are shown
    (in BM25 score order).  Otherwise the full library is shown with
    optional random sub-sampling.
    """
    if bm25_candidate_paths:
        # Preserve BM25 rank order
        items = []
        for rel_path in bm25_candidate_paths:
            meta = skill_library.get_skill_meta(rel_path)
            if meta is not None:
                items.append(meta)
    else:
        all_items = list(skill_library._meta_cache.values())
        if max_skills_shown and max_skills_shown < len(all_items):
            import random as _random
            rng = _random.Random(seed)
            all_items = rng.sample(all_items, max_skills_shown)
        items = sorted(all_items, key=lambda m: m.rel_path)

    if not items:
        return "(skill library is empty)"

    lines = []
    for i, meta in enumerate(items, 1):
        tags = ", ".join(meta.tags) if meta.tags else ""
        scope = meta.scope or "general"
        lines.append(
            f"{i:2d}. **{meta.name}**  [scope={scope}]"
            + (f"  [tags: {tags}]" if tags else "")
        )
        lines.append(f"    {meta.description}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Selection agent message builder
# ---------------------------------------------------------------------------

def build_selection_messages(
    task_messages: List[dict],
    skill_library: SkillLibrary,
    top_k_select: int,
    skill_config=None,
    bm25_candidate_paths: Optional[List[str]] = None,
    max_skills_shown: Optional[int] = None,
    seed: Optional[int] = None,
) -> List[dict]:
    """
    Build [system, user] for the selection agent.
    Aligned with selected_skill_data.py build_messages().

    bm25_candidate_paths: restrict displayed skills to BM25 top-N rel-paths.
    max_skills_shown:     random sub-sample when bm25_candidate_paths is None.
    seed:                 RNG seed for sub-sample.
    """
    _max_shown = max_skills_shown
    if _max_shown is None and skill_config is not None:
        _max_shown = int(getattr(skill_config, "selection_max_skills_shown", 50) or 50)
    if _max_shown is None:
        _max_shown = 50

    system_content = _render(
        _get_prompt("selection_system", skill_config),
        top_k_select=top_k_select,
    )
    file_tree = _format_file_tree_for_selection(
        skill_library, bm25_candidate_paths, _max_shown, seed
    )
    user_content = _render(
        _get_prompt("selection_user", skill_config),
        top_k_select=top_k_select,
        task_description=extract_task_description(task_messages),
        file_tree=file_tree,
    )
    return [
        {"role": "system", "content": system_content},
        {"role": "user",   "content": user_content},
    ]


# ---------------------------------------------------------------------------
# Summary ReAct agent message builder
# ---------------------------------------------------------------------------

def build_summary_initial_messages(
    best_traj: dict,
    max_skills: int,
    skill_config=None,
) -> List[dict]:
    """
    Build [system, user] for the Summary agent (single-turn, no ReAct).
    Aligned with run_skill_inference.py: one inference, one update_skill_library call.

    The summary agent sees:
    - The task the policy received
    - The exact skill content injected into the policy's prompt
      (None → null scheme, no skills were given)
    - The policy's turn-1 code and best-turn code

    It calls update_skill_library(think, new_skills) exactly once.
    No ReAct, no read_skill_files.
    """
    system_content = _render(
        _get_prompt("summary_system", skill_config),
        max_skills=max_skills,
    )
    task_messages = best_traj.get("task_messages", [])

    injected = best_traj.get("injected_skill_content")
    injected_str = injected.strip() if injected else "(no skills were injected — null scheme was used)"

    user_content = _render(
        _get_prompt("summary_user", skill_config),
        task_messages_formatted=_format_task_messages(task_messages),
        injected_skill_content=injected_str,
        turn1_speedup=f"{best_traj.get('turn1_speedup', 0.0):.2f}",
        turn1_code=best_traj.get("turn1_code", "(not available)"),
        best_turn_speedup=f"{best_traj.get('best_turn_speedup', 0.0):.2f}",
        best_turn_code=best_traj.get("best_turn_code", "(not available)"),
        max_skills=max_skills,
    )
    return [
        {"role": "system", "content": system_content},
        {"role": "user",   "content": user_content},
    ]


def build_summary_verify_messages(
    task_messages: List[dict],
    new_skill_content: str,
    skill_config=None,
) -> List[dict]:
    """Inject new_skill_content into turn-1 prompt for post-summary verification."""
    return inject_skill_into_messages(task_messages, new_skill_content, skill_config)
