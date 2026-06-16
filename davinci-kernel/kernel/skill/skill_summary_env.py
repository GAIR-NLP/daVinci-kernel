"""
SkillSummaryParser: single-turn response parser for the Summary agent.

The summary agent makes ONE inference call, calling update_skill_library(think, new_skills).
There is NO multi-turn ReAct, NO read_skill_files tool.
This is aligned with run_skill_inference.py.

Public API:
  parse_summary_response(text, max_new_skills) -> List[NewSkill]
      Parse one assistant response and return validated, filtered skills.

  _parse_tool_call(text) -> Optional[dict]
      Low-level: extract the tool call dict from raw model output.
      Used by selection parser in vllm_async_engine_skill.py too.
"""

import ast
import json
import re
from typing import Dict, List, Optional

from kernel.skill.skill_library import NewSkill


# ---------------------------------------------------------------------------
# Lazy skill filter  (identical to run_skill_inference.py _LAZY_RE)
# ---------------------------------------------------------------------------

_LAZY_PATTERNS = [
    r"\bfall\s*back\s+to\s+(pytorch|torch)\b",
    r"\buse\s+(torch\.compile|cuBLAS|cuDNN|cublas|cudnn)\b",
    r"\bavoid\s+(writing|custom)\s+kernel",
    r"\bskip\s+(the\s+)?(cuda|triton)\s+kernel",
    r"\bdon.t\s+(call|write|use)\s+(cuda|triton)",
    r"\bnot\s+write\s+a\s+kernel\b",
    r"\bdrop\s+the\s+(kernel|cuda|triton)\b",
    r"\bskip\s+(computation|ops?|operator)\b",
    r"\beliminate\s+the\s+(kernel|op)\s+entirely\b",
]
_LAZY_RE = re.compile("|".join(_LAZY_PATTERNS), re.IGNORECASE)


def _is_lazy_skill(skill_dict: dict) -> bool:
    """Return True if the skill encourages bypassing CUDA/Triton entirely."""
    text = " ".join([
        skill_dict.get("name", ""),
        skill_dict.get("description", ""),
        skill_dict.get("content", ""),
    ])
    return bool(_LAZY_RE.search(text))


# ---------------------------------------------------------------------------
# Tool call parser  (handles Qwen3 XML format and JSON fallbacks)
# ---------------------------------------------------------------------------

_TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*(\{.*?\})\s*</tool_call>",
    re.DOTALL,
)
_QWEN3_TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*<function=([^>]+)>(.*?)</function>\s*</tool_call>",
    re.DOTALL,
)
_QWEN3_PARAM_RE = re.compile(
    r"<parameter=([^>]+)>(.*?)</parameter>",
    re.DOTALL,
)
_JSON_BLOCK_RE = re.compile(
    r"```(?:json)?\s*(\{.*?\})\s*```",
    re.DOTALL,
)


def _parse_tool_call(text: str) -> Optional[Dict]:
    """
    Parse a single tool call from assistant response text.

    Priority:
      1. Qwen3 XML: <tool_call><function=name><parameter=p>v</parameter></function></tool_call>
      2. JSON tags:  <tool_call>{"name": ..., "args": {...}}</tool_call>
      3. Fallback:   ```json {"name": ..., ...} ```
    """
    # 1. Qwen3 XML format
    m = _QWEN3_TOOL_CALL_RE.search(text)
    if m:
        func_name = m.group(1).strip()
        args: Dict = {}
        for pm in _QWEN3_PARAM_RE.finditer(m.group(2)):
            key = pm.group(1).strip()
            val_str = pm.group(2).strip()
            try:
                args[key] = json.loads(val_str)
            except json.JSONDecodeError:
                # Qwen3 sometimes emits Python literals (single quotes, etc.)
                try:
                    args[key] = ast.literal_eval(val_str)
                except (ValueError, SyntaxError):
                    args[key] = val_str
        return {"name": func_name, "args": args}

    # 2. JSON in <tool_call> tags
    m = _TOOL_CALL_RE.search(text)
    if m:
        try:
            obj = json.loads(m.group(1))
            if "arguments" in obj and "args" not in obj:
                obj["args"] = obj.pop("arguments")
            return obj
        except json.JSONDecodeError:
            pass

    # 3. ```json {...} ``` fallback
    for m in _JSON_BLOCK_RE.finditer(text):
        try:
            obj = json.loads(m.group(1))
            if isinstance(obj, dict) and "name" in obj:
                if "arguments" in obj and "args" not in obj:
                    obj["args"] = obj.pop("arguments")
                return obj
        except json.JSONDecodeError:
            continue

    return None


def _validate_skill_dict(d: dict) -> bool:
    """Return True if the skill dict has all required non-empty string fields."""
    required = {"name", "description", "scope", "content"}
    return required.issubset(d.keys()) and all(
        isinstance(d[k], str) and d[k].strip() for k in required
    )


# ---------------------------------------------------------------------------
# Public API: single-turn response parser
# ---------------------------------------------------------------------------

def parse_summary_response(
    text: str,
    max_new_skills: int,
) -> List[NewSkill]:
    """
    Parse a single summary agent response and return validated NewSkill objects.

    Aligned with run_skill_inference.py process_record():
      - Parses update_skill_library(think, new_skills) tool call
      - Caps at max_new_skills
      - Validates required fields
      - Filters lazy skills (_LAZY_RE)

    Returns an empty list if no valid tool call or all skills are rejected.
    Logs reasons for rejected skills via print (same format as run_skill_inference.py).
    """
    tool_call = _parse_tool_call(text)
    if tool_call is None or tool_call.get("name") != "update_skill_library":
        print(f"[Skill] summary_parse  no update_skill_library call found  "
              f"text_preview={text[:80]!r}")
        return []

    args = tool_call.get("args", {})
    think = args.get("think", "")
    raw_skills = args.get("new_skills", [])
    if not isinstance(raw_skills, list):
        print(f"[Skill] summary_parse  new_skills is not a list: {type(raw_skills)}")
        return []

    raw_skills = raw_skills[:max_new_skills]
    accepted: List[NewSkill] = []

    for d in raw_skills:
        if not isinstance(d, dict):
            continue
        missing = [f for f in ("name", "description", "scope", "content") if not d.get(f)]
        if missing:
            print(f"[Skill] summary_parse  rejected  name={d.get('name','?')!r}  "
                  f"reason=missing_fields:{missing}")
            continue
        if _is_lazy_skill(d):
            print(f"[Skill] summary_parse  rejected  name={d.get('name','?')!r}  "
                  f"reason=lazy_optimization")
            continue
        skill = NewSkill(
            name=str(d["name"]).strip(),
            description=str(d["description"]).strip(),
            scope=str(d.get("scope", "general")).strip(),
            tags=[str(t) for t in d.get("tags", [])] if isinstance(d.get("tags"), list) else [],
            content=str(d["content"]).strip(),
        )
        accepted.append(skill)
        print(f"[Skill] summary_parse  accepted  name={skill.name!r}  "
              f"scope={skill.scope}  tags={skill.tags}  "
              f"desc={skill.description[:60]!r}")

    if think:
        print(f"[Skill] summary_parse  think={think[:150]!r}")

    return accepted
