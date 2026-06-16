"""
skill_generator.py — Offline skill generation from skill_generation_seed.jsonl

Reads multi-turn optimization trajectories and calls an LLM (via OpenAI-compatible API)
to extract generalizable CUDA/Triton optimization skills.

Key design decisions (vs runtime skill_summary_env.py):
  - Single-turn: one prompt, one tool call response (update_skill_library), done.
  - At most 2 skills per trajectory.
  - Full conversation history included in the user message (all turns, merged).
  - Tool call output (structured JSON) instead of plain text.
  - Rejects lazy/degenerate "skills" (e.g. "don't call CUDA kernels", fall back to PyTorch).
  - System prompt also covers common pitfalls to avoid, not just how to improve.

Usage:
    python skill_generator.py --input ../data/skill/skill_generation_seed.jsonl \
                              --output ../data/skill/generated_skills.jsonl \
                              --model gpt-5.4 \
                              --base-url https://apicz.boyuerichdata.com/v1 \
                              --api-key sk-RrrMwBxWPjBW0uV0D9ISehV6BA7wH8Hftzp5CAYQysYNRagk \
                              --subset 200 \
                              --workers 10 \
                              --resume
"""

import argparse
import json
import os
import re
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Optional

from openai import OpenAI

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert CUDA/Triton kernel optimization engineer.
You have observed a successful multi-turn kernel optimization trajectory where a model iteratively improved a custom GPU kernel.

Your task: extract at most 2 GENERAL, REUSABLE optimization skills from this trajectory.

## What makes a GOOD skill

- Broadly applicable: the technique works across many operators/models, not just this one task.
- Actionable: explains *how* to apply the technique, not just *that* it helps.
- Non-obvious: not trivially covered by basic CUDA/Triton documentation.
- Each skill body must contain these three sections:
    ## Motivation   — why this technique matters and when to use it
    ## Key Idea     — the core mechanism and how to implement it
    ## Example      — a short self-contained code snippet illustrating the idea

## What to AVOID (your output will be rejected if any skill does this)

- Skills that bypass CUDA/Triton entirely (e.g. "use torch.compile", "call cuBLAS directly",
  "avoid writing kernels", "fall back to PyTorch operators"). All skills MUST be about writing
  or structuring actual kernels. And such kernels are correctly implemented.
- Task-specific hacks (e.g. "for this exact shape, hard-code block size 256").
- Observations masquerading as skills (e.g. "the model improved by fusing ops" with no guidance).
- Trivial or obvious advice (e.g. "use shared memory", "reduce memory bandwidth").
- Skills that encourage the optimizer to skip computation or reduce precision without
  explicit justification (e.g. "drop the softmax", "use fp8 blindly").
- Duplicate skills: if two techniques are essentially the same, merge them.

## Common pitfalls to document (in addition to improvements)

When extracting skills, also consider:
- Numerical correctness traps: reduction order, fp16/bf16 overflow, non-associativity.
- Race conditions: missing __syncthreads(), warp divergence from early exits.
- Indexing bugs that appear only at non-power-of-two sizes.
- Performance cliffs: bank conflicts, uncoalesced access patterns, occupancy limits.
- Triton-specific pitfalls: incorrect tl.constexpr usage, wrong mask shapes, autotune overhead.

When calling update_skill_library, first fill in the `think` field: briefly identify the key
problem or bottleneck this trajectory reveals, and what kind of skill would address it. Then
fill in `new_skills` with at most 2 skills derived from that analysis.

Call update_skill_library exactly once. Respond in English only."""

# ---------------------------------------------------------------------------
# Tool definition (OpenAI function-calling format)
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "update_skill_library",
            "description": (
                "Propose new skills to add to the skill library. "
                "Call this exactly once with at most 2 skills."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "think": {
                        "type": "string",
                        "description": (
                            "Your analysis before writing skills: "
                            "what is the key problem or bottleneck revealed by this trajectory? "
                            "What technique turned things around? "
                            "What pitfall caused early failures? "
                            "Use this to justify the skills you are about to propose."
                        ),
                    },
                    "new_skills": {
                        "type": "array",
                        "description": "List of new skills to add (at most 2).",
                        "maxItems": 2,
                        "items": {
                            "type": "object",
                            "required": ["name", "description", "scope", "tags", "content"],
                            "properties": {
                                "name": {
                                    "type": "string",
                                    "description": "Short snake_case identifier, e.g. 'vectorized_load_store'",
                                },
                                "description": {
                                    "type": "string",
                                    "description": "One-line description for the selection agent (≤120 chars).",
                                },
                                "scope": {
                                    "type": "string",
                                    "description": (
                                        "'general' for broadly applicable skills, "
                                        "or 'task_specific/<type>' for narrow ones "
                                        "(e.g. 'task_specific/matmul')."
                                    ),
                                },
                                "tags": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "3-6 keyword tags (snake_case).",
                                },
                                "content": {
                                    "type": "string",
                                    "description": (
                                        "Full skill body in Markdown. "
                                        "Must contain ## Motivation, ## Key Idea, ## Example sections. "
                                        "Keep to ≤512 tokens."
                                    ),
                                },
                            },
                        },
                    }
                },
                "required": ["think", "new_skills"],
            },
        },
    }
]

# ---------------------------------------------------------------------------
# Rejection filters
# ---------------------------------------------------------------------------

_LAZY_PATTERNS = [
    # Discourage bypassing CUDA/Triton
    r"\bfall\s*back\s+to\s+(pytorch|torch)\b",
    r"\buse\s+(torch\.compile|cuBLAS|cuDNN|cublas|cudnn)\b",
    r"\bavoid\s+(writing|custom)\s+kernel",
    r"\bskip\s+(the\s+)?(cuda|triton)\s+kernel",
    r"\bdon.t\s+(call|write|use)\s+(cuda|triton)",
    r"\bnot\s+write\s+a\s+kernel\b",
    r"\bdrop\s+the\s+(kernel|cuda|triton)\b",
    # Encourage removing computation without justification
    r"\bskip\s+(computation|ops?|operator)\b",
    r"\beliminate\s+the\s+(kernel|op)\s+entirely\b",
]
_LAZY_RE = re.compile("|".join(_LAZY_PATTERNS), re.IGNORECASE)


def _is_lazy_skill(skill: dict) -> bool:
    """Return True if this skill appears to encourage lazy/degenerate optimization."""
    text = " ".join([
        skill.get("name", ""),
        skill.get("description", ""),
        skill.get("content", ""),
    ])
    return bool(_LAZY_RE.search(text))


# ---------------------------------------------------------------------------
# Message building
# ---------------------------------------------------------------------------

def _extract_speedup(feedback_content: str) -> Optional[float]:
    """Parse speedup value from server feedback JSON embedded in a user message."""
    m = re.search(r'"speedup"\s*:\s*([0-9.eE+\-]+)', feedback_content)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    return None


def _format_trajectory(record: dict) -> str:
    """
    Format the full conversation as a single user-readable block.

    Structure per round:
      ### Round N (speedup: X.XXx)
      **Assistant:**
      <code>
      **Server Feedback:**
      <json>
    """
    messages = record["messages"]
    round_speedups = record.get("round_speedups", [])
    best_round = record.get("best_round", None)  # 1-indexed

    # messages[0] is the initial user task prompt — extract it
    task_prompt = messages[0]["content"] if messages else ""

    parts = [
        "## Original Task\n",
        task_prompt,
        "\n",
    ]

    # Pair up (assistant, feedback) turns
    # messages layout: [user_task, asst1, feedback1, asst2, feedback2, ...]
    # assistant turns: messages[1], [3], [5], ...
    # feedback turns:  messages[2], [4], [6], ...
    num_rounds = (len(messages) - 1) // 2

    for i in range(num_rounds):
        asst_idx = 1 + i * 2
        fb_idx = 2 + i * 2

        asst_content = messages[asst_idx]["content"] if asst_idx < len(messages) else ""
        fb_content = messages[fb_idx]["content"] if fb_idx < len(messages) else ""

        speedup = round_speedups[i] if i < len(round_speedups) else None
        speedup_str = f"{speedup:.4f}x" if speedup is not None else "?"
        best_marker = " ← BEST" if (best_round is not None and (i + 1) == best_round) else ""

        parts.append(f"### Round {i + 1} (speedup: {speedup_str}{best_marker})\n\n")
        parts.append("**Assistant response:**\n\n")
        parts.append(asst_content)
        parts.append("\n\n")
        if fb_content:
            parts.append("**Server feedback:**\n\n")
            parts.append(fb_content)
            parts.append("\n\n")

    return "".join(parts)


def build_messages(record: dict) -> list[dict]:
    """Build the [system, user] messages for a single trajectory."""
    trajectory_text = _format_trajectory(record)

    best_round = record.get("best_round", None)
    round_speedups = record.get("round_speedups", [])
    turn1_speedup = round_speedups[0] if round_speedups else None
    best_speedup = round_speedups[best_round - 1] if (best_round and best_round <= len(round_speedups)) else None

    header = []
    if turn1_speedup is not None:
        header.append(f"Turn 1 speedup: {turn1_speedup:.4f}x")
    if best_speedup is not None:
        header.append(f"Best turn speedup: {best_speedup:.4f}x (round {best_round})")
    header_str = "  |  ".join(header) + "\n\n" if header else ""

    user_content = (
        f"{header_str}"
        f"The trajectory below shows a multi-turn kernel optimization session. "
        f"The model improved its kernel across rounds — study the full conversation to understand "
        f"what techniques were tried, what worked, and what pitfalls were encountered.\n\n"
        f"{trajectory_text}\n"
        f"Now call update_skill_library with at most 2 new, general, reusable skills "
        f"extracted from this trajectory. Focus on techniques that would help achieve "
        f"similar improvements on OTHER tasks, not just this one."
    )

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

def call_model(
    client: OpenAI,
    model: str,
    messages: list[dict],
    max_retries: int = 3,
    retry_delay: float = 5.0,
) -> Optional[dict]:
    """
    Call the model with tool_choice=required.

    Returns a dict with keys:
      - "tool_args"    : parsed dict from update_skill_library arguments
      - "thinking"     : reasoning/thinking text if present (str or None)
      - "content"      : assistant message text content (str or None)
      - "tool_call_raw": raw tool call as a serialisable dict
    Returns None on unrecoverable failure.
    """
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=TOOLS,
                tool_choice={"type": "function", "function": {"name": "update_skill_library"}},
                temperature=0.7,
                max_tokens=4096,
            )
            choice = response.choices[0]
            msg = choice.message

            # --- Extract thinking (reasoning_content field used by some providers) ---
            thinking = getattr(msg, "reasoning_content", None)
            # Also check inside content list (e.g. Anthropic-style [{type: thinking, ...}])
            if thinking is None and isinstance(msg.content, list):
                for block in msg.content:
                    if isinstance(block, dict) and block.get("type") == "thinking":
                        thinking = block.get("thinking") or block.get("text")
                        break

            # --- Extract text content ---
            if isinstance(msg.content, str):
                text_content = msg.content or None
            elif isinstance(msg.content, list):
                # Concatenate all text blocks
                parts = [
                    b.get("text", "") for b in msg.content
                    if isinstance(b, dict) and b.get("type") == "text"
                ]
                text_content = "".join(parts) or None
            else:
                text_content = None

            # --- Extract tool call ---
            tool_calls = getattr(msg, "tool_calls", None)
            if not tool_calls:
                print(f"  [warn] No tool call in response (attempt {attempt+1})")
                continue

            tc = tool_calls[0]
            if tc.function.name != "update_skill_library":
                print(f"  [warn] Unexpected tool: {tc.function.name} (attempt {attempt+1})")
                continue

            raw_args_str = tc.function.arguments
            tool_args = json.loads(raw_args_str)

            return {
                "tool_args": tool_args,           # parsed dict (includes "think" + "new_skills")
                "thinking": thinking,              # model reasoning/thinking token if any
                "content": text_content,           # assistant text before tool call if any
                "tool_call_raw": {
                    "id": tc.id,
                    "function": {
                        "name": tc.function.name,
                        "arguments": raw_args_str, # raw unparsed JSON string
                    },
                },
            }

        except json.JSONDecodeError as e:
            print(f"  [warn] JSON decode error in tool args: {e} (attempt {attempt+1})")
        except Exception as e:
            print(f"  [warn] API error: {e} (attempt {attempt+1})")
            if attempt < max_retries - 1:
                time.sleep(retry_delay * (attempt + 1))

    return None


# ---------------------------------------------------------------------------
# Per-record processing
# ---------------------------------------------------------------------------

def process_record(
    record: dict,
    client: OpenAI,
    model: str,
) -> Optional[dict]:
    """Process one trajectory record and return a result dict."""
    uuid = record.get("uuid", "?")
    messages = build_messages(record)

    raw = call_model(client, model, messages)
    if raw is None:
        return {"uuid": uuid, "status": "api_error", "skills": [], "rejected": [],
                "think": None, "thinking": None, "content": None, "tool_call_raw": None,
                "input_messages": messages,
                "source_record": _source_meta(record)}

    tool_args = raw["tool_args"]
    # "think" is the model's self-analysis written before new_skills
    think = tool_args.get("think") or None

    raw_skills = tool_args.get("new_skills", [])
    if not isinstance(raw_skills, list):
        return {"uuid": uuid, "status": "bad_format", "skills": [], "rejected": [],
                "think": think, "thinking": raw["thinking"], "content": raw["content"],
                "tool_call_raw": raw["tool_call_raw"],
                "input_messages": messages,
                "source_record": _source_meta(record)}

    # Clamp to 2 skills
    raw_skills = raw_skills[:2]

    accepted = []
    rejected = []
    for skill in raw_skills:
        if not isinstance(skill, dict):
            continue
        # Validate required fields
        missing = [f for f in ("name", "description", "scope", "content") if not skill.get(f)]
        if missing:
            rejected.append({"reason": f"missing_fields:{missing}", "skill": skill})
            continue
        if _is_lazy_skill(skill):
            rejected.append({"reason": "lazy_optimization", "skill": skill})
            continue
        # Normalise tags
        if not isinstance(skill.get("tags"), list):
            skill["tags"] = []
        accepted.append(skill)

    status = "ok" if accepted else "rejected"
    return {
        "uuid": uuid,
        "status": status,
        # ── model outputs (full preservation) ─────────────────────────────
        "think": think,                        # model's pre-skill analysis (from tool arg)
        "thinking": raw["thinking"],           # reasoning/thinking tokens or None
        "content": raw["content"],             # assistant text message or None
        "tool_call_raw": raw["tool_call_raw"], # raw tool call dict (id + args string)
        # ── processed skill results ────────────────────────────────────────
        "skills": accepted,
        "rejected": rejected,
        # ── inputs ────────────────────────────────────────────────────────
        "input_messages": messages,
        "source_record": _source_meta(record),
    }


def _source_meta(record: dict) -> dict:
    return {
        "best_round": record.get("best_round"),
        "round_speedups": record.get("round_speedups"),
        "num_rounds": record.get("num_rounds"),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Generate skills from optimization trajectories.")
    parser.add_argument(
        "--input", "-i",
        default="data/skill/skill_generation_seed.jsonl",
        help="Path to skill_generation_seed.jsonl",
    )
    parser.add_argument(
        "--output", "-o",
        default="data/skill/generated_skills.jsonl",
        help="Output JSONL path",
    )
    parser.add_argument(
        "--model", "-m",
        default="gpt-4o",
        help="Model name",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="OpenAI-compatible API base URL (e.g. http://localhost:8000/v1)",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="API key (defaults to OPENAI_API_KEY env var)",
    )
    parser.add_argument(
        "--subset", "-n",
        type=int,
        default=None,
        help="Only process first N records (for testing)",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Skip first N records",
    )
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=1,
        help="Number of parallel worker threads",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip records whose uuid already appears in the output file",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Number of API retries per record",
    )
    return parser.parse_args()


def load_done_uuids(output_path: str) -> set:
    done = set()
    if not os.path.exists(output_path):
        return done
    with open(output_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if "uuid" in obj:
                    done.add(obj["uuid"])
            except json.JSONDecodeError:
                pass
    return done


def main():
    args = parse_args()

    # Build OpenAI client
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY", "EMPTY")
    client_kwargs: dict[str, Any] = {"api_key": api_key}
    if args.base_url:
        client_kwargs["base_url"] = args.base_url
    client = OpenAI(**client_kwargs)

    # Load records
    print(f"Loading records from {args.input} ...")
    records = []
    with open(args.input, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"[warn] Skipping malformed line: {e}")

    # Apply offset / subset
    records = records[args.offset:]
    if args.subset is not None:
        records = records[: args.subset]

    print(f"Total records to process: {len(records)}")
    # args.resume = True
    # Resume support
    done_uuids: set = set()
    if args.resume:
        done_uuids = load_done_uuids(args.output)
        before = len(records)
        records = [r for r in records if r.get("uuid") not in done_uuids]
        print(f"Resuming: skipped {before - len(records)} already-done records, {len(records)} remaining")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    out_mode = "a" if args.resume else "w"

    stats = {"ok": 0, "rejected": 0, "api_error": 0, "bad_format": 0, "total_skills": 0}

    def _process(record):
        return process_record(record, client, args.model)

    print(f"Processing with {args.workers} worker(s), model={args.model} ...")
    with open(args.output, out_mode) as fout:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(_process, r): r for r in records}
            done_count = 0
            for future in as_completed(futures):
                done_count += 1
                record = futures[future]
                uuid = record.get("uuid", "?")
                try:
                    result = future.result()
                except Exception:
                    tb = traceback.format_exc()
                    print(f"  [error] uuid={uuid}: {tb}")
                    result = {"uuid": uuid, "status": "exception", "skills": []}

                if result is None:
                    result = {"uuid": uuid, "status": "none", "skills": []}

                fout.write(json.dumps(result, ensure_ascii=False) + "\n")
                fout.flush()

                status = result.get("status", "?")
                n_skills = len(result.get("skills", []))
                n_rejected = len(result.get("rejected", []))
                stats[status] = stats.get(status, 0) + 1
                stats["total_skills"] += n_skills

                print(
                    f"  [{done_count}/{len(records)}] uuid={uuid}  "
                    f"status={status}  skills={n_skills}  rejected={n_rejected}"
                )

    print("\n=== Summary ===")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    print(f"Output written to: {args.output}")


if __name__ == "__main__":
    main()
