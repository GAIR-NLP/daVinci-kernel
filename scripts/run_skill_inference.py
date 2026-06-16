"""
run_skill_inference.py — Skill generation with optional BM25 retrieval augmentation.

Extends skill_generator.py with two settings:

  no_skills   : identical to skill_generator.py — no retrieval, no injection.
  with_skills : retrieves top-k existing skills from the library via BM25 and
                injects them into the user message so the model knows what already
                exists (to avoid duplication and find complementary insights).

Output format is identical to skill_generator.py in both settings; with_skills
adds two extra fields: `retrieved_skills` and `retrieved_skills_content`.

Resume: --resume skips uuids already in --output.
        --resume-from FILE [FILE ...] additionally loads done uuids from other
        files (e.g. a pre-existing generated_skills.jsonl), regardless of setting.

Usage:
    # 1. Extract skill library (run once after skill_generator.py has run):
    python extract_skills.py --input  ../data/skill/generated_skills.jsonl \\
                             --output ../data/skill/skill_library.jsonl

    # 2. Run with_skills (picks up where skill_generator.py left off):
    python run_skill_inference.py \
        --input         ../data/skill/skill_generation_seed.jsonl \
        --output        ../data/skill/generated_skills_with_retrieval.jsonl \
        --skill-lib     ../data/skill/skill_library.jsonl \
        --setting       with_skills \
        --top-k         3 \
        --model         gpt-5.4 \
        --base-url      https://apicz.boyuerichdata.com/v1 \
        --api-key       sk-7RQKqvghSTbyev4xnnq3s8qML4Awkm2Ja86pMK42l3pohYn7 \
        --subset        4672 \
        --workers       20 \
        --resume \
        --resume-from   ../data/skill/generated_skills.jsonl

    # 3. Run no_skills on remaining records (continues into same output file):
    python run_skill_inference.py \\
        --input         ../data/skill/skill_generation_seed.jsonl \\
        --output        ../data/skill/generated_skills_with_retrieval.jsonl \\
        --setting       no_skills \\
        --model         gpt-5.4 \\
        --base-url      https://apicz.boyuerichdata.com/v1 \\
        --api-key       sk-xxx \\
        --workers       10 \\
        --resume \\
        --resume-from   ../data/skill/generated_skills.jsonl
"""

import argparse
import json
import os
import re
import string
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Optional

from openai import OpenAI

try:
    from rank_bm25 import BM25Okapi
    _HAS_RANK_BM25 = True
except ImportError:
    _HAS_RANK_BM25 = False

# ---------------------------------------------------------------------------
# Prompts  (copied verbatim from skill_generator.py)
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
# Tool definition  (identical to skill_generator.py)
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
# Rejection filters  (identical to skill_generator.py)
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


def _is_lazy_skill(skill: dict) -> bool:
    text = " ".join([
        skill.get("name", ""),
        skill.get("description", ""),
        skill.get("content", ""),
    ])
    return bool(_LAZY_RE.search(text))


# ---------------------------------------------------------------------------
# BM25 retriever
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> list[str]:
    text = text.lower()
    text = re.sub(r"[_\-/]", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    return [t for t in text.split() if t]


class BM25Retriever:
    def __init__(self, skills: list[dict]):
        if not _HAS_RANK_BM25:
            raise ImportError(
                "rank_bm25 is required for skill retrieval. "
                "Install it with: pip install rank-bm25"
            )
        self.skills = skills
        tokenized_corpus = []
        for s in skills:
            doc_text = " ".join([
                s.get("name", ""),
                s.get("description", ""),
                " ".join(s.get("tags", [])),
                s.get("content", ""),
            ])
            tokenized_corpus.append(_tokenize(doc_text))
        self._bm25 = BM25Okapi(tokenized_corpus)

    def retrieve(self, query: str, top_k: int = 3) -> list[dict]:
        if not self.skills:
            return []
        q_tokens = _tokenize(query)
        scores = self._bm25.get_scores(q_tokens)
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [
            {**self.skills[i], "_bm25_score": round(float(scores[i]), 4)}
            for i in ranked[:top_k]
            if scores[i] > 0
        ]


def load_skill_library(path: str) -> list[dict]:
    skills = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    skills.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return skills


# ---------------------------------------------------------------------------
# Skill injection into user message
# ---------------------------------------------------------------------------

_EXISTING_SKILLS_HEADER = """\
## Existing Skills (retrieved as relevant to this task)

The following skills are already in the library. Study them to avoid duplication
and to look for *complementary* or more specific insights in the trajectory below.

"""

_EXISTING_SKILLS_FOOTER = "\n\n---\n\n"


def _format_existing_skills_block(skills: list[dict]) -> str:
    parts = []
    for s in skills:
        parts.append(f"### {s['name']}\n")
        parts.append(f"_{s.get('description', '')}_\n\n")
        parts.append(s.get("content", "").strip())
        parts.append("\n\n")
    return "".join(parts).strip()


def inject_skills_into_user_message(user_content: str, skills: list[dict]) -> str:
    """Prepend the retrieved skill block to the user message."""
    block = (
        _EXISTING_SKILLS_HEADER
        + _format_existing_skills_block(skills)
        + _EXISTING_SKILLS_FOOTER
    )
    return block + user_content


# ---------------------------------------------------------------------------
# Trajectory formatting  (identical to skill_generator.py)
# ---------------------------------------------------------------------------

def _format_trajectory(record: dict) -> str:
    messages = record["messages"]
    round_speedups = record.get("round_speedups", [])
    best_round = record.get("best_round", None)

    task_prompt = messages[0]["content"] if messages else ""
    parts = ["## Original Task\n", task_prompt, "\n"]

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


def _build_user_content(record: dict) -> str:
    trajectory_text = _format_trajectory(record)

    best_round = record.get("best_round", None)
    round_speedups = record.get("round_speedups", [])
    turn1_speedup = round_speedups[0] if round_speedups else None
    best_speedup = (
        round_speedups[best_round - 1]
        if (best_round and best_round <= len(round_speedups))
        else None
    )

    header = []
    if turn1_speedup is not None:
        header.append(f"Turn 1 speedup: {turn1_speedup:.4f}x")
    if best_speedup is not None:
        header.append(f"Best turn speedup: {best_speedup:.4f}x (round {best_round})")
    header_str = "  |  ".join(header) + "\n\n" if header else ""

    return (
        f"{header_str}"
        f"The trajectory below shows a multi-turn kernel optimization session. "
        f"The model improved its kernel across rounds — study the full conversation to understand "
        f"what techniques were tried, what worked, and what pitfalls were encountered.\n\n"
        f"{trajectory_text}\n"
        f"Now call update_skill_library with at most 2 new, general, reusable skills "
        f"extracted from this trajectory. Focus on techniques that would help achieve "
        f"similar improvements on OTHER tasks, not just this one."
    )


def build_messages(
    record: dict,
    retriever: Optional[BM25Retriever],
    top_k: int,
    setting: str,
) -> tuple[list[dict], list[dict]]:
    """
    Returns (messages, retrieved_skills).

    no_skills  : [system, user]  — identical to skill_generator.py
    with_skills: [system, user]  — user message has existing-skills block prepended
    """
    user_content = _build_user_content(record)
    retrieved: list[dict] = []

    if setting == "with_skills" and retriever is not None:
        query = record.get("messages", [{}])[0].get("content", "")[:800]
        retrieved = retriever.retrieve(query, top_k=top_k)
        if retrieved:
            user_content = inject_skills_into_user_message(user_content, retrieved)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_content},
    ]
    return messages, retrieved


# ---------------------------------------------------------------------------
# LLM call  (identical to skill_generator.py)
# ---------------------------------------------------------------------------

def call_model(
    client: OpenAI,
    model: str,
    messages: list[dict],
    max_retries: int = 3,
    retry_delay: float = 5.0,
) -> Optional[dict]:
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

            thinking = getattr(msg, "reasoning_content", None)
            if thinking is None and isinstance(msg.content, list):
                for block in msg.content:
                    if isinstance(block, dict) and block.get("type") == "thinking":
                        thinking = block.get("thinking") or block.get("text")
                        break

            if isinstance(msg.content, str):
                text_content = msg.content or None
            elif isinstance(msg.content, list):
                parts = [
                    b.get("text", "") for b in msg.content
                    if isinstance(b, dict) and b.get("type") == "text"
                ]
                text_content = "".join(parts) or None
            else:
                text_content = None

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
                "tool_args":     tool_args,
                "thinking":      thinking,
                "content":       text_content,
                "tool_call_raw": {
                    "id": tc.id,
                    "function": {
                        "name":      tc.function.name,
                        "arguments": raw_args_str,
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

def _source_meta(record: dict) -> dict:
    return {
        "best_round":    record.get("best_round"),
        "round_speedups": record.get("round_speedups"),
        "num_rounds":    record.get("num_rounds"),
    }


def process_record(
    record: dict,
    client: OpenAI,
    model: str,
    retriever: Optional[BM25Retriever],
    top_k: int,
    setting: str,
    max_retries: int,
) -> dict:
    uuid = record.get("uuid", "?")
    messages, retrieved_skills = build_messages(record, retriever, top_k, setting)

    raw = call_model(client, model, messages, max_retries=max_retries)
    if raw is None:
        return {
            "uuid": uuid, "status": "api_error", "setting": setting,
            "skills": [], "rejected": [],
            "think": None, "thinking": None, "content": None, "tool_call_raw": None,
            "input_messages": messages,
            "retrieved_skills": _strip_content(retrieved_skills),
            "retrieved_skills_content": [s.get("content", "") for s in retrieved_skills],
            "source_record": _source_meta(record),
        }

    tool_args = raw["tool_args"]
    think = tool_args.get("think") or None
    raw_skills = tool_args.get("new_skills", [])
    if not isinstance(raw_skills, list):
        return {
            "uuid": uuid, "status": "bad_format", "setting": setting,
            "skills": [], "rejected": [],
            "think": think, "thinking": raw["thinking"], "content": raw["content"],
            "tool_call_raw": raw["tool_call_raw"],
            "input_messages": messages,
            "retrieved_skills": _strip_content(retrieved_skills),
            "retrieved_skills_content": [s.get("content", "") for s in retrieved_skills],
            "source_record": _source_meta(record),
        }

    raw_skills = raw_skills[:2]
    accepted, rejected = [], []
    for skill in raw_skills:
        if not isinstance(skill, dict):
            continue
        missing = [f for f in ("name", "description", "scope", "content") if not skill.get(f)]
        if missing:
            rejected.append({"reason": f"missing_fields:{missing}", "skill": skill})
            continue
        if _is_lazy_skill(skill):
            rejected.append({"reason": "lazy_optimization", "skill": skill})
            continue
        if not isinstance(skill.get("tags"), list):
            skill["tags"] = []
        accepted.append(skill)

    status = "ok" if accepted else "rejected"
    return {
        "uuid":    uuid,
        "status":  status,
        "setting": setting,
        # ── model outputs ───────────────────────────────────────────────────
        "think":         think,
        "thinking":      raw["thinking"],
        "content":       raw["content"],
        "tool_call_raw": raw["tool_call_raw"],
        # ── processed skill results ─────────────────────────────────────────
        "skills":   accepted,
        "rejected": rejected,
        # ── inputs ──────────────────────────────────────────────────────────
        "input_messages": messages,
        # ── retrieval (empty list for no_skills) ─────────────────────────────
        "retrieved_skills":         _strip_content(retrieved_skills),
        "retrieved_skills_content": [s.get("content", "") for s in retrieved_skills],
        # ── metadata ────────────────────────────────────────────────────────
        "source_record": _source_meta(record),
    }


def _strip_content(skills: list[dict]) -> list[dict]:
    """Return skill metadata without the full content body (saves space)."""
    return [{k: v for k, v in s.items() if k != "content"} for s in skills]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Skill generation with optional BM25 retrieval augmentation."
    )
    p.add_argument("--input",       "-i", default="../data/skill/skill_generation_seed.jsonl")
    p.add_argument("--output",      "-o", default="../data/skill/generated_skills_with_retrieval.jsonl")
    p.add_argument("--skill-lib",         default="../data/skill/skill_library.jsonl",
                   help="Skill library JSONL produced by extract_skills.py (required for with_skills).")
    p.add_argument("--setting",     "-s", choices=["with_skills", "no_skills"],
                   default="no_skills")
    p.add_argument("--top-k",             type=int, default=3,
                   help="Number of skills to retrieve per record (with_skills only).")
    p.add_argument("--model",       "-m", default="gpt-4o")
    p.add_argument("--base-url",          default=None)
    p.add_argument("--api-key",           default=None)
    p.add_argument("--subset",      "-n", type=int, default=None)
    p.add_argument("--offset",            type=int, default=0)
    p.add_argument("--workers",     "-w", type=int, default=1)
    p.add_argument("--resume",            action="store_true",
                   help="Skip uuids already in --output.")
    p.add_argument("--resume-from",       nargs="*", default=[], metavar="FILE",
                   help="Additional JSONL files to read already-done uuids from "
                        "(e.g. a pre-existing generated_skills.jsonl).")
    p.add_argument("--max-retries",       type=int, default=3)
    return p.parse_args()


def load_done_uuids(paths: list[str]) -> set:
    done = set()
    for path in paths:
        if not os.path.exists(path):
            print(f"[resume] {path} not found, skipping.")
            continue
        count = 0
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if "uuid" in obj:
                        done.add(obj["uuid"])
                        count += 1
                except json.JSONDecodeError:
                    pass
        print(f"[resume] loaded {count} uuids from {path}")
    return done


def main():
    args = parse_args()

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY", "EMPTY")
    client_kwargs: dict[str, Any] = {"api_key": api_key}
    if args.base_url:
        client_kwargs["base_url"] = args.base_url
    client = OpenAI(**client_kwargs)

    # Skill library + retriever (only needed for with_skills)
    retriever: Optional[BM25Retriever] = None
    if args.setting == "with_skills":
        if not _HAS_RANK_BM25:
            print("[error] rank_bm25 not installed. Run: pip install rank-bm25", file=sys.stderr)
            sys.exit(1)
        if not os.path.exists(args.skill_lib):
            print(f"[warn] Skill library not found: {args.skill_lib}")
            print("[warn] Proceeding without retrieval (equivalent to no_skills).")
        else:
            skills = load_skill_library(args.skill_lib)
            print(f"Loaded {len(skills)} skills from {args.skill_lib}")
            if skills:
                retriever = BM25Retriever(skills)

    # Load seed records
    print(f"Loading records from {args.input} ...")
    records = []
    with open(args.input) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"[warn] Skipping malformed line: {e}")

    records = records[args.offset:]
    if args.subset is not None:
        records = records[: args.subset]
    print(f"Total records: {len(records)}")

    # Resume
    if args.resume or args.resume_from:
        resume_paths = list(args.resume_from or [])
        if args.resume:
            resume_paths.append(args.output)
        done_uuids = load_done_uuids(resume_paths)
        before = len(records)
        records = [r for r in records if r.get("uuid") not in done_uuids]
        print(f"Resume: skipped {before - len(records)}, {len(records)} remaining")

    if not records:
        print("Nothing to do.")
        return

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    out_mode = "a" if (args.resume or args.resume_from) else "w"

    stats: dict = {"ok": 0, "rejected": 0, "api_error": 0, "bad_format": 0, "total_skills": 0}

    def _process(rec):
        return process_record(
            rec, client, args.model, retriever,
            args.top_k, args.setting, args.max_retries,
        )

    print(f"Setting={args.setting}  top_k={args.top_k}  model={args.model}  workers={args.workers}")
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
                    result = {"uuid": uuid, "status": "exception", "setting": args.setting,
                              "skills": [], "rejected": []}
                if result is None:
                    result = {"uuid": uuid, "status": "none", "setting": args.setting,
                              "skills": [], "rejected": []}

                fout.write(json.dumps(result, ensure_ascii=False) + "\n")
                fout.flush()

                status = result.get("status", "?")
                n_skills = len(result.get("skills", []))
                n_rejected = len(result.get("rejected", []))
                n_retrieved = len(result.get("retrieved_skills", []))
                stats[status] = stats.get(status, 0) + 1
                stats["total_skills"] += n_skills

                print(
                    f"  [{done_count}/{len(records)}] uuid={uuid}  "
                    f"status={status}  skills={n_skills}  rejected={n_rejected}"
                    + (f"  retrieved={n_retrieved}" if args.setting == "with_skills" else "")
                )

    print("\n=== Summary ===")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
