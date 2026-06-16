"""
selected_skill_data.py — Skill selector: given a trajectory, BM25-retrieve top-20
candidate skills from the library, then ask the LLM to pick the top-3 most relevant.

Pipeline per record:
  1. Parse the task from the seed record (first user message = task description + code).
  2. BM25-retrieve top-20 skills from the library (full content indexed, only headers shown).
  3. Build a single-turn [system, user] message with the task + top-20 headers.
  4. Call the model once with tool_choice=required → select_skills tool.
     Tool schema: { think, selected_skills: [{name, reason}] × ≤3 }
  5. Save: uuid, status, think, thinking, content, tool_call_raw,
          bm25_candidates (top-20 headers), selected_skills (full records),
          input_messages, source_record.

Usage:
    python selected_skill_data.py \
        --input      ../data/skill/skill_generation_seed.jsonl \
        --output     ../data/skill/selected_skill_data.jsonl \
        --skill-lib  ../data/skill/skill_library.jsonl \
        --top-bm25   30 \
        --top-select 3 \
        --model      gpt-5.4 \
        --base-url   https://apicz.boyuerichdata.com/v1 \
        --api-key    sk-7RQKqvghSTbyev4xnnq3s8qML4Awkm2Ja86pMK42l3pohYn7 \
        --subset     2000 \
        --workers    20 \
        --resume
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
# Prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert CUDA/Triton kernel optimization engineer.

You will be shown:
1. A kernel optimization task (original PyTorch code + performance target).
2. A numbered list of candidate optimization skills from the skill library.
   Each entry shows only the skill name, description, tags, and scope —
   NOT the full content.

Your job: identify the top 3 skills most likely to help solve THIS specific task.

## Selection criteria
- Relevance: the technique directly applies to the operator/pattern in the task.
- Impact: likely to produce a meaningful speedup for this workload.
- Complementarity: prefer a diverse set that covers different aspects
  (e.g. one memory, one compute, one correctness pitfall).

## What to avoid
- Selecting skills just because they sound impressive.
- Selecting skills about bypassing CUDA/Triton (e.g. "use torch.compile").
- Selecting more than 3 skills.

When calling select_skills, first fill in `think`: briefly explain what the
key bottleneck of this task is and why the chosen skills address it.
Then fill in `selected_skills` with the names and one-sentence reasons.

Call select_skills exactly once. Respond in English only."""


# ---------------------------------------------------------------------------
# Tool definition
# ---------------------------------------------------------------------------

def make_tools(top_select: int) -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": "select_skills",
                "description": (
                    f"Select the top {top_select} most relevant skills from the "
                    f"candidate list for this task. Call exactly once."
                ),
                "parameters": {
                    "type": "object",
                    "required": ["think", "selected_skills"],
                    "properties": {
                        "think": {
                            "type": "string",
                            "description": (
                                "Your analysis before selecting: what is the key bottleneck "
                                "or challenge in this task? Which techniques are most directly "
                                "applicable and why?"
                            ),
                        },
                        "selected_skills": {
                            "type": "array",
                            "description": f"Ordered list of selected skills (at most {top_select}).",
                            "maxItems": top_select,
                            "items": {
                                "type": "object",
                                "required": ["name", "reason"],
                                "properties": {
                                    "name": {
                                        "type": "string",
                                        "description": "Exact snake_case skill name from the candidate list.",
                                    },
                                    "reason": {
                                        "type": "string",
                                        "description": "One sentence: why this skill applies to this task.",
                                    },
                                },
                            },
                        },
                    },
                },
            },
        }
    ]


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
            raise ImportError("rank_bm25 required. Install with: pip install rank-bm25")
        self.skills = skills
        tokenized_corpus = []
        for s in skills:
            # Index the FULL skill content for better recall
            doc_text = " ".join([
                s.get("name", ""),
                s.get("description", ""),
                " ".join(s.get("tags", [])),
                s.get("content", ""),
            ])
            tokenized_corpus.append(_tokenize(doc_text))
        self._bm25 = BM25Okapi(tokenized_corpus)

    def retrieve(self, query: str, top_k: int = 20) -> list[dict]:
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
# Task parsing
# ---------------------------------------------------------------------------

def extract_task_text(record: dict, max_chars: int = 2000) -> str:
    """
    Extract the task description from the first user message.
    Used both as the BM25 query and as the task block shown to the model.
    """
    msgs = record.get("messages", [])
    if msgs:
        return msgs[0].get("content", "")[:max_chars]
    return ""


def extract_bm25_query(record: dict, max_chars: int = 1000) -> str:
    """
    Shorter version used as BM25 query: first user message + original code if available.
    """
    parts = []
    code = record.get("original_python_code", "")
    if code:
        parts.append(code[:500])
    msgs = record.get("messages", [])
    if msgs:
        parts.append(msgs[0].get("content", "")[:500])
    return " ".join(parts)[:max_chars]


# ---------------------------------------------------------------------------
# Message building
# ---------------------------------------------------------------------------

def _format_candidates(candidates: list[dict]) -> str:
    """Format top-N candidates as a numbered list (headers only, no content)."""
    lines = []
    for i, s in enumerate(candidates, 1):
        tags = ", ".join(s.get("tags", []))
        scope = s.get("scope", "general")
        lines.append(
            f"{i:2d}. **{s['name']}**  [scope={scope}]  [tags: {tags}]\n"
            f"    {s.get('description', '')}"
        )
    return "\n".join(lines)


def build_messages(
    record: dict,
    candidates: list[dict],
) -> list[dict]:
    """Build [system, user] for the selector."""
    task_text = extract_task_text(record)
    candidates_block = _format_candidates(candidates)

    user_content = (
        "## Task\n\n"
        f"{task_text}\n\n"
        "---\n\n"
        "## Candidate Skills (from the library)\n\n"
        f"{candidates_block}\n\n"
        "---\n\n"
        "Now call select_skills to choose the top 3 most relevant skills for this task."
    )

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_content},
    ]


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

def call_model(
    client: OpenAI,
    model: str,
    messages: list[dict],
    tools: list[dict],
    max_retries: int = 3,
    retry_delay: float = 5.0,
) -> Optional[dict]:
    """
    Single tool-call request.
    Returns dict with keys: tool_args, thinking, content, tool_call_raw.
    Returns None on unrecoverable failure.
    """
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=tools,
                tool_choice={"type": "function", "function": {"name": "select_skills"}},
                temperature=0.7,
                max_tokens=2048,
            )
            msg = response.choices[0].message

            # reasoning tokens
            thinking = getattr(msg, "reasoning_content", None)
            if thinking is None and isinstance(msg.content, list):
                for block in msg.content:
                    if isinstance(block, dict) and block.get("type") == "thinking":
                        thinking = block.get("thinking") or block.get("text")
                        break

            # text content
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
            if tc.function.name != "select_skills":
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
            print(f"  [warn] JSON decode error: {e} (attempt {attempt+1})")
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
        "best_round":     record.get("best_round"),
        "round_speedups": record.get("round_speedups"),
        "num_rounds":     record.get("num_rounds"),
    }


def _skill_header(s: dict) -> dict:
    """Return only the header fields (no full content) for a skill."""
    return {
        "name":        s.get("name"),
        "description": s.get("description"),
        "scope":       s.get("scope"),
        "tags":        s.get("tags", []),
        "_bm25_score": s.get("_bm25_score"),
    }


def process_record(
    record: dict,
    client: OpenAI,
    model: str,
    retriever: BM25Retriever,
    skill_lookup: dict[str, dict],
    tools: list[dict],
    top_bm25: int,
    max_retries: int,
) -> dict:
    uuid = record.get("uuid", "?")

    # 1. BM25 retrieval
    query = extract_bm25_query(record)
    candidates = retriever.retrieve(query, top_k=top_bm25)

    # 2. Build messages
    messages = build_messages(record, candidates)

    # 3. Call model
    raw = call_model(client, model, messages, tools, max_retries=max_retries)

    bm25_candidates = [_skill_header(c) for c in candidates]

    if raw is None:
        return {
            "uuid":   uuid,
            "status": "api_error",
            "think":         None,
            "thinking":      None,
            "content":       None,
            "tool_call_raw": None,
            "bm25_candidates":   bm25_candidates,
            "selected_skills":   [],
            "input_messages":    messages,
            "source_record":     _source_meta(record),
        }

    tool_args = raw["tool_args"]
    think = tool_args.get("think") or None
    raw_selected = tool_args.get("selected_skills", [])
    if not isinstance(raw_selected, list):
        raw_selected = []

    # 4. Resolve selected skill names → full skill records
    selected_skills = []
    for item in raw_selected:
        name = item.get("name", "")
        full = skill_lookup.get(name)
        if full:
            selected_skills.append({
                "name":        name,
                "reason":      item.get("reason", ""),
                "description": full.get("description", ""),
                "scope":       full.get("scope", ""),
                "tags":        full.get("tags", []),
                "content":     full.get("content", ""),
                "_bm25_score": next(
                    (c["_bm25_score"] for c in candidates if c["name"] == name), None
                ),
            })
        else:
            # Model hallucinated a name not in candidates — record but flag
            selected_skills.append({
                "name":        name,
                "reason":      item.get("reason", ""),
                "_not_found":  True,
            })

    status = "ok" if selected_skills else "empty"
    return {
        "uuid":   uuid,
        "status": status,
        # ── model outputs ────────────────────────────────────────────────
        "think":         think,
        "thinking":      raw["thinking"],
        "content":       raw["content"],
        "tool_call_raw": raw["tool_call_raw"],
        # ── retrieval + selection ────────────────────────────────────────
        "bm25_candidates":  bm25_candidates,    # top-20 headers shown to model
        "selected_skills":  selected_skills,    # top-3 with full content
        # ── inputs ──────────────────────────────────────────────────────
        "input_messages":   messages,
        "source_record":    _source_meta(record),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="BM25 + LLM skill selector.")
    p.add_argument("--input",       "-i", default="../data/skill/skill_generation_seed.jsonl")
    p.add_argument("--output",      "-o", default="../data/skill/selected_skill_data.jsonl")
    p.add_argument("--skill-lib",         default="../data/skill/skill_library.jsonl",
                   help="Skill library JSONL (from extract_skills.py).")
    p.add_argument("--top-bm25",          type=int, default=20,
                   help="Number of BM25 candidates to retrieve.")
    p.add_argument("--top-select",        type=int, default=3,
                   help="Number of skills the LLM selects from the candidates.")
    p.add_argument("--model",       "-m", default="gpt-4o")
    p.add_argument("--base-url",          default=None)
    p.add_argument("--api-key",           default=None)
    p.add_argument("--subset",      "-n", type=int, default=None)
    p.add_argument("--offset",            type=int, default=0)
    p.add_argument("--workers",     "-w", type=int, default=1)
    p.add_argument("--resume",            action="store_true",
                   help="Skip uuids already in --output.")
    p.add_argument("--resume-from",       nargs="*", default=[], metavar="FILE",
                   help="Additional JSONL files to read already-done uuids from.")
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

    if not _HAS_RANK_BM25:
        print("[error] rank_bm25 not installed. Run: pip install rank-bm25", file=sys.stderr)
        sys.exit(1)

    # OpenAI client
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY", "EMPTY")
    client_kwargs: dict[str, Any] = {"api_key": api_key}
    if args.base_url:
        client_kwargs["base_url"] = args.base_url
    client = OpenAI(**client_kwargs)

    # Load skill library
    if not os.path.exists(args.skill_lib):
        print(f"[error] Skill library not found: {args.skill_lib}", file=sys.stderr)
        print("[error] Run extract_skills.py first.", file=sys.stderr)
        sys.exit(1)
    skills = load_skill_library(args.skill_lib)
    print(f"Loaded {len(skills)} skills from {args.skill_lib}")

    retriever = BM25Retriever(skills)
    # Name → full skill dict (for resolving selected names → full records)
    skill_lookup: dict[str, dict] = {s["name"]: s for s in skills}

    tools = make_tools(args.top_select)

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

    stats: dict = {"ok": 0, "empty": 0, "api_error": 0, "total_selected": 0}

    def _process(rec):
        return process_record(
            rec, client, args.model, retriever, skill_lookup,
            tools, args.top_bm25, args.max_retries,
        )

    print(
        f"top_bm25={args.top_bm25}  top_select={args.top_select}  "
        f"model={args.model}  workers={args.workers}"
    )
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
                    result = {
                        "uuid":   uuid,
                        "status": "exception",
                        "selected_skills": [],
                        "bm25_candidates": [],
                    }

                fout.write(json.dumps(result, ensure_ascii=False) + "\n")
                fout.flush()

                status = result.get("status", "?")
                n_sel = len(result.get("selected_skills", []))
                n_cand = len(result.get("bm25_candidates", []))
                stats[status] = stats.get(status, 0) + 1
                stats["total_selected"] += n_sel

                print(
                    f"  [{done_count}/{len(records)}] uuid={uuid}  "
                    f"status={status}  candidates={n_cand}  selected={n_sel}"
                )

    print("\n=== Summary ===")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
