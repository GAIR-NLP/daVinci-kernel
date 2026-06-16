"""
Merge and shuffle the following datasets into a single parquet for SFT training:
  1. drkernel-coldstart-8k.parquet  (already has `messages` column)
  2. generated_skills_with_retrieval.jsonl  (tool: update_skill_library)
  3. selected_skill_data.jsonl             (tool: select_skills)
  4. generated_skills.jsonl               (tool: update_skill_library, skip api_error rows)
  5. skill_injected_policy_sft.parquet    (policy SFT with skill injected into prompt)

Output: data/skill/combined_sft.parquet
        columns: messages, tools, enable_thinking

Tool schema is embedded as `tools` column so the chat template can render
tool_call / tool assistant turns properly.
"""

import json
import os
import random
import pyarrow as pa
import pyarrow.parquet as pq

# ── tool schemas ─────────────────────────────────────────────────────────────

UPDATE_SKILL_LIBRARY_TOOL = {
    "type": "function",
    "function": {
        "name": "update_skill_library",
        "description": (
            "Add new reusable optimization skills to the skill library based on "
            "what you learned from the current trajectory."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "think": {
                    "type": "string",
                    "description": "Brief analysis of what the key bottleneck was and what the winning technique was.",
                },
                "new_skills": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "scope": {"type": "string"},
                            "description": {"type": "string"},
                            "tags": {"type": "array", "items": {"type": "string"}},
                            "content": {"type": "string"},
                        },
                        "required": ["name", "scope", "description", "tags", "content"],
                    },
                    "description": "List of new skills to add.",
                },
            },
            "required": ["think", "new_skills"],
        },
    },
}

SELECT_SKILLS_TOOL = {
    "type": "function",
    "function": {
        "name": "select_skills",
        "description": (
            "Select the most relevant skills from the skill library for the current task."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "think": {
                    "type": "string",
                    "description": "Brief analysis of the task bottleneck and why these skills are chosen.",
                },
                "selected_skills": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "reason": {"type": "string"},
                        },
                        "required": ["name", "reason"],
                    },
                    "description": "Top 3 most relevant skill names with reasons.",
                },
            },
            "required": ["think", "selected_skills"],
        },
    },
}


# ── helpers ───────────────────────────────────────────────────────────────────

def make_tool_call_message(tool_call_raw: dict) -> dict:
    """Convert raw tool_call dict to an assistant tool_call message.

    QWEN3CHATTEMPLATE (constant.py) uses `tool_call.arguments | items` which
    requires arguments to be a dict, not a JSON string.
    """
    args = tool_call_raw["function"]["arguments"]
    if isinstance(args, str):
        args = json.loads(args)
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": tool_call_raw["id"],
                "type": "function",
                "function": {
                    "name": tool_call_raw["function"]["name"],
                    "arguments": args,
                },
            }
        ],
    }


def build_messages_from_skill_row(row: dict) -> list[dict] | None:
    """
    Build a messages list from a skill JSONL row.
    Returns None if the row is invalid / missing required fields.
    """
    tool_call_raw = row.get("tool_call_raw")
    input_messages = row.get("input_messages")

    if not tool_call_raw or not input_messages:
        return None
    if row.get("status") != "ok":
        return None

    messages = list(input_messages)  # [system, user, ...]

    # Optional thinking prefix in the assistant reply
    thinking = row.get("thinking")
    content_prefix = row.get("content")

    assistant_msg = make_tool_call_message(tool_call_raw)
    # If the model produced <think>…</think> before the tool call, attach it
    if thinking and str(thinking) not in ("None", ""):
        assistant_msg["thinking"] = str(thinking)
    if content_prefix and str(content_prefix) not in ("None", ""):
        assistant_msg["content"] = str(content_prefix)

    messages.append(assistant_msg)
    return messages


# ── 1. parquet coldstart ──────────────────────────────────────────────────────

PARQUET_PATH = (
    "."  # set to your daVinci-kernel repo root
    "/data/drkernel-coldstart-8k/drkernel-coldstart-8k.parquet"
)

print("Loading coldstart parquet …")
t = pq.read_table(PARQUET_PATH)
coldstart_rows = []
for i in range(t.num_rows):
    row = {col: t[col][i].as_py() for col in t.schema.names}
    coldstart_rows.append({
        "messages": row["messages"],
        "tools": None,
        "enable_thinking": row.get("enable_thinking", False),
    })
print(f"  coldstart rows: {len(coldstart_rows)}")


# ── 2. generated_skills_with_retrieval.jsonl ──────────────────────────────────

GENERATED_SKILLS_PATH = (
    "."  # set to your daVinci-kernel repo root
    "/data/skill/generated_skills_with_retrieval.jsonl"
)

print("Loading generated_skills_with_retrieval.jsonl …")
gen_skill_rows = []
skipped_gen = 0
with open(GENERATED_SKILLS_PATH) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        msgs = build_messages_from_skill_row(row)
        if msgs is None:
            skipped_gen += 1
            continue
        gen_skill_rows.append({
            "messages": msgs,
            "tools": [UPDATE_SKILL_LIBRARY_TOOL],
            "enable_thinking": False,
        })
print(f"  generated_skills rows: {len(gen_skill_rows)}  (skipped {skipped_gen})")



# ── 3. selected_skill_data.jsonl ──────────────────────────────────────────────

SELECTED_SKILLS_PATH = (
    "."  # set to your daVinci-kernel repo root
    "/data/skill/selected_skill_data.jsonl"
)

print("Loading selected_skill_data.jsonl …")
sel_skill_rows = []
skipped_sel = 0
with open(SELECTED_SKILLS_PATH) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        msgs = build_messages_from_skill_row(row)
        if msgs is None:
            skipped_sel += 1
            continue
        sel_skill_rows.append({
            "messages": msgs,
            "tools": [SELECT_SKILLS_TOOL],
            "enable_thinking": False,
        })
print(f"  selected_skill rows: {len(sel_skill_rows)}  (skipped {skipped_sel})")


# ── 4. generated_skills.jsonl  (skip api_error rows) ─────────────────────────

GEN_SKILLS2_PATH = (
    "."  # set to your daVinci-kernel repo root
    "/data/skill/generated_skills.jsonl"
)

print("Loading generated_skills.jsonl …")
gen_skills2_rows = []
skipped_gen2 = 0
with open(GEN_SKILLS2_PATH) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        msgs = build_messages_from_skill_row(row)
        if msgs is None:
            skipped_gen2 += 1
            continue
        gen_skills2_rows.append({
            "messages": msgs,
            "tools": [UPDATE_SKILL_LIBRARY_TOOL],
            "enable_thinking": False,
        })
print(f"  generated_skills (new) rows: {len(gen_skills2_rows)}  (skipped {skipped_gen2})")


# ── 5. skill_injected_policy_sft.parquet ──────────────────────────────────────

SKILL_INJECTED_PARQUET_PATH = (
    "."  # set to your daVinci-kernel repo root
    "/data/skill/skill_injected_policy_sft.parquet"
)

print("Loading skill_injected_policy_sft.parquet …")
skill_injected_rows = []
if os.path.exists(SKILL_INJECTED_PARQUET_PATH):
    t_inj = pq.read_table(SKILL_INJECTED_PARQUET_PATH)
    for i in range(t_inj.num_rows):
        row = {col: t_inj[col][i].as_py() for col in t_inj.schema.names}
        skill_injected_rows.append({
            "messages": json.loads(row["messages"]) if isinstance(row["messages"], str) else row["messages"],
            "tools": None,
            "enable_thinking": bool(row.get("enable_thinking", False)),
        })
    print(f"  skill_injected_policy rows: {len(skill_injected_rows)}")
else:
    print(f"  WARNING: {SKILL_INJECTED_PARQUET_PATH} not found — skipping. "
          f"Run data/skill/generate_skill_injected_sft.py first.")


# ── merge + shuffle ───────────────────────────────────────────────────────────

all_rows = coldstart_rows + gen_skill_rows + sel_skill_rows + gen_skills2_rows + skill_injected_rows
print(f"\nTotal before shuffle: {len(all_rows)}")

random.seed(42)
random.shuffle(all_rows)

# ── write parquet ─────────────────────────────────────────────────────────────

OUT_PATH = (
    "."  # set to your daVinci-kernel repo root
    "/data/skill/combined_sft_v2.parquet"
)

print(f"Writing to {OUT_PATH} …")

schema = pa.schema([
    pa.field("messages", pa.list_(pa.struct([
        pa.field("role", pa.string()),
        pa.field("content", pa.string()),
    ]))),
    pa.field("tools", pa.large_string()),        # JSON-serialized list or null
    pa.field("enable_thinking", pa.bool_()),
])

messages_list = []
tools_list = []
enable_thinking_list = []

for r in all_rows:
    # Normalise messages: keep only role+content for coldstart rows;
    # for skill rows the assistant message may have extra keys (tool_calls,
    # thinking) that pyarrow can't represent in a fixed struct schema —
    # so we JSON-serialise the full messages list just like the tools column.
    messages_list.append(json.dumps(r["messages"]))
    tools_list.append(json.dumps(r["tools"]) if r["tools"] is not None else None)
    enable_thinking_list.append(bool(r["enable_thinking"]) if r["enable_thinking"] is not None else False)

# Use large_string for all three variable-length columns to avoid size limits
table = pa.table(
    {
        "messages": pa.array(messages_list, type=pa.large_string()),
        "tools": pa.array(tools_list, type=pa.large_string()),
        "enable_thinking": pa.array(enable_thinking_list, type=pa.bool_()),
    }
)

pq.write_table(table, OUT_PATH)
print(f"Done. Wrote {len(all_rows)} rows → {OUT_PATH}")
print(f"  coldstart:                    {len(coldstart_rows)}")
print(f"  update_skill_library (retri): {len(gen_skill_rows)}")
print(f"  select_skills:                {len(sel_skill_rows)}")
print(f"  update_skill_library (new):   {len(gen_skills2_rows)}")
print(f"  skill_injected_policy:        {len(skill_injected_rows)}")
