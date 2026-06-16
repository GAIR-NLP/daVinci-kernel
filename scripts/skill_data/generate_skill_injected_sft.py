"""
Generate skill-injected policy SFT data.

For each entry in skill_generation_seed.jsonl that has a matched selected_skills
entry (from selected_skill_data.jsonl), inject the selected skill content into
the original multi-turn policy conversation, then save as parquet.

This data closes the SFT–RL distribution gap: during RL rollout the policy
sees prompts with skill content injected; this SFT data teaches the model how
to utilise those injected skills when writing CUDA kernels.

Output columns (identical to combined_sft.parquet):
  messages        large_string  JSON-serialised list of message dicts
  tools           large_string  null (policy SFT has no tool calls)
  enable_thinking bool          false

Usage (run from the KernelGYM root):
  python data/skill/generate_skill_injected_sft.py
"""

import json
import sys
import os

import pyarrow as pa
import pyarrow.parquet as pq
from omegaconf import OmegaConf

# ── path setup ────────────────────────────────────────────────────────────────
REPO_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
DRKERNEL_ROOT = os.path.join(REPO_ROOT, "drkernel")
sys.path.insert(0, DRKERNEL_ROOT)

from kernel.skill.skill_prompt_builder import inject_skill_into_messages  # noqa: E402

# ── config ────────────────────────────────────────────────────────────────────
BASE = os.environ.get("DAVINCI_KERNEL_ROOT", ".")
SEED_PATH    = f"{BASE}/data/skill/skill_generation_seed.jsonl"
SEL_PATH     = f"{BASE}/data/skill/selected_skill_data.jsonl"
OUT_PATH     = f"{BASE}/data/skill/skill_injected_policy_sft.parquet"
PROMPTS_YAML = f"{BASE}/drkernel/kernel/config/skill_prompts.yaml"


# ── load skill_cfg (for inject header/footer templates) ───────────────────────
skill_cfg = OmegaConf.load(PROMPTS_YAML)


def build_skill_context(selected_skills: list[dict]) -> str:
    """Format a list of skill dicts into a single skill_content string.

    Mirrors SkillLibrary.get_skill_content() output format so that the
    injected text is identical to what vllm_async_engine_skill.py produces
    at RL inference time.
    """
    parts = []
    for sk in selected_skills:
        name = sk.get("name", "")
        description = sk.get("description", "")
        scope = sk.get("scope", "general")
        tags = sk.get("tags", [])
        content = sk.get("content", "")
        tags_str = json.dumps(tags) if isinstance(tags, list) else str(tags)
        header = (
            f"---\n"
            f"name: {name}\n"
            f"description: {description}\n"
            f"scope: {scope}\n"
            f"tags: {tags_str}\n"
            f"---\n\n"
            f"{content}"
        )
        parts.append(header)
    return "\n\n---\n\n".join(parts)


# ── load selected_skill_data.jsonl → uuid → selected_skills ──────────────────
print("Loading selected_skill_data.jsonl …")
uuid_to_skills: dict[str, list[dict]] = {}
skipped_sel = 0
with open(SEL_PATH) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        if row.get("status") != "ok":
            skipped_sel += 1
            continue
        skills = row.get("selected_skills", [])
        if not skills:
            skipped_sel += 1
            continue
        uuid_to_skills[str(row["uuid"])] = skills
print(f"  matched uuids: {len(uuid_to_skills)}  (skipped {skipped_sel})")


# ── process skill_generation_seed.jsonl ───────────────────────────────────────
print("Loading skill_generation_seed.jsonl and injecting skills …")
output_rows = []
skipped_seed = 0

with open(SEED_PATH) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        seed = json.loads(line)
        uuid_key = str(seed.get("uuid", ""))
        if uuid_key not in uuid_to_skills:
            skipped_seed += 1
            continue

        messages = seed.get("messages")
        if not messages:
            skipped_seed += 1
            continue

        selected_skills = uuid_to_skills[uuid_key]
        skill_context = build_skill_context(selected_skills)

        # Inject skill content into the conversation using the same function
        # and config as RL inference — ensures format consistency.
        injected_messages = inject_skill_into_messages(
            messages, skill_context, skill_cfg
        )

        output_rows.append({
            "messages": injected_messages,
            "tools": None,
            "enable_thinking": bool(seed.get("enable_thinking", False)),
        })

print(f"  generated rows: {len(output_rows)}  (skipped seed entries: {skipped_seed})")


# ── write parquet ─────────────────────────────────────────────────────────────
print(f"Writing to {OUT_PATH} …")

messages_list = [json.dumps(r["messages"]) for r in output_rows]
tools_list = [None] * len(output_rows)
enable_thinking_list = [r["enable_thinking"] for r in output_rows]

table = pa.table(
    {
        "messages": pa.array(messages_list, type=pa.large_string()),
        "tools": pa.array(tools_list, type=pa.large_string()),
        "enable_thinking": pa.array(enable_thinking_list, type=pa.bool_()),
    }
)

pq.write_table(table, OUT_PATH)
print(f"Done. Wrote {len(output_rows)} rows → {OUT_PATH}")
