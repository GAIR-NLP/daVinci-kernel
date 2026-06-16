"""
extract_skills.py — Extract accepted skills from generated_skills.jsonl into a flat skill library.

Usage:
    python extract_skills.py \
        --input  ../data/skill/generated_skills.jsonl \
        --output ../data/skill/skill_library.jsonl

Output format (one skill per line):
    {
        "name": "vectorized_load_store",
        "description": "...",
        "scope": "general",
        "tags": ["memory", "vectorization"],
        "content": "## Motivation\n...\n## Key Idea\n...\n## Example\n...",
        "source_uuid": 42
    }
"""

import argparse
import json
import os
from collections import Counter


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  "-i", default="../data/skill/generated_skills.jsonl")
    parser.add_argument("--output", "-o", default="../data/skill/skill_library.jsonl")
    parser.add_argument(
        "--dedup",
        action="store_true",
        default=True,
        help="Skip skills whose name already appeared (keep first occurrence).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    records = []
    with open(args.input) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    seen_names: set = set()
    skills_out = []
    status_counts: Counter = Counter()

    for rec in records:
        status_counts[rec.get("status", "?")] += 1
        for skill in rec.get("skills", []):
            name = skill.get("name", "").strip()
            if args.dedup and name in seen_names:
                continue
            seen_names.add(name)
            skills_out.append({
                "name":        name,
                "description": skill.get("description", ""),
                "scope":       skill.get("scope", "general"),
                "tags":        skill.get("tags", []),
                "content":     skill.get("content", ""),
                "source_uuid": rec.get("uuid"),
            })

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        for s in skills_out:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"Input records : {len(records)}")
    print(f"Status counts : {dict(status_counts)}")
    print(f"Skills written: {len(skills_out)}  (dedup={args.dedup})")
    print(f"Output        : {args.output}")


if __name__ == "__main__":
    main()
