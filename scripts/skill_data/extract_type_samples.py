#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd


TYPE_KEYS = (
    "type",
    "data_type",
    "task_type",
    "category",
    "source_type",
    "sample_type",
)

POLICY_KEYS = ("policy", "system_policy", "policy_text")


def _safe_json_loads(v: Any) -> Any:
    if isinstance(v, str):
        s = v.strip()
        if (s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]")):
            try:
                return json.loads(s)
            except Exception:
                return v
    return v


def _iter_dicts(obj: Any) -> Iterable[Dict[str, Any]]:
    if isinstance(obj, dict):
        yield obj
        for vv in obj.values():
            yield from _iter_dicts(vv)
    elif isinstance(obj, list):
        for item in obj:
            yield from _iter_dicts(item)


def _extract_type(row: Dict[str, Any], messages_obj: Any) -> str:
    for k in TYPE_KEYS:
        if k in row and pd.notna(row[k]):
            return str(row[k]).strip() or "unknown"

    for d in _iter_dicts(messages_obj):
        for k in TYPE_KEYS:
            if k in d and d[k] is not None:
                s = str(d[k]).strip()
                if s:
                    return s

    text = json.dumps(messages_obj, ensure_ascii=False) if messages_obj is not None else ""
    for pat in (
        r'"type"\s*:\s*"([^"]+)"',
        r'"task_type"\s*:\s*"([^"]+)"',
        r'"data_type"\s*:\s*"([^"]+)"',
        r"\btype\s*[:=]\s*([A-Za-z0-9_\-./]+)",
    ):
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            return m.group(1).strip()

    return "unknown"


def _extract_policy_text(row: Dict[str, Any], messages_obj: Any) -> str:
    for k in POLICY_KEYS:
        if k in row and pd.notna(row[k]):
            return str(row[k])

    for d in _iter_dicts(messages_obj):
        for k in POLICY_KEYS:
            if k in d and d[k] is not None:
                return str(d[k])

    if isinstance(messages_obj, list):
        system_contents: List[str] = []
        all_contents: List[str] = []
        for m in messages_obj:
            if not isinstance(m, dict):
                continue
            c = m.get("content")
            if c is None:
                continue
            cs = str(c)
            all_contents.append(cs)
            if str(m.get("role", "")).lower() == "system":
                system_contents.append(cs)
        if system_contents:
            return "\n".join(system_contents)
        return "\n".join(all_contents)

    return json.dumps(messages_obj, ensure_ascii=False) if messages_obj is not None else ""


def _jsonable(v: Any) -> Any:
    if isinstance(v, float) and pd.isna(v):
        return None
    if hasattr(v, "item"):
        try:
            return v.item()
        except Exception:
            pass
    return v


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample one row per (type, policy_has_skill).")
    parser.add_argument("input_parquet", type=str)
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="",
        help="Output json path. Default: <input_stem>_type_policy_samples.json",
    )
    args = parser.parse_args()

    in_path = Path(args.input_parquet)
    if not in_path.exists():
        raise FileNotFoundError(f"Input not found: {in_path}")

    out_path = Path(args.output) if args.output else in_path.with_name(f"{in_path.stem}_type_policy_samples.json")

    df = pd.read_parquet(in_path)
    records = df.to_dict(orient="records")

    picked: Dict[str, Dict[str, Any]] = {}
    for rec in records:
        messages_obj = _safe_json_loads(rec.get("messages"))
        data_type = _extract_type(rec, messages_obj)
        policy_text = _extract_policy_text(rec, messages_obj)
        policy_has_skill = "skill" in policy_text.lower()
        key = f"{data_type}__policy_has_skill_{str(policy_has_skill).lower()}"
        if key in picked:
            continue

        cleaned = {k: _jsonable(v) for k, v in rec.items()}
        picked[key] = {
            "type": data_type,
            "policy_has_skill": policy_has_skill,
            "sample": cleaned,
        }

    out = {
        "input": str(in_path),
        "total_rows": len(records),
        "group_count": len(picked),
        "groups": picked,
    }
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved: {out_path}")
    print(f"groups: {len(picked)}")


if __name__ == "__main__":
    main()
