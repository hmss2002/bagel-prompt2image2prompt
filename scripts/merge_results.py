#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Merge per-rank JSONL outputs into a single combined file keyed by sample_id.

This script takes:
- gen_results_rank*.jsonl
- caption_results_rank*.jsonl
and produces combined_results.jsonl for scoring.
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List


# ------------------------------
# JSONL helpers
# ------------------------------

def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                rows.append(json.loads(s))
    return rows


def read_dir_jsonl(dir_path: str, prefix: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not os.path.isdir(dir_path):
        return rows
    files = sorted([f for f in os.listdir(dir_path) if f.startswith(prefix) and f.endswith(".jsonl")])
    for name in files:
        rows.extend(read_jsonl(os.path.join(dir_path, name)))
    return rows


def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ------------------------------
# Main entrypoint
# ------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen_dir", type=str, required=True)
    ap.add_argument("--caption_dir", type=str, required=True)
    ap.add_argument("--out_jsonl", type=str, required=True)
    args = ap.parse_args()

    gen_rows = read_dir_jsonl(args.gen_dir, "gen_results_rank")
    caption_rows = read_dir_jsonl(args.caption_dir, "caption_results_rank")

    # Index by sample_id for fast merge.
    gen_map = {r["sample_id"]: r for r in gen_rows}
    cap_map = {r["sample_id"]: r for r in caption_rows}

    merged: List[Dict[str, Any]] = []
    for sample_id in sorted(gen_map.keys()):
        g = gen_map[sample_id]
        c = cap_map.get(sample_id)
        row = {
            "sample_id": sample_id,
            "prompt_id": g.get("prompt_id"),
            "prompt_text": g.get("prompt_text"),
            "prompt_slots": g.get("prompt_slots"),
            "seed": g.get("seed"),
            "image_path": g.get("image_path"),
            "gen_status": g.get("status"),
            "gen_error": g.get("error"),
            "caption_status": c.get("status") if c else "missing",
            "caption_error": c.get("error") if c else "missing",
            "inferred_text": c.get("inferred_text") if c else None,
            "params": g.get("params"),
        }
        merged.append(row)

    write_jsonl(args.out_jsonl, merged)
    print(f"[merge_results] merged={len(merged)} out={args.out_jsonl}")


if __name__ == "__main__":
    main()
