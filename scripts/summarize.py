#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Summarize scored results and produce a report with a small gallery.

The report includes:
- Metric mean/std
- Top cases by chosen metric
- Bottom cases by chosen metric
"""
from __future__ import annotations

import argparse
import json
import math
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


# ------------------------------
# Statistics helpers
# ------------------------------

def mean_std(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0}
    m = sum(values) / len(values)
    var = sum((v - m) ** 2 for v in values) / len(values)
    return {"mean": m, "std": math.sqrt(var)}


def get_metric(rows: List[Dict[str, Any]], key: str) -> List[float]:
    vals: List[float] = []
    for r in rows:
        m = r.get("metrics", {})
        v = m.get(key)
        if isinstance(v, (int, float)):
            vals.append(float(v))
    return vals


def choose_sort_key(rows: List[Dict[str, Any]]) -> str:
    if any(r.get("metrics", {}).get("embed_cos") is not None for r in rows):
        return "embed_cos"
    if any(r.get("metrics", {}).get("slot_f1") is not None for r in rows):
        return "slot_f1"
    return "token_f1"


# ------------------------------
# Report writer
# ------------------------------

def write_report(path: str, rows: List[Dict[str, Any]], top_k: int, bottom_k: int) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sort_key = choose_sort_key(rows)

    def metric_val(r: Dict[str, Any]) -> float:
        v = r.get("metrics", {}).get(sort_key)
        return float(v) if isinstance(v, (int, float)) else -1.0

    rows_sorted = sorted(rows, key=metric_val, reverse=True)
    top_rows = rows_sorted[:top_k]
    bottom_rows = list(reversed(rows_sorted[-bottom_k:]))

    metric_keys = [
        "embed_cos",
        "token_f1",
        "rouge_l",
        "slot_f1",
        "subj_f1",
        "attr_f1",
        "scene_f1",
        "style_f1",
        "action_f1",
        "lighting_f1",
        "camera_f1",
        "composition_f1",
        "subject_f1",
        "attributes_f1",
    ]

    with open(path, "w", encoding="utf-8") as f:
        f.write("# Experiment 2 Report\n\n")
        f.write(f"Sort key: {sort_key}\n\n")

        f.write("## Metrics Summary\n\n")
        for k in metric_keys:
            stats = mean_std(get_metric(rows, k))
            if stats["mean"] == 0.0 and stats["std"] == 0.0 and not get_metric(rows, k):
                continue
            f.write(f"- {k}: mean={stats['mean']:.4f} std={stats['std']:.4f}\n")

        f.write("\n## Top Cases\n\n")
        for r in top_rows:
            f.write(f"### {r.get('sample_id')}\n\n")
            f.write(f"- prompt: {r.get('prompt_text','')}\n")
            f.write(f"- inferred: {r.get('inferred_text','')}\n")
            if r.get("reconstructed_text"):
                f.write(f"- reconstructed: {r.get('reconstructed_text','')}\n")
            f.write(f"- {sort_key}: {r.get('metrics', {}).get(sort_key)}\n\n")
            if r.get("image_path"):
                f.write(f"![]({r.get('image_path')})\n\n")

        f.write("\n## Bottom Cases\n\n")
        for r in bottom_rows:
            f.write(f"### {r.get('sample_id')}\n\n")
            f.write(f"- prompt: {r.get('prompt_text','')}\n")
            f.write(f"- inferred: {r.get('inferred_text','')}\n")
            if r.get("reconstructed_text"):
                f.write(f"- reconstructed: {r.get('reconstructed_text','')}\n")
            f.write(f"- {sort_key}: {r.get('metrics', {}).get(sort_key)}\n\n")
            if r.get("image_path"):
                f.write(f"![]({r.get('image_path')})\n\n")


# ------------------------------
# Main entrypoint
# ------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_jsonl", type=str, required=True)
    ap.add_argument("--out_report", type=str, required=True)
    ap.add_argument("--top_k", type=int, default=10)
    ap.add_argument("--bottom_k", type=int, default=10)
    args = ap.parse_args()

    rows = read_jsonl(args.input_jsonl)
    write_report(args.out_report, rows, args.top_k, args.bottom_k)

    print(f"[summarize] report={args.out_report}")


if __name__ == "__main__":
    main()
