#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Materialize prompts and samples for Experiment 2 with dedup and coverage control.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class PromptRecord:
    prompt_id: str
    prompt_text: str
    template_id: Optional[str]
    slots: Optional[Dict[str, str]]
    seed_base: int
    created_at: str


def load_templates(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("templates", [])


def load_baseline_prompts(path: str) -> List[str]:
    prompts: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                prompts.append(s)
    return prompts


def choose_template_prompt(rng: random.Random, template: Dict[str, Any]) -> Dict[str, Any]:
    slots = template.get("slots", {})
    filled: Dict[str, str] = {}
    for k, vals in slots.items():
        if not vals:
            continue
        filled[k] = rng.choice(vals)
    prompt_text = template.get("template", "").format(**filled)
    return {"text": prompt_text, "slots": filled, "template_id": template.get("id")}


def materialize_prompts(
    baseline_prompts: List[str],
    templates: List[Dict[str, Any]],
    num_prompts: int,
    seed_base: int,
) -> List[PromptRecord]:
    rng = random.Random(seed_base)
    records: List[PromptRecord] = []
    seen = set()
    created_at = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())

    def add_prompt(text: str, template_id: Optional[str], slots: Optional[Dict[str, str]]) -> None:
        t = text.strip()
        if not t or t in seen:
            return
        seen.add(t)
        records.append(PromptRecord(
            prompt_id="",
            prompt_text=t,
            template_id=template_id,
            slots=slots,
            seed_base=seed_base,
            created_at=created_at,
        ))

    # Add baseline prompts first
    for p in baseline_prompts:
        add_prompt(p, "baseline", None)
        if len(records) >= num_prompts:
            break

    if len(records) < num_prompts and templates:
        # Coverage pass: ensure each slot value appears at least once
        for t in templates:
            slots = t.get("slots", {})
            for slot_key, values in slots.items():
                for v in values:
                    filled: Dict[str, str] = {}
                    for k, vals in slots.items():
                        filled[k] = v if k == slot_key else rng.choice(vals)
                    text = t.get("template", "").format(**filled)
                    add_prompt(text, t.get("id"), filled)
                    if len(records) >= num_prompts:
                        break
                if len(records) >= num_prompts:
                    break
            if len(records) >= num_prompts:
                break

    # Random fill with dedup
    max_attempts = max(100, num_prompts * 20)
    attempts = 0
    while len(records) < num_prompts and attempts < max_attempts and templates:
        t = rng.choice(templates)
        res = choose_template_prompt(rng, t)
        add_prompt(res["text"], res["template_id"], res["slots"])
        attempts += 1

    if len(records) < num_prompts:
        print(f"[warn] requested={num_prompts} unique={len(records)} (insufficient unique prompts)")

    # Assign prompt_id
    for i, rec in enumerate(records):
        rec.prompt_id = f"p{i:05d}"

    return records


def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt_file", type=str, required=True)
    ap.add_argument("--template_file", type=str, required=True)
    ap.add_argument("--num_prompts", type=int, required=True)
    ap.add_argument("--num_seeds", type=int, required=True)
    ap.add_argument("--seed_base", type=int, default=1234)
    ap.add_argument("--out_dir", type=str, required=True)
    args = ap.parse_args()

    baseline_prompts = load_baseline_prompts(args.prompt_file)
    templates = load_templates(args.template_file)

    if not baseline_prompts and not templates:
        raise ValueError("No baseline prompts or templates available.")

    prompt_records = materialize_prompts(
        baseline_prompts=baseline_prompts,
        templates=templates,
        num_prompts=args.num_prompts,
        seed_base=args.seed_base,
    )

    prompts_path = os.path.join(args.out_dir, "prompts_expanded.jsonl")
    samples_path = os.path.join(args.out_dir, "samples.jsonl")

    prompt_rows: List[Dict[str, Any]] = []
    sample_rows: List[Dict[str, Any]] = []

    for rec in prompt_records:
        prompt_rows.append({
            "prompt_id": rec.prompt_id,
            "prompt_text": rec.prompt_text,
            "template_id": rec.template_id,
            "slots": rec.slots,
            "seed_base": rec.seed_base,
            "created_at": rec.created_at,
        })

        for s in range(args.num_seeds):
            seed = rec.seed_base + s
            sample_id = f"{rec.prompt_id}_seed{seed}"
            sample_rows.append({
                "sample_id": sample_id,
                "prompt_id": rec.prompt_id,
                "prompt_text": rec.prompt_text,
                "seed": seed,
                "seed_base": rec.seed_base,
                "created_at": rec.created_at,
            })

    write_jsonl(prompts_path, prompt_rows)
    write_jsonl(samples_path, sample_rows)

    print(f"[make_prompts] prompts={len(prompt_rows)} samples={len(sample_rows)}")
    print(f"[make_prompts] prompts_path={prompts_path}")
    print(f"[make_prompts] samples_path={samples_path}")


if __name__ == "__main__":
    main()
