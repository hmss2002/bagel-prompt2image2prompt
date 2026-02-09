#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bucketed prompt generator with weighted vocab and slot tracking.

High-level flow:
1) Load templates, buckets, and vocab.
2) Sample prompts per bucket/complexity.
3) Deduplicate by Jaccard overlap.
4) Write prompts_expanded.jsonl and samples.jsonl.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class PromptRecord:
    # A single prompt with its generation metadata and slot values.
    prompt_id: str
    prompt_text: str
    bucket: str
    complexity: str
    template_id: str
    slots: Dict[str, Any]
    seed_base: int
    created_at: str
    version: str


# ------------------------------
# Basic file helpers
# ------------------------------

def read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_weighted_line(line: str) -> Tuple[str, float]:
    # Parse "value|weight" lines. Missing/invalid weight defaults to 1.0.
    s = line.strip()
    if "|" in s:
        val, w = s.rsplit("|", 1)
        try:
            return val.strip(), float(w)
        except ValueError:
            return s, 1.0
    return s, 1.0


def load_weighted_list(path: str) -> List[Tuple[str, float]]:
    items: List[Tuple[str, float]] = []
    if not os.path.isfile(path):
        return items
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            items.append(parse_weighted_line(s))
    return items


def pick_weighted(rng: random.Random, items: List[Tuple[str, float]]) -> str:
    # Weighted random pick; returns empty string if list is empty.
    if not items:
        return ""
    values = [v for v, _ in items]
    weights = [w for _, w in items]
    return rng.choices(values, weights=weights, k=1)[0]


# ------------------------------
# Config loaders
# ------------------------------

def load_templates(path: str) -> List[Dict[str, Any]]:
    data = read_json(path)
    return data.get("templates", [])


def load_bucket_config(path: str) -> Dict[str, Any]:
    return read_json(path)


# ------------------------------
# Dedup and tokenization helpers
# ------------------------------

def tokenize(text: str) -> List[str]:
    return [t for t in re.split(r"[^a-z0-9]+", text.lower()) if t]


def jaccard(a: List[str], b: List[str]) -> float:
    sa = set(a)
    sb = set(b)
    if not sa or not sb:
        return 0.0
    inter = sa.intersection(sb)
    union = sa.union(sb)
    return len(inter) / max(1, len(union))


# ------------------------------
# Bucket count calculators
# ------------------------------

def calc_bucket_counts(total: int, buckets: List[Dict[str, Any]]) -> Dict[str, int]:
    # Allocate total prompts across buckets by weight.
    weights = [b.get("weight", 1.0) for b in buckets]
    weight_sum = sum(weights) if weights else 1.0
    raw = [total * (w / weight_sum) for w in weights]
    counts = [int(x) for x in raw]
    remainder = total - sum(counts)
    if remainder > 0:
        order = sorted(range(len(raw)), key=lambda i: raw[i] - counts[i], reverse=True)
        for i in range(remainder):
            counts[order[i % len(order)]] += 1
    return {buckets[i]["name"]: counts[i] for i in range(len(buckets))}


def calc_complexity_counts(total: int, ratios: Dict[str, float]) -> Dict[str, int]:
    # Allocate per-bucket counts by simple/medium/complex ratio.
    keys = ["simple", "medium", "complex"]
    weights = [ratios.get(k, 0.0) for k in keys]
    weight_sum = sum(weights) if sum(weights) > 0 else 1.0
    raw = [total * (w / weight_sum) for w in weights]
    counts = [int(x) for x in raw]
    remainder = total - sum(counts)
    if remainder > 0:
        order = sorted(range(len(raw)), key=lambda i: raw[i] - counts[i], reverse=True)
        for i in range(remainder):
            counts[order[i % len(order)]] += 1
    return {keys[i]: counts[i] for i in range(len(keys))}


# ------------------------------
# Vocab loading (with optional style disable)
# ------------------------------

def load_vocab(vocab_dir: str, disable_styles: bool) -> Dict[str, List[Tuple[str, float]]]:
    return {
        "person": load_weighted_list(os.path.join(vocab_dir, "subjects", "person.txt")),
        "animal": load_weighted_list(os.path.join(vocab_dir, "subjects", "animal.txt")),
        "vehicle": load_weighted_list(os.path.join(vocab_dir, "subjects", "vehicle.txt")),
        "architecture": load_weighted_list(os.path.join(vocab_dir, "subjects", "architecture.txt")),
        "indoor_object": load_weighted_list(os.path.join(vocab_dir, "subjects", "indoor_object.txt")),
        "food": load_weighted_list(os.path.join(vocab_dir, "subjects", "food.txt")),
        "nature": load_weighted_list(os.path.join(vocab_dir, "subjects", "nature.txt")),
        "abstract": load_weighted_list(os.path.join(vocab_dir, "subjects", "abstract.txt")),
        "color": load_weighted_list(os.path.join(vocab_dir, "attributes", "color.txt")),
        "material": load_weighted_list(os.path.join(vocab_dir, "attributes", "material.txt")),
        "texture": load_weighted_list(os.path.join(vocab_dir, "attributes", "texture.txt")),
        "mood": load_weighted_list(os.path.join(vocab_dir, "attributes", "mood.txt")),
        "clothing": load_weighted_list(os.path.join(vocab_dir, "attributes", "clothing.txt")),
        "age_gender": load_weighted_list(os.path.join(vocab_dir, "attributes", "age_gender.txt")),
        "human_action": load_weighted_list(os.path.join(vocab_dir, "actions", "human_action.txt")),
        "animal_action": load_weighted_list(os.path.join(vocab_dir, "actions", "animal_action.txt")),
        "object_state": load_weighted_list(os.path.join(vocab_dir, "actions", "object_state.txt")),
        "relation": load_weighted_list(os.path.join(vocab_dir, "actions", "relation.txt")),
        "scene_indoor": load_weighted_list(os.path.join(vocab_dir, "scenes", "indoor.txt")),
        "scene_city": load_weighted_list(os.path.join(vocab_dir, "scenes", "outdoor_city.txt")),
        "scene_nature": load_weighted_list(os.path.join(vocab_dir, "scenes", "outdoor_nature.txt")),
        "scene_time": load_weighted_list(os.path.join(vocab_dir, "scenes", "weather_time.txt")),
        "style_photo": ([] if disable_styles else load_weighted_list(os.path.join(vocab_dir, "styles", "photo.txt"))),
        "style_painting": ([] if disable_styles else load_weighted_list(os.path.join(vocab_dir, "styles", "painting.txt"))),
        "style_illustration": ([] if disable_styles else load_weighted_list(os.path.join(vocab_dir, "styles", "illustration.txt"))),
        "style_anime": ([] if disable_styles else load_weighted_list(os.path.join(vocab_dir, "styles", "anime.txt"))),
        "style_3d": ([] if disable_styles else load_weighted_list(os.path.join(vocab_dir, "styles", "3d_render.txt"))),
        "style_abstract": ([] if disable_styles else load_weighted_list(os.path.join(vocab_dir, "styles", "abstract.txt"))),
        "style_long_tail": ([] if disable_styles else load_weighted_list(os.path.join(vocab_dir, "styles", "long_tail.txt"))),
        "lighting": load_weighted_list(os.path.join(vocab_dir, "lighting", "lighting.txt")),
    }


# ------------------------------
# Slot pickers
# ------------------------------

def pick_scene(rng: random.Random, vocab: Dict[str, List[Tuple[str, float]]], bucket: str) -> str:
    if bucket in {"indoor_object", "food"}:
        scene = pick_weighted(rng, vocab["scene_indoor"])
        return scene
    if bucket in {"architecture", "vehicle", "person", "animal"}:
        scene = pick_weighted(rng, vocab["scene_city"])
    else:
        scene = pick_weighted(rng, vocab["scene_nature"])
    time = pick_weighted(rng, vocab["scene_time"])
    if scene and time:
        return f"{scene}, {time}"
    return scene or time


def pick_style(rng: random.Random, vocab: Dict[str, List[Tuple[str, float]]], bucket: str) -> str:
    if bucket == "abstract":
        return pick_weighted(rng, vocab["style_abstract"])
    if bucket == "rare":
        base = pick_weighted(rng, vocab["style_long_tail"])
        extra = pick_weighted(rng, vocab["style_illustration"]) or pick_weighted(rng, vocab["style_photo"])
        return f"{base}, {extra}" if extra else base
    pool: List[Tuple[str, float]] = []
    pool += vocab["style_photo"]
    pool += vocab["style_painting"]
    pool += vocab["style_illustration"]
    pool += vocab["style_3d"]
    if bucket == "person":
        pool += vocab["style_anime"]
    return pick_weighted(rng, pool)


def pick_attributes(rng: random.Random, vocab: Dict[str, List[Tuple[str, float]]], bucket: str) -> str:
    parts: List[str] = []
    if vocab["color"] and rng.random() < 0.7:
        parts.append(pick_weighted(rng, vocab["color"]))
    if vocab["texture"] and rng.random() < 0.6:
        parts.append(pick_weighted(rng, vocab["texture"]))
    if bucket == "person" and vocab["clothing"] and rng.random() < 0.6:
        parts.append(pick_weighted(rng, vocab["clothing"]))
    if vocab["mood"] and rng.random() < 0.4:
        parts.append(pick_weighted(rng, vocab["mood"]))
    if bucket in {"indoor_object", "architecture", "vehicle"} and vocab["material"] and rng.random() < 0.5:
        parts.append(pick_weighted(rng, vocab["material"]))
    if not parts:
        parts.append(pick_weighted(rng, vocab["texture"]) or pick_weighted(rng, vocab["color"]))
    return ", ".join([p for p in parts if p])


def pick_action(rng: random.Random, vocab: Dict[str, List[Tuple[str, float]]], bucket: str) -> str:
    if bucket == "person":
        return pick_weighted(rng, vocab["human_action"])
    if bucket == "animal":
        return pick_weighted(rng, vocab["animal_action"])
    return pick_weighted(rng, vocab["object_state"])


def pick_subject(rng: random.Random, vocab: Dict[str, List[Tuple[str, float]]], bucket: str) -> str:
    items = vocab.get(bucket, [])
    if not items:
        return pick_weighted(rng, vocab.get("person", []))
    return pick_weighted(rng, items)


def build_slots(rng: random.Random, vocab: Dict[str, List[Tuple[str, float]]], bucket: str) -> Dict[str, Any]:
    # Build a full slot dict for a single prompt instance.
    slots: Dict[str, Any] = {}
    slots["subject"] = pick_subject(rng, vocab, bucket)
    slots["subject2"] = pick_subject(rng, vocab, bucket)
    slots["relation"] = pick_weighted(rng, vocab["relation"])
    slots["action"] = pick_action(rng, vocab, bucket)
    slots["scene"] = pick_scene(rng, vocab, bucket)
    slots["style"] = pick_style(rng, vocab, bucket)
    slots["lighting"] = pick_weighted(rng, vocab["lighting"])
    slots["attributes"] = pick_attributes(rng, vocab, bucket)
    slots["color"] = pick_weighted(rng, vocab["color"])
    return slots


# ------------------------------
# Prompt materialization
# ------------------------------

def cleanup_prompt(text: str) -> str:
    # Normalize spacing and strip extra commas.
    text = re.sub(r"\s+,", ",", text)
    text = re.sub(r",\s+", ", ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip(" ,")


def materialize_prompt(rng: random.Random, template: Dict[str, Any], slots: Dict[str, Any]) -> str:
    values = {k: slots.get(k, "") for k in template.get("slots", [])}
    text = template.get("template", "").format(**values)
    return cleanup_prompt(text)


# ------------------------------
# Main entrypoint
# ------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--template_file", type=str, required=True)
    ap.add_argument("--bucket_config", type=str, required=True)
    ap.add_argument("--vocab_dir", type=str, required=True)
    ap.add_argument("--disable_styles", action="store_true")
    ap.add_argument("--num_prompts", type=int, required=True)
    ap.add_argument("--num_seeds", type=int, required=True)
    ap.add_argument("--seed_base", type=int, default=1234)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--jaccard_dedup", type=float, default=0.85)
    args = ap.parse_args()

    rng = random.Random(args.seed_base)
    templates = load_templates(args.template_file)
    bucket_cfg = load_bucket_config(args.bucket_config)
    vocab = load_vocab(args.vocab_dir, args.disable_styles)
    version = bucket_cfg.get("version", "v2")

    # Index templates by bucket and complexity for fast sampling.
    templates_by_bucket: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    for t in templates:
        bucket = t.get("bucket", "any")
        complexity = t.get("complexity", "medium")
        templates_by_bucket.setdefault(bucket, {}).setdefault(complexity, []).append(t)

    buckets = bucket_cfg.get("buckets", [])
    bucket_counts = calc_bucket_counts(args.num_prompts, buckets)

    records: List[PromptRecord] = []
    seen = set()
    token_sets: List[List[str]] = []
    created_at = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())

    def is_near_dup(text: str) -> bool:
        toks = tokenize(text)
        for prev in token_sets[-200:]:
            if jaccard(toks, prev) >= args.jaccard_dedup:
                return True
        return False

    # Generate prompts bucket-by-bucket.
    for bucket in bucket_counts:
        total_bucket = bucket_counts[bucket]
        ratios = next((b.get("complexity", {}) for b in buckets if b.get("name") == bucket), {})
        comp_counts = calc_complexity_counts(total_bucket, ratios)
        for complexity, count in comp_counts.items():
            if count <= 0:
                continue
            candidates = templates_by_bucket.get(bucket, {}).get(complexity, [])
            if not candidates:
                candidates = templates_by_bucket.get(bucket, {}).get("medium", [])
            attempts = 0
            while count > 0 and attempts < args.num_prompts * 20:
                t = rng.choice(candidates) if candidates else None
                if not t:
                    break
                slots = build_slots(rng, vocab, bucket)
                text = materialize_prompt(rng, t, slots)
                if not text or text in seen or is_near_dup(text):
                    attempts += 1
                    continue
                seen.add(text)
                token_sets.append(tokenize(text))
                records.append(PromptRecord(
                    prompt_id="",
                    prompt_text=text,
                    bucket=bucket,
                    complexity=complexity,
                    template_id=t.get("id", ""),
                    slots=slots,
                    seed_base=args.seed_base,
                    created_at=created_at,
                    version=version,
                ))
                count -= 1
                attempts += 1
            if count > 0:
                print(f"[warn] bucket={bucket} complexity={complexity} shortfall={count}")

    # Assign prompt IDs.
    for i, rec in enumerate(records):
        rec.prompt_id = f"p{i:05d}"

    prompts_path = os.path.join(args.out_dir, "prompts_expanded.jsonl")
    samples_path = os.path.join(args.out_dir, "samples.jsonl")
    os.makedirs(args.out_dir, exist_ok=True)

    # Write prompts (one per record).
    with open(prompts_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps({
                "prompt_id": rec.prompt_id,
                "prompt_text": rec.prompt_text,
                "bucket": rec.bucket,
                "complexity": rec.complexity,
                "template_id": rec.template_id,
                "slots": rec.slots,
                "seed_base": rec.seed_base,
                "created_at": rec.created_at,
                "version": rec.version,
            }, ensure_ascii=False) + "\n")

    # Write samples (one per prompt per seed).
    with open(samples_path, "w", encoding="utf-8") as f:
        for rec in records:
            for s in range(args.num_seeds):
                seed = rec.seed_base + s
                sample_id = f"{rec.prompt_id}_seed{seed}"
                f.write(json.dumps({
                    "sample_id": sample_id,
                    "prompt_id": rec.prompt_id,
                    "prompt_text": rec.prompt_text,
                    "seed": seed,
                    "seed_base": rec.seed_base,
                    "bucket": rec.bucket,
                    "complexity": rec.complexity,
                    "gt_slots": rec.slots,
                    "created_at": rec.created_at,
                    "version": rec.version,
                }, ensure_ascii=False) + "\n")

    print(f"[make_prompts] prompts={len(records)} samples={len(records) * args.num_seeds}")
    print(f"[make_prompts] prompts_path={prompts_path}")
    print(f"[make_prompts] samples_path={samples_path}")

    # Quick stats to validate distributions.
    bucket_stats: Dict[str, int] = {}
    comp_stats: Dict[str, int] = {}
    lengths: List[int] = []
    for rec in records:
        bucket_stats[rec.bucket] = bucket_stats.get(rec.bucket, 0) + 1
        comp_stats[rec.complexity] = comp_stats.get(rec.complexity, 0) + 1
        lengths.append(len(rec.prompt_text.split()))

    avg_len = sum(lengths) / len(lengths) if lengths else 0.0
    print(f"[make_prompts] avg_len={avg_len:.2f}")
    print(f"[make_prompts] bucket_counts={bucket_stats}")
    print(f"[make_prompts] complexity_counts={comp_stats}")


if __name__ == "__main__":
    main()
