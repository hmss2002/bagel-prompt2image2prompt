#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate images from frozen samples list (Text-to-Image).

This script:
- Loads the experiment config.
- Shards samples across ranks.
- Runs BAGEL inference to produce images.
- Writes per-rank JSONL outputs with metadata.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List

import torch


# ------------------------------
# Config and JSONL utilities
# ------------------------------

def load_yaml(path: str) -> Dict[str, Any]:
    try:
        import yaml
    except Exception as e:
        raise RuntimeError("PyYAML is required to read config files.") from e
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                rows.append(json.loads(s))
    return rows


def shard(items: List[Dict[str, Any]], rank: int, world_size: int) -> List[Dict[str, Any]]:
    # Deterministic sharding: each rank gets a strided slice.
    return items[rank::world_size]


# ------------------------------
# Reproducibility helpers
# ------------------------------

def set_seed(seed: int) -> None:
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if hasattr(torch, "npu") and torch.npu.is_available():
        torch.npu.manual_seed(seed)


# ------------------------------
# Output helper
# ------------------------------

def append_jsonl(path: str, obj: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


# ------------------------------
# Main entrypoint
# ------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--samples_jsonl", type=str, required=True)
    ap.add_argument("--output_dir", type=str, required=True)
    ap.add_argument("--model_path", type=str, default=None)
    ap.add_argument("--dtype", type=str, default=None)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--cfg_text_scale", type=float, default=None)
    ap.add_argument("--cfg_img_scale", type=float, default=None)
    ap.add_argument("--cfg_interval", type=str, default=None)
    ap.add_argument("--timestep_shift", type=float, default=None)
    ap.add_argument("--cfg_renorm_type", type=str, default=None)
    ap.add_argument("--cfg_renorm_min", type=float, default=None)
    ap.add_argument("--image_w", type=int, default=None)
    ap.add_argument("--image_h", type=int, default=None)
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    model_path = args.model_path or cfg["model_path"]
    dtype = args.dtype or cfg.get("dtype", "bf16")

    # Import inference_ascend from Bagel repo.
    bagel_root = "/home/ma-user/work/code/bagel"
    if bagel_root not in sys.path:
        sys.path.append(bagel_root)
    import inference_ascend as ia

    # Initialize distributed runtime.
    rank, local_rank, world_size = ia.get_rank_info()
    ia.maybe_init_distributed(world_size)
    device, torch_dtype = ia.init_device(local_rank, dtype)

    samples = read_jsonl(args.samples_jsonl)
    my_samples = shard(samples, rank, world_size)

    os.makedirs(args.output_dir, exist_ok=True)
    image_dir = os.path.join(args.output_dir, "images")
    os.makedirs(image_dir, exist_ok=True)
    out_jsonl = os.path.join(args.output_dir, f"gen_results_rank{rank}.jsonl")

    inferencer = ia.build_inferencer(model_path, device, torch_dtype)

    # Collect inference hyperparameters from config/CLI overrides.
    cfg_interval = args.cfg_interval or cfg.get("cfg_interval", "0.4,1.0")
    cfg_interval_parsed = ia.parse_cfg_interval(cfg_interval)

    inference_hyper: Dict[str, Any] = dict(
        cfg_text_scale=args.cfg_text_scale if args.cfg_text_scale is not None else cfg.get("cfg_text_scale", 4.0),
        cfg_img_scale=args.cfg_img_scale if args.cfg_img_scale is not None else cfg.get("cfg_img_scale", 1.0),
        cfg_interval=cfg_interval_parsed,
        timestep_shift=args.timestep_shift if args.timestep_shift is not None else cfg.get("timestep_shift", 3.0),
        num_timesteps=args.steps if args.steps is not None else cfg.get("steps", 50),
        cfg_renorm_min=args.cfg_renorm_min if args.cfg_renorm_min is not None else cfg.get("cfg_renorm_min", 0.0),
        cfg_renorm_type=args.cfg_renorm_type if args.cfg_renorm_type is not None else cfg.get("cfg_renorm_type", "global"),
    )

    image_w = args.image_w if args.image_w is not None else cfg.get("image_w", 1024)
    image_h = args.image_h if args.image_h is not None else cfg.get("image_h", 576)
    inference_hyper["image_shapes"] = (int(image_w), int(image_h))

    # Generate images for this rank's shard.
    for row in my_samples:
        sample_id = row["sample_id"]
        prompt_text = row["prompt_text"]
        seed = int(row["seed"])
        set_seed(seed)

        t0 = time.time()
        record: Dict[str, Any] = {
            "sample_id": sample_id,
            "prompt_id": row["prompt_id"],
            "prompt_text": prompt_text,
            "prompt_slots": row.get("gt_slots") or row.get("slots"),
            "seed": seed,
            "rank": rank,
            "params": {
                "steps": inference_hyper["num_timesteps"],
                "cfg_text_scale": inference_hyper["cfg_text_scale"],
                "cfg_img_scale": inference_hyper["cfg_img_scale"],
                "cfg_interval": cfg_interval_parsed,
                "timestep_shift": inference_hyper["timestep_shift"],
                "image_w": image_w,
                "image_h": image_h,
                "dtype": dtype,
            },
            "status": "ok",
            "error": None,
            "image_path": None,
        }

        try:
            out = inferencer(text=prompt_text, **inference_hyper)
            img = out.get("image", None)
            if img is None:
                raise RuntimeError("No image returned by inferencer.")
            image_path = os.path.join(image_dir, f"{sample_id}.png")
            img.save(image_path)
            record["image_path"] = image_path
        except Exception as e:
            record["status"] = "error"
            record["error"] = str(e)

        record["seconds"] = time.time() - t0
        append_jsonl(out_jsonl, record)
        print(f"[rank{rank}] gen {sample_id} status={record['status']}", flush=True)

    ia.maybe_destroy_distributed()


if __name__ == "__main__":
    main()
