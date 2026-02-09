#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Caption images (Image-to-Text) using frozen generation outputs.

This script:
- Reads gen_results_all.jsonl
- Shards by rank
- Runs I2T captioning
- Writes per-rank caption_results_rank*.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List

from PIL import Image
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
    ap.add_argument("--gen_results_jsonl", type=str, required=True)
    ap.add_argument("--output_dir", type=str, required=True)
    ap.add_argument("--model_path", type=str, default=None)
    ap.add_argument("--dtype", type=str, default=None)
    ap.add_argument("--caption_prompt", type=str, default=None)
    ap.add_argument("--max_new_tokens", type=int, default=None)
    ap.add_argument("--do_sample", action="store_true")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    model_path = args.model_path or cfg["model_path"]
    dtype = args.dtype or cfg.get("dtype", "bf16")
    caption_prompt = args.caption_prompt or cfg.get("caption_prompt", "Describe the image in detail.")
    max_new_tokens = args.max_new_tokens if args.max_new_tokens is not None else cfg.get("max_new_tokens", 128)

    bagel_root = "/home/ma-user/work/code/bagel"
    if bagel_root not in sys.path:
        sys.path.append(bagel_root)
    import inference_ascend as ia

    # Initialize distributed runtime.
    rank, local_rank, world_size = ia.get_rank_info()
    ia.maybe_init_distributed(world_size)
    device, torch_dtype = ia.init_device(local_rank, dtype)

    gen_rows = read_jsonl(args.gen_results_jsonl)
    valid_rows = [r for r in gen_rows if r.get("status") == "ok" and r.get("image_path")]
    my_rows = shard(valid_rows, rank, world_size)

    os.makedirs(args.output_dir, exist_ok=True)
    out_jsonl = os.path.join(args.output_dir, f"caption_results_rank{rank}.jsonl")

    inferencer = ia.build_inferencer(model_path, device, torch_dtype)

    # Inference hyperparameters for I2T.
    inference_hyper: Dict[str, Any] = dict(
        max_think_token_n=max_new_tokens,
        do_sample=args.do_sample,
        cfg_text_scale=cfg.get("cfg_text_scale", 4.0),
        cfg_img_scale=cfg.get("cfg_img_scale", 1.0),
        cfg_interval=ia.parse_cfg_interval(cfg.get("cfg_interval", "0.4,1.0")),
        timestep_shift=cfg.get("timestep_shift", 3.0),
        num_timesteps=cfg.get("steps", 50),
        cfg_renorm_min=cfg.get("cfg_renorm_min", 0.0),
        cfg_renorm_type=cfg.get("cfg_renorm_type", "global"),
    )

    # Caption each image in the shard.
    for row in my_rows:
        sample_id = row["sample_id"]
        t0 = time.time()
        record: Dict[str, Any] = {
            "sample_id": sample_id,
            "prompt_id": row.get("prompt_id"),
            "prompt_text": row.get("prompt_text"),
            "seed": row.get("seed"),
            "rank": rank,
            "image_path": row.get("image_path"),
            "caption_prompt": caption_prompt,
            "inferred_text": None,
            "status": "ok",
            "error": None,
        }

        try:
            img = Image.open(row["image_path"]).convert("RGB")
            out = inferencer(image=img, text=caption_prompt, understanding_output=True, **inference_hyper)
            inferred = out.get("text", None)
            if not inferred:
                raise RuntimeError("No text returned by inferencer.")
            record["inferred_text"] = inferred
        except Exception as e:
            record["status"] = "error"
            record["error"] = str(e)

        record["seconds"] = time.time() - t0
        append_jsonl(out_jsonl, record)
        print(f"[rank{rank}] caption {sample_id} status={record['status']}", flush=True)

    ia.maybe_destroy_distributed()


if __name__ == "__main__":
    main()
