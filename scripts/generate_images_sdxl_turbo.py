#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate images with SDXL Turbo (Text-to-Image).

This script mirrors generate_images.py output schema so downstream merge/score
remain unchanged. It reads samples.jsonl, shards by rank, and writes
per-rank gen_results_rank*.jsonl.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any, Dict, List

import torch


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
    return items[rank::world_size]


def append_jsonl(path: str, obj: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def get_rank_info() -> tuple[int, int, int]:
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    return rank, local_rank, world_size


def try_enable_npu() -> bool:
    try:
        import torch_npu  # noqa: F401
    except Exception:
        return False
    return hasattr(torch, "npu") and torch.npu.is_available()


def resolve_device(requested: str) -> str:
    if requested == "cuda":
        if torch.cuda.is_available():
            return "cuda"
        raise RuntimeError("Requested cuda but torch.cuda.is_available() is false.")
    if requested == "npu":
        if try_enable_npu():
            return "npu"
        raise RuntimeError("Requested npu but torch_npu is not available.")
    return "cpu"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--samples_jsonl", type=str, required=True)
    ap.add_argument("--output_dir", type=str, required=True)
    ap.add_argument("--model_path", type=str, default=None)
    ap.add_argument("--dtype", type=str, default=None)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--image_w", type=int, default=None)
    ap.add_argument("--image_h", type=int, default=None)
    ap.add_argument("--device", type=str, default=None, choices=["cpu", "cuda", "npu"])
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    model_path = args.model_path or cfg.get("t2i_model_path") or cfg["model_path"]
    dtype_str = args.dtype or cfg.get("dtype", "bf16")
    steps = args.steps if args.steps is not None else cfg.get("steps", 8)
    image_w = args.image_w if args.image_w is not None else cfg.get("image_w", 1024)
    image_h = args.image_h if args.image_h is not None else cfg.get("image_h", 1024)
    requested_device = args.device or cfg.get("t2i_device", "cpu")
    device = resolve_device(requested_device)

    rank, local_rank, world_size = get_rank_info()
    if device == "cuda":
        torch.cuda.set_device(local_rank)
    elif device == "npu":
        torch.npu.set_device(local_rank)

    try:
        from diffusers import StableDiffusionXLPipeline
    except Exception as e:
        raise RuntimeError("diffusers is required for SDXL Turbo.") from e

    torch_dtype = torch.float16 if dtype_str in {"fp16", "float16"} else torch.bfloat16

    pipe = StableDiffusionXLPipeline.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        variant="fp16" if dtype_str in {"fp16", "float16"} else None,
        use_safetensors=True,
        local_files_only=True,
    )
    pipe = pipe.to(device)

    samples = read_jsonl(args.samples_jsonl)
    my_samples = shard(samples, rank, world_size)

    os.makedirs(args.output_dir, exist_ok=True)
    image_dir = os.path.join(args.output_dir, "images")
    os.makedirs(image_dir, exist_ok=True)
    out_jsonl = os.path.join(args.output_dir, f"gen_results_rank{rank}.jsonl")

    for row in my_samples:
        sample_id = row["sample_id"]
        prompt_text = row["prompt_text"]
        seed = int(row["seed"])
        generator = torch.Generator(device="cpu").manual_seed(seed)

        t0 = time.time()
        record: Dict[str, Any] = {
            "sample_id": sample_id,
            "prompt_id": row["prompt_id"],
            "prompt_text": prompt_text,
            "prompt_slots": row.get("gt_slots") or row.get("slots"),
            "seed": seed,
            "rank": rank,
            "params": {
                "steps": steps,
                "image_w": image_w,
                "image_h": image_h,
                "dtype": dtype_str,
                "device": device,
            },
            "status": "ok",
            "error": None,
            "image_path": None,
        }

        try:
            out = pipe(
                prompt_text,
                num_inference_steps=steps,
                height=int(image_h),
                width=int(image_w),
                generator=generator,
            )
            img = out.images[0] if out and out.images else None
            if img is None:
                raise RuntimeError("No image returned by SDXL Turbo.")
            image_path = os.path.join(image_dir, f"{sample_id}.png")
            img.save(image_path)
            record["image_path"] = image_path
        except Exception as e:
            record["status"] = "error"
            record["error"] = str(e)

        record["seconds"] = time.time() - t0
        append_jsonl(out_jsonl, record)
        print(f"[rank{rank}] gen {sample_id} status={record['status']}", flush=True)


if __name__ == "__main__":
    main()
