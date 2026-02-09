#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Caption images using BLIP (Image-to-Text).

This script mirrors caption_images.py output schema so merge/score work
without changes. It reads gen_results_all.jsonl, shards by rank, and writes
caption_results_rank*.jsonl.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any, Dict, List

from PIL import Image
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


def normalize_prompt(text: str) -> str:
    return " ".join(text.strip().lower().split())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--gen_results_jsonl", type=str, required=True)
    ap.add_argument("--output_dir", type=str, required=True)
    ap.add_argument("--model_path", type=str, default=None)
    ap.add_argument("--dtype", type=str, default=None)
    ap.add_argument("--caption_prompt", type=str, default=None)
    ap.add_argument("--device", type=str, default=None, choices=["cpu", "cuda", "npu"])
    ap.add_argument("--max_new_tokens", type=int, default=None)
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    model_path = args.model_path or cfg.get("i2t_model_path")
    dtype_str = args.dtype or cfg.get("dtype", "bf16")
    caption_prompt = args.caption_prompt or cfg.get(
        "caption_prompt",
        "",
    )
    max_new_tokens = args.max_new_tokens if args.max_new_tokens is not None else cfg.get("max_new_tokens", 128)
    requested_device = args.device or cfg.get("i2t_device", "cpu")
    device = resolve_device(requested_device)

    rank, local_rank, world_size = get_rank_info()
    if device == "cuda":
        torch.cuda.set_device(local_rank)
    elif device == "npu":
        torch.npu.set_device(local_rank)

    if not model_path:
        raise RuntimeError("i2t_model_path is required for BLIP captioning.")

    # Offline loading support
    if cfg.get("offline_embeddings") or cfg.get("offline_models"):
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("HF_HOME", "/home/ma-user/work/models")

    try:
        from transformers import BlipProcessor, BlipForConditionalGeneration
    except Exception as e:
        raise RuntimeError("transformers is required for BLIP captioning.") from e

    processor = BlipProcessor.from_pretrained(model_path, local_files_only=True)
    model = BlipForConditionalGeneration.from_pretrained(model_path, local_files_only=True)
    model = model.to(device)

    gen_rows = read_jsonl(args.gen_results_jsonl)
    valid_rows = [r for r in gen_rows if r.get("status") == "ok" and r.get("image_path")]
    my_rows = shard(valid_rows, rank, world_size)

    os.makedirs(args.output_dir, exist_ok=True)
    out_jsonl = os.path.join(args.output_dir, f"caption_results_rank{rank}.jsonl")

    norm_prompt = normalize_prompt(caption_prompt) if caption_prompt else ""
    use_text_prompt = bool(norm_prompt) and norm_prompt not in {"none", "null"}

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
            if use_text_prompt:
                inputs = processor(images=img, text=caption_prompt, return_tensors="pt")
            else:
                inputs = processor(images=img, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            out_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
            inferred = processor.decode(out_ids[0], skip_special_tokens=True).strip()

            if use_text_prompt and normalize_prompt(inferred) == norm_prompt:
                inputs = processor(images=img, return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}
                out_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
                inferred = processor.decode(out_ids[0], skip_special_tokens=True).strip()

            if not inferred:
                raise RuntimeError("No text returned by BLIP.")
            record["inferred_text"] = inferred
        except Exception as e:
            record["status"] = "error"
            record["error"] = str(e)

        record["seconds"] = time.time() - t0
        append_jsonl(out_jsonl, record)
        print(f"[rank{rank}] caption {sample_id} status={record['status']}", flush=True)


if __name__ == "__main__":
    main()
