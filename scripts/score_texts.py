#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Score prompt vs inferred text with embeddings and slot/keyword overlap.

Key ideas:
- If inferred text contains a JSON object, parse it into slots.
- Otherwise, fall back to keyword matching and token metrics.
- Optional embedding cosine similarity for semantic comparison.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple


# Slot keys expected in JSON outputs or slot vocab eval
SLOT_KEYS = [
    "subject",
    "attributes",
    "action",
    "scene",
    "style",
    "lighting",
    "camera",
    "composition",
]


# ------------------------------
# JSONL and CSV helpers
# ------------------------------

def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                rows.append(json.loads(s))
    return rows


def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    keys = sorted({k for r in rows for k in r.keys()})
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


# ------------------------------
# Text similarity helpers
# ------------------------------

def tokenize(text: str) -> List[str]:
    return [t for t in re.split(r"[^a-z0-9]+", text.lower()) if t]


def token_f1(a: str, b: str) -> Tuple[float, float, float]:
    ta = set(tokenize(a))
    tb = set(tokenize(b))
    if not ta or not tb:
        return 0.0, 0.0, 0.0
    inter = ta.intersection(tb)
    p = len(inter) / len(tb)
    r = len(inter) / len(ta)
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f1


def lcs_length(a: List[str], b: List[str]) -> int:
    if not a or not b:
        return 0
    dp = [0] * (len(b) + 1)
    for i in range(1, len(a) + 1):
        prev = 0
        for j in range(1, len(b) + 1):
            tmp = dp[j]
            if a[i - 1] == b[j - 1]:
                dp[j] = prev + 1
            else:
                dp[j] = max(dp[j], dp[j - 1])
            prev = tmp
    return dp[-1]


def rouge_l(a: str, b: str) -> float:
    ta = tokenize(a)
    tb = tokenize(b)
    if not ta or not tb:
        return 0.0
    lcs = lcs_length(ta, tb)
    return lcs / max(1, len(ta))


# ------------------------------
# Slot vocab helpers
# ------------------------------

def load_vocab_list(path: str) -> List[str]:
    items: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                items.append(s)
    return items


def load_synonyms(path: str) -> Dict[str, List[str]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_vocab_map(vocab: List[str], synonyms: Dict[str, List[str]]) -> Dict[str, List[str]]:
    vocab_map: Dict[str, List[str]] = {}
    for v in vocab:
        expanded = [v]
        if v in synonyms:
            expanded.extend(synonyms[v])
        vocab_map[v] = expanded
    return vocab_map


def match_slots(text: str, vocab_map: Dict[str, List[str]]) -> List[str]:
    text_l = text.lower()
    hits: List[str] = []
    for canon, variants in vocab_map.items():
        for v in variants:
            if v.lower() in text_l:
                hits.append(canon)
                break
    return sorted(set(hits))


def slot_f1(a_hits: List[str], b_hits: List[str]) -> Tuple[float, float, float]:
    a_set = set(a_hits)
    b_set = set(b_hits)
    if not a_set or not b_set:
        return 0.0, 0.0, 0.0
    inter = a_set.intersection(b_set)
    p = len(inter) / len(b_set)
    r = len(inter) / len(a_set)
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f1


# ------------------------------
# Embedding helpers
# ------------------------------

def load_embedding_model(path: Optional[str], offline: bool) -> Optional[Tuple[str, Any]]:
    if offline:
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("HF_HOME", os.path.dirname(path) if path else os.getcwd())
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(path) if path else SentenceTransformer("all-MiniLM-L6-v2")
        return ("st", model)
    except Exception:
        pass

    try:
        from transformers import AutoModel, AutoTokenizer
        import torch

        model_id = path if path else "sentence-transformers/all-MiniLM-L6-v2"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModel.from_pretrained(model_id)
        model.eval()
        return ("hf", (tokenizer, model))
    except Exception:
        return None


def cosine(a, b) -> float:
    import numpy as np
    a = np.asarray(a)
    b = np.asarray(b)
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float((a @ b) / denom)


def encode_hf(tokenizer, model, texts: List[str]) -> List[List[float]]:
    import torch
    with torch.no_grad():
        inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        outputs = model(**inputs)
        last_hidden = outputs.last_hidden_state
        mask = inputs["attention_mask"].unsqueeze(-1).float()
        pooled = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
        return pooled.cpu().tolist()


# ------------------------------
# JSON slot parsing and reconstruction
# ------------------------------

def parse_json_slots(text: str) -> Optional[Dict[str, List[str]]]:
    if not text:
        return None
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    raw = text[start:end + 1]
    try:
        obj = json.loads(raw)
    except Exception:
        return None
    if not isinstance(obj, dict):
        return None
    out: Dict[str, List[str]] = {}
    for key in SLOT_KEYS:
        val = obj.get(key, [])
        if isinstance(val, str):
            items = [v.strip() for v in val.split(",") if v.strip()]
        elif isinstance(val, list):
            items = [str(v).strip() for v in val if str(v).strip()]
        else:
            items = []
        out[key] = items
    return out


def reconstruct_prompt(slots: Dict[str, List[str]]) -> str:
    parts: List[str] = []
    for key in SLOT_KEYS:
        vals = slots.get(key, [])
        if vals:
            parts.append(", ".join(vals))
    return ", ".join([p for p in parts if p])


def slot_f1_from_slots(gt: Dict[str, Any], pred: Dict[str, List[str]]) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    f1_values: List[float] = []
    for key in SLOT_KEYS:
        gt_vals = gt.get(key, []) if isinstance(gt, dict) else []
        if isinstance(gt_vals, str):
            gt_list = [v.strip() for v in gt_vals.split(",") if v.strip()]
        elif isinstance(gt_vals, list):
            gt_list = [str(v).strip() for v in gt_vals if str(v).strip()]
        else:
            gt_list = []
        pred_list = pred.get(key, [])
        _, _, f1 = slot_f1(gt_list, pred_list)
        metrics[f"{key}_f1"] = f1
        f1_values.append(f1)
    metrics["slot_f1"] = sum(f1_values) / len(f1_values) if f1_values else 0.0
    return metrics


# ------------------------------
# Main entrypoint
# ------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_jsonl", type=str, required=True)
    ap.add_argument("--out_jsonl", type=str, required=True)
    ap.add_argument("--out_csv", type=str, required=True)
    ap.add_argument("--slot_vocab_dir", type=str, required=True)
    ap.add_argument("--embedding_model_path", type=str, default=None)
    ap.add_argument("--offline_embeddings", action="store_true")
    ap.add_argument("--no_embeddings", action="store_true")
    args = ap.parse_args()

    rows = read_jsonl(args.input_jsonl)

    synonyms = load_synonyms(os.path.join(args.slot_vocab_dir, "synonyms.json"))
    vocab_subject = build_vocab_map(load_vocab_list(os.path.join(args.slot_vocab_dir, "subject.txt")), synonyms)
    vocab_attribute = build_vocab_map(load_vocab_list(os.path.join(args.slot_vocab_dir, "attribute.txt")), synonyms)
    vocab_scene = build_vocab_map(load_vocab_list(os.path.join(args.slot_vocab_dir, "scene.txt")), synonyms)
    vocab_style = build_vocab_map(load_vocab_list(os.path.join(args.slot_vocab_dir, "style.txt")), synonyms)

    embedder: Optional[Tuple[str, Any]] = None
    if not args.no_embeddings:
        embedder = load_embedding_model(args.embedding_model_path, args.offline_embeddings)

    output_rows: List[Dict[str, Any]] = []
    for r in rows:
        prompt = r.get("prompt_text", "") or ""
        inferred = r.get("inferred_text", "") or ""
        prompt_slots = r.get("prompt_slots")

        pred_slots = parse_json_slots(inferred)
        reconstructed = None
        inferred_for_score = inferred
        slot_metrics: Dict[str, float] = {}

        # If JSON slots exist, score on reconstructed text instead of raw output.
        if pred_slots:
            reconstructed = reconstruct_prompt(pred_slots)
            inferred_for_score = reconstructed

        if prompt_slots and pred_slots:
            slot_metrics = slot_f1_from_slots(prompt_slots, pred_slots)

        token_p, token_r, token_f1_val = token_f1(prompt, inferred_for_score)
        rouge_l_val = rouge_l(prompt, inferred_for_score)

        if prompt_slots and pred_slots:
            subj_f1_val = slot_metrics.get("subject_f1", 0.0)
            attr_f1_val = slot_metrics.get("attributes_f1", 0.0)
            scene_f1_val = slot_metrics.get("scene_f1", 0.0)
            style_f1_val = slot_metrics.get("style_f1", 0.0)
        else:
            subj_p = match_slots(prompt, vocab_subject)
            subj_i = match_slots(inferred, vocab_subject)
            attr_p = match_slots(prompt, vocab_attribute)
            attr_i = match_slots(inferred, vocab_attribute)
            scene_p = match_slots(prompt, vocab_scene)
            scene_i = match_slots(inferred, vocab_scene)
            style_p = match_slots(prompt, vocab_style)
            style_i = match_slots(inferred, vocab_style)

            _, _, subj_f1_val = slot_f1(subj_p, subj_i)
            _, _, attr_f1_val = slot_f1(attr_p, attr_i)
            _, _, scene_f1_val = slot_f1(scene_p, scene_i)
            _, _, style_f1_val = slot_f1(style_p, style_i)

        embed_cos = None
        if embedder is not None:
            try:
                backend, obj = embedder
                if backend == "st":
                    emb = obj.encode([prompt, inferred_for_score], normalize_embeddings=True)
                    embed_cos = cosine(emb[0], emb[1])
                else:
                    tokenizer, model = obj
                    emb = encode_hf(tokenizer, model, [prompt, inferred_for_score])
                    embed_cos = cosine(emb[0], emb[1])
            except Exception:
                embed_cos = None

        metrics = {
            "embed_cos": embed_cos,
            "token_precision": token_p,
            "token_recall": token_r,
            "token_f1": token_f1_val,
            "rouge_l": rouge_l_val,
            "subj_f1": subj_f1_val,
            "attr_f1": attr_f1_val,
            "scene_f1": scene_f1_val,
            "style_f1": style_f1_val,
        }

        if slot_metrics:
            metrics.update(slot_metrics)

        out = dict(r)
        if pred_slots:
            out["pred_slots"] = pred_slots
        if reconstructed is not None:
            out["reconstructed_text"] = reconstructed
        out["metrics"] = metrics
        output_rows.append(out)

    write_jsonl(args.out_jsonl, output_rows)

    # Flatten metrics into columns for CSV.
    flat_rows: List[Dict[str, Any]] = []
    for r in output_rows:
        row = {k: v for k, v in r.items() if k != "metrics"}
        for mk, mv in r.get("metrics", {}).items():
            row[mk] = mv
        flat_rows.append(row)
    write_csv(args.out_csv, flat_rows)

    print(f"[score_texts] scored={len(output_rows)} out={args.out_jsonl}")


if __name__ == "__main__":
    main()
