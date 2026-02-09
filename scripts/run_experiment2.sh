#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Experiment 2 Orchestrator (v2 default)
# - Reads config values
# - Generates prompts/samples
# - Runs T2I + I2T (distributed)
# - Merges and scores results
# ============================================================

ROOT_DIR="/home/ma-user/work/code/bagel_experiment2"
CONFIG_PATH="${ROOT_DIR}/configs/exp2_json_slots_default.yaml"
WORLD_SIZE="${WORLD_SIZE:-4}"

usage() {
  echo "Usage: $0 [--config path] [--nproc N]" >&2
}

# ------------------------------------------------------------
# CLI argument parsing
# ------------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG_PATH="$2"
      shift 2
      ;;
    --nproc)
      WORLD_SIZE="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

export CONFIG_PATH

# ------------------------------------------------------------
# Resolve config values via Python (pipe-delimited for bash)
# ------------------------------------------------------------
IFS='|' read -r OUTPUT_DIR PROMPT_FILE TEMPLATE_FILE NUM_PROMPTS NUM_SEEDS SEED_BASE EMBED_MODEL OFFLINE_EMB PROMPT_MODE VOCAB_DIR BUCKET_CONFIG DISABLE_STYLES <<< "$(python - <<'PY'
import os, yaml
cfg = yaml.safe_load(open(os.environ["CONFIG_PATH"], "r", encoding="utf-8"))
values = [
    cfg.get("output_dir"),
    cfg.get("prompt_file", ""),
    cfg.get("template_file"),
    str(cfg.get("num_prompts")),
    str(cfg.get("num_seeds")),
    str(cfg.get("seed_base")),
    cfg.get("embedding_model_path", ""),
    str(cfg.get("offline_embeddings", False)),
    cfg.get("prompt_mode", "v1"),
    cfg.get("vocab_dir", ""),
    cfg.get("bucket_config", ""),
    str(cfg.get("disable_styles", False)),
]
print("|".join(values))
PY
)"

RUN_ID=$(date +"%Y%m%d_%H%M%S")
RUN_DIR="${OUTPUT_DIR}/exp2_run_${RUN_ID}"
mkdir -p "${RUN_DIR}"

# ------------------------------------------------------------
# Record meta (config snapshot + BAGEL git SHA)
# ------------------------------------------------------------
python - <<PY
import json, os, time, yaml, subprocess
cfg_path = os.environ["CONFIG_PATH"]
cfg = yaml.safe_load(open(cfg_path, "r", encoding="utf-8"))
meta = {
    "time": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
    "config_path": cfg_path,
    "config": cfg,
    "bagel_git": None,
}
try:
    meta["bagel_git"] = subprocess.check_output(["git", "-C", "/home/ma-user/work/code/bagel", "rev-parse", "HEAD"], text=True).strip()
except Exception:
    meta["bagel_git"] = None
out_path = os.path.join("${RUN_DIR}", "meta.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)
print(f"[meta] {out_path}")
PY

# ------------------------------------------------------------
# Prompt generation (v2 or legacy v1)
# ------------------------------------------------------------
STYLE_ARGS=()
if [[ "${DISABLE_STYLES}" == "True" ]]; then
  STYLE_ARGS+=("--disable_styles")
fi

if [[ "${PROMPT_MODE}" == "v2" ]]; then
  python "${ROOT_DIR}/scripts/make_prompts.py" \
    --template_file "${TEMPLATE_FILE}" \
    --bucket_config "${BUCKET_CONFIG}" \
    --vocab_dir "${VOCAB_DIR}" \
    --num_prompts "${NUM_PROMPTS}" \
    --num_seeds "${NUM_SEEDS}" \
    --seed_base "${SEED_BASE}" \
    --out_dir "${RUN_DIR}" \
    "${STYLE_ARGS[@]}"
else
  python "${ROOT_DIR}/legacy/scripts/make_prompts_v1.py" \
    --prompt_file "${PROMPT_FILE}" \
    --template_file "${TEMPLATE_FILE}" \
    --num_prompts "${NUM_PROMPTS}" \
    --num_seeds "${NUM_SEEDS}" \
    --seed_base "${SEED_BASE}" \
    --out_dir "${RUN_DIR}" \
    "${STYLE_ARGS[@]}"
fi

# ------------------------------------------------------------
# T2I generation (distributed)
# ------------------------------------------------------------
torchrun --nproc_per_node=${WORLD_SIZE} "${ROOT_DIR}/scripts/generate_images.py" \
  --config "${CONFIG_PATH}" \
  --samples_jsonl "${RUN_DIR}/samples.jsonl" \
  --output_dir "${RUN_DIR}"

# ------------------------------------------------------------
# Merge T2I outputs into a single JSONL
# ------------------------------------------------------------
python - <<PY
import json, os
run_dir = "${RUN_DIR}"
rows = []
for name in sorted(os.listdir(run_dir)):
    if name.startswith("gen_results_rank") and name.endswith(".jsonl"):
        with open(os.path.join(run_dir, name), "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if s:
                    rows.append(json.loads(s))
all_path = os.path.join(run_dir, "gen_results_all.jsonl")
with open(all_path, "w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"[merge_gen] {all_path} rows={len(rows)}")
PY

# ------------------------------------------------------------
# I2T captioning (distributed)
# ------------------------------------------------------------
torchrun --nproc_per_node=${WORLD_SIZE} "${ROOT_DIR}/scripts/caption_images.py" \
  --config "${CONFIG_PATH}" \
  --gen_results_jsonl "${RUN_DIR}/gen_results_all.jsonl" \
  --output_dir "${RUN_DIR}"

# ------------------------------------------------------------
# Merge, score, and summarize
# ------------------------------------------------------------
python "${ROOT_DIR}/scripts/merge_results.py" \
  --gen_dir "${RUN_DIR}" \
  --caption_dir "${RUN_DIR}" \
  --out_jsonl "${RUN_DIR}/combined_results.jsonl"

EMB_ARGS=()
if [[ "${EMBED_MODEL}" != "" ]]; then
  EMB_ARGS+=("--embedding_model_path" "${EMBED_MODEL}")
fi
if [[ "${OFFLINE_EMB}" == "True" ]]; then
  EMB_ARGS+=("--offline_embeddings")
fi

python "${ROOT_DIR}/scripts/score_texts.py" \
  --input_jsonl "${RUN_DIR}/combined_results.jsonl" \
  --out_jsonl "${RUN_DIR}/scored_results.jsonl" \
  --out_csv "${RUN_DIR}/scored_results.csv" \
  --slot_vocab_dir "${ROOT_DIR}/data/slot_vocab" \
  "${EMB_ARGS[@]}"

python "${ROOT_DIR}/scripts/summarize.py" \
  --input_jsonl "${RUN_DIR}/scored_results.jsonl" \
  --out_report "${RUN_DIR}/report.md"

# ------------------------------------------------------------
# Compact CSV for quick preview
# ------------------------------------------------------------
python - <<PY
import csv, json, os
run_dir = "${RUN_DIR}"
scored_path = os.path.join(run_dir, "scored_results.jsonl")
out_csv = os.path.join(run_dir, "pairs_preview.csv")
rows = []
with open(scored_path, "r", encoding="utf-8") as f:
    for line in f:
        s = line.strip()
        if s:
            rows.append(json.loads(s))

with open(out_csv, "w", encoding="utf-8", newline="") as f:
    w = csv.writer(f)
    w.writerow(["sample_id", "prompt", "inferred_text", "reconstructed_text", "embed_cos", "slot_f1", "token_f1"])
    for r in rows:
        m = r.get("metrics", {})
        w.writerow([
            r.get("sample_id", ""),
            r.get("prompt_text", ""),
            r.get("inferred_text", ""),
            r.get("reconstructed_text", ""),
            m.get("embed_cos", ""),
            m.get("slot_f1", ""),
            m.get("token_f1", ""),
        ])
print(f"[pairs_preview] {out_csv}")
PY

echo "[done] ${RUN_DIR}"
