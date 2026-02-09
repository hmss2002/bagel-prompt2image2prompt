#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/ma-user/work/code/prompt2image2prompt-pipeline"
CONFIG_PATH="${ROOT_DIR}/configs/p2i2p_sdxl-turbo__blip.yaml"
WORLD_SIZE="${WORLD_SIZE:-4}"

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
      echo "Usage: $0 [--config path] [--nproc N]" >&2
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

"${ROOT_DIR}/scripts/run_p2i2p_sdxl-turbo__i2t.sh" --config "${CONFIG_PATH}" --nproc "${WORLD_SIZE}"
