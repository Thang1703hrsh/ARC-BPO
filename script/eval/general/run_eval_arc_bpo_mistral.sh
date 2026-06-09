#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash script/eval/general/run_eval_arc_bpo_mistral.sh <model-path-or-hf-repo>
# or:
#   MODEL_NAME=<model-path-or-hf-repo> bash script/eval/general/run_eval_arc_bpo_mistral.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_NAME="${1:-${MODEL_NAME:-}}"

if [[ -z "${MODEL_NAME}" ]]; then
  echo "Error: provide a trained Mistral checkpoint path or HF repo id." >&2
  echo "Example: bash $0 output/<run>/LATEST" >&2
  exit 1
fi

export MODEL_NAME
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export TP_SIZE="${TP_SIZE:-2}"
export MAX_LEN="${MAX_LEN:-4096}"
export BATCH_SIZE="${BATCH_SIZE:-auto:4}"

bash "${SCRIPT_DIR}/runall.sh"
