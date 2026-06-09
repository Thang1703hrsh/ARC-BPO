#!/usr/bin/env bash
set -euo pipefail

# Convenience wrapper. Use the per-model scripts directly when running jobs
# separately on a cluster scheduler.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_FILTER="${MODEL_FILTER:-all}"

run_if_selected() {
  local key="$1"
  local script="$2"
  if [[ "${MODEL_FILTER}" == "all" || "${MODEL_FILTER}" == "${key}" ]]; then
    bash "${SCRIPT_DIR}/${script}"
  fi
}

run_if_selected "mistral" "arc_bpo_mistral.sh"
run_if_selected "llama" "arc_bpo_llama.sh"
run_if_selected "qwen" "arc_bpo_qwen.sh"
