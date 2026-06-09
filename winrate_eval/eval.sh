#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

export OPENAI_CLIENT_CONFIG_PATH="$PROJECT_ROOT/winrate_eval/client_configs/openai_configs.yaml"

alpaca_eval evaluate_from_model \
  --model_configs "$PROJECT_ROOT/winrate_eval/model_configs/tisdpo/config.yaml" \
  --evaluation_dataset "$PROJECT_ROOT/processed_data/hh_reference.json" \
  --annotators_config "$PROJECT_ROOT/winrate_eval/annotators_configs/mistral/config.yaml" \
  --output_path "$PROJECT_ROOT/winrate_results/mistral/tisdpo_hh" \
  --base_dir "$PROJECT_ROOT/winrate_eval/model_configs/tisdpo" \
  # --reference_outputs "$PROJECT_ROOT/processed_data/tldr_reference.json" \
