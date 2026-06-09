#!/bin/bash

# Run evaluations for all models in model_list.txt
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_LIST="$SCRIPT_DIR/model_list.txt"

# Check if model list exists
if [ ! -f "$MODEL_LIST" ]; then
    echo "Error: model_list.txt not found at $MODEL_LIST"
    exit 1
fi

# Read models from file and count them
mapfile -t MODELS < <(grep -v '^[[:space:]]*$' "$MODEL_LIST")
TOTAL_MODELS=${#MODELS[@]}

echo "================================================"
echo "Running evaluations for $TOTAL_MODELS models"
echo "================================================"

# Loop through each model
for i in "${!MODELS[@]}"; do
    MODEL="${MODELS[$i]}"
    MODEL_NUM=$((i + 1))

    echo ""
    echo "================================================"
    echo "Model $MODEL_NUM/$TOTAL_MODELS: $MODEL"
    echo "================================================"

    # Export the model name and run the evaluation script
    export MODEL_NAME="$MODEL"
    bash "$SCRIPT_DIR/runall.sh"

    echo "Completed evaluations for: $MODEL"
done

echo ""
echo "================================================"
echo "All models evaluated successfully!"
echo "================================================"
