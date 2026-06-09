#!/bin/bash

# Run all evaluation scripts sequentially
set -e  # Exit if any script fails

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ============================================
# CENTRALIZED CONFIGURATION
# ============================================
# Change these variables to apply to all evaluation scripts
export CUDA_VISIBLE_DEVICES="0,1"
export MODEL_NAME="${MODEL_NAME:-RLHFlow/LLaMA3-SFT-v2}"  # Model to evaluate (uses env var if set, otherwise default)
export TP_SIZE=2                                    # Tensor parallelism size
export DTYPE="auto"                                 # Data type
export GPU_UTIL=0.9                                 # GPU memory utilization
export BATCH_SIZE="auto:4"                          # Batch size
export MAX_LEN=4096                                 # Max sequence length
export WANDB_API_KEY=""
export HF_TOKEN=""

# Optional: uncomment to run wandb in offline mode
# export WANDB_MODE="offline"

# Log levels
export LM_EVAL_LOGLEVEL=DEBUG
export VLLM_LOGLEVEL=INFO
# ============================================

# Function to cleanup GPU processes
cleanup_gpu() {
    echo "Cleaning up GPU processes..."
    pkill -9 -f "vllm" || true
    sleep 3  # Wait for GPU memory to be released
    echo "GPU cleanup complete."
}

echo "Starting evaluation pipeline..."
echo "================================"

echo "1/6 Running ARC evaluation..."
bash "$SCRIPT_DIR/arc.sh"
cleanup_gpu

echo "2/6 Running GSM8K evaluation..."
bash "$SCRIPT_DIR/gsm8k.sh"
cleanup_gpu

echo "3/6 Running HellaSwag evaluation..."
bash "$SCRIPT_DIR/hellaswag.sh"
cleanup_gpu

echo "4/6 Running MMLU evaluation..."
bash "$SCRIPT_DIR/mmlu.sh"
cleanup_gpu

echo "5/6 Running QA evaluation..."
bash "$SCRIPT_DIR/qa.sh"
cleanup_gpu

echo "6/6 Running Winogrande evaluation..."
bash "$SCRIPT_DIR/wino.sh"

echo "================================"
echo "All evaluations completed!"
