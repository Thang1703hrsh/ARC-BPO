#!/usr/bin/env bash
set -euo pipefail

# SFT baseline from TBPO.tex:
# Mistral 7B SFT checkpoint + UltraFeedback Binarized chosen responses.

export WANDB_API_KEY="${WANDB_API_KEY:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

MODEL_CONFIG="${MODEL_CONFIG:-mistral_7b}"
DATASETS_RAW="${DATASETS_RAW:-HuggingFaceH4/ultrafeedback_binarized}"
TRAIN_SPLIT="${TRAIN_SPLIT:-train_prefs}"
TEST_SPLIT="${TEST_SPLIT:-test_prefs}"

# Full fine-tuning matches the paper setting better; set USE_LORA=true for a
# lower-memory smoke run.
USE_LORA="${USE_LORA:-false}"
USE_BASELINE_HEAD="false"

BATCH_SIZE="${BATCH_SIZE:-8}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
LR="${LR:-2e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-10.0}"
OPTIMIZER="${OPTIMIZER:-AdamW}"
SCHEDULER="${SCHEDULER:-cosine}"
WARMUP_RATIO="${WARMUP_RATIO:-0.05}"
MIN_LOG_INTERVAL_SECS="${MIN_LOG_INTERVAL_SECS:-1.0}"

MAX_LENGTH="${MAX_LENGTH:-2048}"
N_EPOCHS="${N_EPOCHS:-1}"
DO_FIRST_EVAL="${DO_FIRST_EVAL:-true}"
ACTIVATION_CHECKPOINTING="${ACTIVATION_CHECKPOINTING:-true}"
TRAINER="${TRAINER:-FSDPTrainer}"

python3 train.py \
  model="${MODEL_CONFIG}" \
  model.use_lora="${USE_LORA}" \
  model.use_baseline_head="${USE_BASELINE_HEAD}" \
  loss=sft \
  datasets="${DATASETS_RAW}" \
  dataset_train_split="${TRAIN_SPLIT}" \
  dataset_test_split="${TEST_SPLIT}" \
  trainer="${TRAINER}" \
  batch_size="${BATCH_SIZE}" \
  eval_batch_size="${EVAL_BATCH_SIZE}" \
  gradient_accumulation_steps="${GRAD_ACCUM}" \
  lr="${LR}" \
  weight_decay="${WEIGHT_DECAY}" \
  max_grad_norm="${MAX_GRAD_NORM}" \
  optimizer="${OPTIMIZER}" \
  scheduler="${SCHEDULER}" \
  warmup_ratio="${WARMUP_RATIO}" \
  minimum_log_interval_secs="${MIN_LOG_INTERVAL_SECS}" \
  max_length="${MAX_LENGTH}" \
  n_epochs="${N_EPOCHS}" \
  do_first_eval="${DO_FIRST_EVAL}" \
  activation_checkpointing="${ACTIVATION_CHECKPOINTING}"
