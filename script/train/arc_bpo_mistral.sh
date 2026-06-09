#!/usr/bin/env bash
set -euo pipefail

# ARC-BPO on Mistral-7B-v0.1 from experiments.md.

export WANDB_API_KEY="${WANDB_API_KEY:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

MODEL_CONFIG="${MODEL_CONFIG:-mistral_7b}"
DATASETS_RAW="${DATASETS_RAW:-HuggingFaceH4/ultrafeedback_binarized}"
TRAIN_SPLIT="${TRAIN_SPLIT:-train_prefs}"
TEST_SPLIT="${TEST_SPLIT:-test_prefs}"

BATCH_SIZE="${BATCH_SIZE:-32}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
LR="${LR:-5e-7}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-10.0}"
OPTIMIZER="${OPTIMIZER:-RMSprop}"
SCHEDULER="${SCHEDULER:-cosine}"
WARMUP_RATIO="${WARMUP_RATIO:-0.05}"
MAX_LENGTH="${MAX_LENGTH:-2048}"
N_EPOCHS="${N_EPOCHS:-1}"
TRAINER="${TRAINER:-FSDPTrainer}"
ACTIVATION_CHECKPOINTING="${ACTIVATION_CHECKPOINTING:-true}"
DO_FIRST_EVAL="${DO_FIRST_EVAL:-true}"
MIN_LOG_INTERVAL_SECS="${MIN_LOG_INTERVAL_SECS:-1.0}"

USE_LORA="${USE_LORA:-true}"
USE_BASELINE_HEAD="false"

BETA="${BETA:-0.1}"
DELTA_STAR="${DELTA_STAR:-2.0}"
ARC_T="${ARC_T:-2.0}"
KAPPA="${KAPPA:-2.0}"
SBA_LAMBDA="${SBA_LAMBDA:-1.0}"
SBA_SCALE="${SBA_SCALE:-4.0}"
EXP_CLIP="${EXP_CLIP:-30.0}"
USE_ADVANTAGE_SHAPE="${USE_ADVANTAGE_SHAPE:-false}"
FALLBACK_TO_UNIFORM_SHAPE="${FALLBACK_TO_UNIFORM_SHAPE:-true}"

python3 train.py \
  model="${MODEL_CONFIG}" \
  model.use_lora="${USE_LORA}" \
  model.use_baseline_head="${USE_BASELINE_HEAD}" \
  loss=arc_bpo \
  loss.beta="${BETA}" \
  loss.delta_star="${DELTA_STAR}" \
  loss.T="${ARC_T}" \
  loss.kappa="${KAPPA}" \
  loss.sba_lambda="${SBA_LAMBDA}" \
  loss.sba_scale="${SBA_SCALE}" \
  loss.exp_clip="${EXP_CLIP}" \
  loss.use_advantage_shape="${USE_ADVANTAGE_SHAPE}" \
  loss.fallback_to_uniform_shape="${FALLBACK_TO_UNIFORM_SHAPE}" \
  datasets="${DATASETS_RAW}" \
  dataset_train_split="${TRAIN_SPLIT}" \
  dataset_test_split="${TEST_SPLIT}" \
  trainer="${TRAINER}" \
  batch_size="${BATCH_SIZE}" \
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
