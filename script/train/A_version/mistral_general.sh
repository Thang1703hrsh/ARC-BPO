#!/usr/bin/env bash
set -euo pipefail

export WANDB_API_KEY=""
export CUDA_VISIBLE_DEVICES="0,1,2,3"

# Mistral/Q-version training launcher for this repo.

### OPTIONS

## MODEL
MODEL_CONFIG="mistral_7b"
USE_BASELINE_HEAD="false"

## LOSS
LOSS_NAME="A_tbpo"
BREGMAN_NAME="sba"
BREGMAN_LAM="0.0"
BREGMAN_S="4.0"

## GENERAL CONFIG
BATCH_SIZE="32"
GRAD_ACCUM="2"
DATASETS_RAW="HuggingFaceH4/ultrafeedback_binarized"
TRAIN_SPLIT="train_prefs"
TEST_SPLIT="test_prefs"
LR="5e-7"
WEIGHT_DECAY="0.0"
MAX_GRAD_NORM="10.0"
OPTIMIZER="RMSprop"
SCHEDULER="cosine"
WARMUP_RATIO="0.05"
MIN_LOG_INTERVAL_SECS="1.0"

MAX_LENGTH="2048"
N_EPOCHS="1"
DO_FIRST_EVAL="true"
ACTIVATION_CHECKPOINTING="true"

python3 train.py \
  model=${MODEL_CONFIG} \
  model.use_baseline_head=${USE_BASELINE_HEAD} \
  loss=${LOSS_NAME} \
  loss.bregman_loss.name=${BREGMAN_NAME} \
  loss.bregman_loss.lam=${BREGMAN_LAM} \
  loss.bregman_loss.s=${BREGMAN_S} \
  datasets=${DATASETS_RAW} \
  dataset_train_split=${TRAIN_SPLIT} \
  dataset_test_split=${TEST_SPLIT} \
  batch_size=${BATCH_SIZE} \
  gradient_accumulation_steps=${GRAD_ACCUM} \
  lr=${LR} \
  weight_decay=${WEIGHT_DECAY} \
  max_grad_norm=${MAX_GRAD_NORM} \
  optimizer=${OPTIMIZER} \
  scheduler=${SCHEDULER} \
  warmup_ratio=${WARMUP_RATIO} \
  minimum_log_interval_secs=${MIN_LOG_INTERVAL_SECS} \
  max_length=${MAX_LENGTH} \
  n_epochs=${N_EPOCHS} \
  do_first_eval=${DO_FIRST_EVAL} \
  activation_checkpointing=${ACTIVATION_CHECKPOINTING} \
