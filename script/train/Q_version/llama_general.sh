#!/usr/bin/env bash
set -euo pipefail

export WANDB_API_KEY=""
export CUDA_VISIBLE_DEVICES="0,1,2,3"

# Mistral/Q-version training launcher for this repo.

### OPTIONS

## MODEL
MODEL_CONFIG="llama_8b"
USE_BASELINE_HEAD="true"
BASELINE_MLP_DIM="1024"
BASELINE_DROPOUT="0.0"
BASELINE_L="-10.0"
BASELINE_U="10.0"

## LOSS
LOSS_NAME="Q_tbpo"
BREGMAN_NAME="sba"
BREGMAN_LAM="0.0"
BREGMAN_S="4.0"

## GENERAL CONFIG
BATCH_SIZE="32"
GRAD_ACCUM="2"
DATASETS_RAW="princeton-nlp/llama3-ultrafeedback-armorm"
TRAIN_SPLIT="train"
TEST_SPLIT="test"
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
  model.baseline_mlp_dim=${BASELINE_MLP_DIM} \
  model.baseline_dropout=${BASELINE_DROPOUT} \
  model.baseline_l=${BASELINE_L} \
  model.baseline_u=${BASELINE_U} \
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

