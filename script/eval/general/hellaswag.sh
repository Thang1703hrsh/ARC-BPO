#!/bin/bash
set -e

# Use exported variables from runall.sh if available, otherwise use defaults
: ${WANDB_API_KEY:=""}
: ${CUDA_VISIBLE_DEVICES:="0,1"}
: ${LM_EVAL_LOGLEVEL:=DEBUG}
: ${VLLM_LOGLEVEL:=INFO}

export WANDB_API_KEY
export CUDA_VISIBLE_DEVICES
export LM_EVAL_LOGLEVEL
export VLLM_LOGLEVEL

# Define variables - use exported values from runall.sh or defaults
MODEL_NAME="${MODEL_NAME:-}"
TASKS="hellaswag"
TP_SIZE="${TP_SIZE:-2}"
DTYPE="${DTYPE:-auto}"
GPU_UTIL="${GPU_UTIL:-0.9}"
BATCH_SIZE="${BATCH_SIZE:-auto:4}"
MAX_LEN="${MAX_LEN:-4096}"                   

# Construct model arguments
MODEL_ARGS="pretrained=${MODEL_NAME},tensor_parallel_size=${TP_SIZE},dtype=${DTYPE},gpu_memory_utilization=${GPU_UTIL},max_model_len=${MAX_LEN}"
#MODEL_ARGS="pretrained=${MODEL_NAME},tensor_parallel_size=${TP_SIZE},dtype=${DTYPE},gpu_memory_utilization=${GPU_UTIL}"
# Execute lm_eval
lm_eval --model vllm \
        --model_args "${MODEL_ARGS}" \
        --tasks "${TASKS}" \
        --num_fewshot=10 \
        --batch_size "${BATCH_SIZE}" \
        --output_path "output/${MODEL_NAME}/${TASKS}" \
        --wandb_args entity=Token_BPO,project=Token_BPO,name=${MODEL_NAME}_${TASKS},job_type=eval \
        --log_samples \
        2>&1 | tee /tmp/lm_eval_debug.log
