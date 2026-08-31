#!/usr/bin/env bash
set -euo pipefail

# ARC-BPO on Llama-3-8B from experiments.md.

export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export WANDB_MODE=disabled

# --- GPU selection (override on the new server) ---------------------------
# Pass the GPUs to use as a comma-separated list, e.g. GPU_IDS=0,1,2,3.
# Defaults to a single GPU (0) so it runs anywhere without assuming 4 cards.
GPU_IDS="${GPU_IDS:-0}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${GPU_IDS}}"

# Count visible GPUs from CUDA_VISIBLE_DEVICES.
NUM_GPUS="$(awk -F',' '{n=0; for(i=1;i<=NF;i++) if($i!="") n++; print n}' <<<"${CUDA_VISIBLE_DEVICES}")"
if [[ "${NUM_GPUS}" -lt 1 ]]; then NUM_GPUS=1; fi

# Absolute output dir anchored to the repo root (script is at script/train/),
# so checkpoints always land in <repo>/output regardless of where you run this.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/output}"
LOG_DIR="${LOG_DIR:-${OUTPUT_DIR}/logs}"

# Run from the repo root so relative paths inside train.py/Hydra always resolve
# correctly even when this script is launched from another directory by SLURM.
cd "${REPO_ROOT}"

MODEL_CONFIG="${MODEL_CONFIG:-llama_8b}"
MODEL_REVISION="${MODEL_REVISION:-}"
DATASETS_RAW="${DATASETS_RAW:-princeton-nlp/llama3-ultrafeedback-armorm}"
DATASET_REVISION="${DATASET_REVISION:-}"
TRAIN_SPLIT="${TRAIN_SPLIT:-train}"
TEST_SPLIT="${TEST_SPLIT:-test}"
SEED="${SEED:-0}"
EXP_NAME="${EXP_NAME:-}"
RUN_DIR="${RUN_DIR:-}"

# --- Batch sizing (auto, scales with GPU count) ---------------------------
# Per-GPU microbatch is the real memory knob: per_gpu = BATCH_SIZE/GRAD_ACCUM/NUM_GPUS.
# Defaults: GRAD_ACCUM=4 and per-GPU microbatch=2 -> BATCH_SIZE = 8*NUM_GPUS.
# Lower PER_GPU_MICROBATCH if you hit OOM; raise it on 80GB cards to go faster.
GRAD_ACCUM="${GRAD_ACCUM:-4}"
PER_GPU_MICROBATCH="${PER_GPU_MICROBATCH:-2}"
BATCH_SIZE="${BATCH_SIZE:-$(( PER_GPU_MICROBATCH * GRAD_ACCUM * NUM_GPUS ))}"

# Hard constraint: BATCH_SIZE must be divisible by GRAD_ACCUM*NUM_GPUS, else the
# trainer silently drops the remainder (data loss) or builds empty microbatches.
DIVISOR=$(( GRAD_ACCUM * NUM_GPUS ))
if (( BATCH_SIZE % DIVISOR != 0 )); then
  echo "ERROR: BATCH_SIZE=${BATCH_SIZE} is not divisible by GRAD_ACCUM*NUM_GPUS=${DIVISOR}." >&2
  echo "       Set BATCH_SIZE to a multiple of ${DIVISOR} (e.g. $(( (BATCH_SIZE/DIVISOR + 1) * DIVISOR )))." >&2
  exit 1
fi
echo "[GPUS] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} (NUM_GPUS=${NUM_GPUS})"
echo "[BATCH] global=${BATCH_SIZE} grad_accum=${GRAD_ACCUM} per_gpu_microbatch=$(( BATCH_SIZE / DIVISOR ))"
LR="${LR:-5e-7}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-10.0}"
OPTIMIZER="${OPTIMIZER:-RMSprop}"
SCHEDULER="${SCHEDULER:-cosine}"
WARMUP_RATIO="${WARMUP_RATIO:-0.05}"
MAX_LENGTH="${MAX_LENGTH:-2048}"
N_EPOCHS="${N_EPOCHS:-1}"
N_EXAMPLES="${N_EXAMPLES:-}"
TRAINER="${TRAINER:-FSDPTrainer}"
ACTIVATION_CHECKPOINTING="${ACTIVATION_CHECKPOINTING:-true}"
DO_FIRST_EVAL="${DO_FIRST_EVAL:-true}"
N_EVAL_EXAMPLES="${N_EVAL_EXAMPLES:-64}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-16}"
SAVE_CHECKPOINT="${SAVE_CHECKPOINT:-true}"
SAVE_EVERY_EXAMPLES="${SAVE_EVERY_EXAMPLES:-5000}"
MIN_LOG_INTERVAL_SECS="${MIN_LOG_INTERVAL_SECS:-1.0}"

USE_LORA="${USE_LORA:-true}"
USE_BASELINE_HEAD="false"

# Optional HuggingFace upload after a successful train. Leave HF_REPO_ID empty
# to keep the current local-only behavior.
HF_REPO_ID="${HF_REPO_ID:-}"
HF_PRIVATE="${HF_PRIVATE:-true}"
HF_UPLOAD_ADAPTER_ONLY="${HF_UPLOAD_ADAPTER_ONLY:-true}"

# --- ARC-BPO loss hyperparameters (Llama-3-8B / Llama3-UltraFeedback-ArmoRM) ---
# Base 8B SFT checkpoint. The data loader maps Princeton's aligned
# all_generated_responses/all_rm_scores fields to detached per-chunk proxies.
# Delta remains a fixed finite margin so allocation is the controlled factor.
BETA="${BETA:-0.1}"
DELTA_STAR="${DELTA_STAR:-2.5}"
ARC_T="${ARC_T:-2.0}"
KAPPA="${KAPPA:-2.0}"
SBA_LAMBDA="${SBA_LAMBDA:-1.0}"
SBA_SCALE="${SBA_SCALE:-4.0}"
EXP_CLIP="${EXP_CLIP:-30.0}"
# Uniform remains the public default. Controlled advantage runs should set
# USE_ADVANTAGE_SHAPE=true and FALLBACK_TO_UNIFORM_SHAPE=false.
USE_ADVANTAGE_SHAPE="${USE_ADVANTAGE_SHAPE:-false}"
FALLBACK_TO_UNIFORM_SHAPE="${FALLBACK_TO_UNIFORM_SHAPE:-true}"
WINSORIZE_ADVANTAGES="${WINSORIZE_ADVANTAGES:-true}"

# Deterministic chunker floor/ceiling (matches experiments.md: min 4 / max 64).
MIN_TOKENS_PER_CHUNK="${MIN_TOKENS_PER_CHUNK:-4}"
MAX_TOKENS_PER_CHUNK="${MAX_TOKENS_PER_CHUNK:-64}"

TRAIN_LIMIT_ARGS=(n_epochs="${N_EPOCHS}")
if [[ -n "${N_EXAMPLES}" ]]; then
  TRAIN_LIMIT_ARGS=(n_examples="${N_EXAMPLES}" n_epochs=null)
fi

mkdir -p "${LOG_DIR}"
RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
TRAIN_LOG="${TRAIN_LOG:-${LOG_DIR}/arc_bpo_llama_${RUN_STAMP}.log}"

CMD=(
  python3 -u train.py
  model="${MODEL_CONFIG}"
  model.use_lora="${USE_LORA}"
  model.use_baseline_head="${USE_BASELINE_HEAD}"
  loss=arc_bpo
  loss.beta="${BETA}"
  loss.delta_star="${DELTA_STAR}"
  loss.T="${ARC_T}"
  loss.kappa="${KAPPA}"
  loss.sba_lambda="${SBA_LAMBDA}"
  loss.sba_scale="${SBA_SCALE}"
  loss.exp_clip="${EXP_CLIP}"
  loss.use_advantage_shape="${USE_ADVANTAGE_SHAPE}"
  loss.fallback_to_uniform_shape="${FALLBACK_TO_UNIFORM_SHAPE}"
  loss.winsorize_advantages="${WINSORIZE_ADVANTAGES}"
  loss.min_tokens_per_chunk="${MIN_TOKENS_PER_CHUNK}"
  loss.max_tokens_per_chunk="${MAX_TOKENS_PER_CHUNK}"
  output_dir="${OUTPUT_DIR}"
  datasets="${DATASETS_RAW}"
  dataset_train_split="${TRAIN_SPLIT}"
  dataset_test_split="${TEST_SPLIT}"
  trainer="${TRAINER}"
  batch_size="${BATCH_SIZE}"
  gradient_accumulation_steps="${GRAD_ACCUM}"
  lr="${LR}"
  weight_decay="${WEIGHT_DECAY}"
  max_grad_norm="${MAX_GRAD_NORM}"
  optimizer="${OPTIMIZER}"
  scheduler="${SCHEDULER}"
  warmup_ratio="${WARMUP_RATIO}"
  minimum_log_interval_secs="${MIN_LOG_INTERVAL_SECS}"
  max_length="${MAX_LENGTH}"
  "${TRAIN_LIMIT_ARGS[@]}"
  eval_batch_size="${EVAL_BATCH_SIZE}"
  n_eval_examples="${N_EVAL_EXAMPLES}"
  save_checkpoint="${SAVE_CHECKPOINT}"
  save_every_examples="${SAVE_EVERY_EXAMPLES}"
  do_first_eval="${DO_FIRST_EVAL}"
  activation_checkpointing="${ACTIVATION_CHECKPOINTING}"
  seed="${SEED}"
  wandb.enabled=false
)

if [[ -n "${EXP_NAME}" ]]; then
  CMD+=(exp_name="${EXP_NAME}")
fi
if [[ -n "${RUN_DIR}" ]]; then
  CMD+=(local_run_dir="${RUN_DIR}")
fi
if [[ -n "${MODEL_REVISION}" ]]; then
  CMD+=(model.revision="${MODEL_REVISION}")
fi
if [[ -n "${DATASET_REVISION}" ]]; then
  CMD+=(dataset_revision="${DATASET_REVISION}")
fi

echo "[LOG] ${TRAIN_LOG}"
printf '[RUN]'
printf ' %q' "${CMD[@]}"
printf '\n'
"${CMD[@]}" 2>&1 | tee "${TRAIN_LOG}"

if [[ -n "${HF_REPO_ID}" ]]; then
  export HF_REPO_ID HF_PRIVATE HF_UPLOAD_ADAPTER_ONLY OUTPUT_DIR USE_LORA
  python3 - <<'PY'
import os

from huggingface_hub import HfApi, upload_folder


def as_bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


output_dir = os.environ["OUTPUT_DIR"]
repo_id = os.environ["HF_REPO_ID"]
hf_private = as_bool(os.environ.get("HF_PRIVATE", "true"))
use_lora = as_bool(os.environ.get("USE_LORA", "true"))
upload_adapter_only = as_bool(os.environ.get("HF_UPLOAD_ADAPTER_ONLY", "true"))

latest_dirs = []
for root, dirs, _ in os.walk(output_dir):
    if "LATEST" in dirs:
        latest_path = os.path.join(root, "LATEST")
        latest_dirs.append((os.path.getmtime(latest_path), latest_path))

if not latest_dirs:
    raise RuntimeError(f"No LATEST checkpoint found under {output_dir}")

latest_path = max(latest_dirs)[1]
upload_path = latest_path
if use_lora and upload_adapter_only:
    upload_path = os.path.join(latest_path, "adapter")
    adapter_config = os.path.join(upload_path, "adapter_config.json")
    adapter_model = os.path.join(upload_path, "adapter_model.safetensors")
    if not os.path.isfile(adapter_config):
        raise RuntimeError(f"No LoRA adapter_config.json found at {upload_path}")
    if not os.path.isfile(adapter_model) or os.path.getsize(adapter_model) <= 1024:
        raise RuntimeError(
            f"LoRA adapter_model.safetensors is missing or too small: {adapter_model}"
        )

print(f"[HF UPLOAD] repo={repo_id}")
print(f"[HF UPLOAD] folder={upload_path}")
api = HfApi()
api.create_repo(repo_id, private=hf_private, exist_ok=True)
upload_folder(
    repo_id=repo_id,
    folder_path=upload_path,
    commit_message="Upload ARC-BPO LLaMA checkpoint",
)
print(f"[HF UPLOAD] Done: https://huggingface.co/{repo_id}")
PY
fi
