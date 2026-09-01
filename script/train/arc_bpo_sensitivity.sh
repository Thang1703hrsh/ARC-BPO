#!/usr/bin/env bash
set -euo pipefail

# Bash launcher for the Llama-3 / 10k / global-bs64 sensitivity grid on 2xA100.
# The Python files remain the implementation engine; this wrapper provides one
# stable, environment-variable-driven command for server use.

usage() {
  cat <<'EOF'
Usage:
  bash script/train/arc_bpo_sensitivity.sh MODE

MODE:
  audit       Generate and audit the 14 run configs without training.
  smoke       Train the first setting only.
  full        Train all 14 settings sequentially (default).
  evaluate    Evaluate trained/checkpoint_exists rows on 2 GPUs.
  summarize   Build the sensitivity tables from the manifest.

Common overrides:
  GPU_IDS=0,1
  OUTPUT_ROOT=outputs/sensitivity/llama3-10k-bs64-2xa100-ga8
  HF_REPO_ID=ducthang1703/llama3-arc-bpo-sensitivity-10k-bs64-2xa100-ga8
  HF_PRIVATE=false
  HF_UPLOAD_ADAPTER_ONLY=true

Authentication (choose one):
  1. Run `hf auth login` before this launcher.
  2. Export HF_TOKEN in the current shell.
  3. Set HF_TOKEN_FILE to a chmod-600 file containing only the token.

Never commit a live Hugging Face token to this script.
EOF
}

is_true() {
  case "${1,,}" in
    1|true|yes|y|on) return 0 ;;
    *) return 1 ;;
  esac
}

MODE="${1:-${MODE:-full}}"
if [[ "${MODE}" == "-h" || "${MODE}" == "--help" ]]; then
  usage
  exit 0
fi
if (( $# > 1 )); then
  usage >&2
  exit 2
fi
case "${MODE}" in
  audit|smoke|full|evaluate|summarize) ;;
  *)
    echo "ERROR: unsupported mode '${MODE}'." >&2
    usage >&2
    exit 2
    ;;
esac

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "ERROR: Python environment not found at ${PYTHON_BIN}." >&2
  echo "Create it with: uv venv .venv --python 3.11" >&2
  exit 1
fi

GPU_IDS="${GPU_IDS:-${CUDA_VISIBLE_DEVICES:-0,1}}"
export CUDA_VISIBLE_DEVICES="${GPU_IDS}"

IFS=',' read -r -a GPU_ARRAY <<<"${CUDA_VISIBLE_DEVICES}"
NUM_GPUS=0
for gpu_id in "${GPU_ARRAY[@]}"; do
  if [[ -n "${gpu_id//[[:space:]]/}" ]]; then
    NUM_GPUS=$((NUM_GPUS + 1))
  fi
done

EXPECTED_GPUS="${EXPECTED_GPUS:-2}"
EXPECTED_GPU_NAME="${EXPECTED_GPU_NAME:-A100}"
if (( NUM_GPUS != EXPECTED_GPUS )); then
  echo "ERROR: GPU_IDS=${GPU_IDS} exposes ${NUM_GPUS} GPUs; expected ${EXPECTED_GPUS}." >&2
  exit 1
fi

# These values describe the fixed built-in preset. Changing them here would not
# patch the scientific config, so reject inconsistent values instead of silently
# running a different experiment than the command claims.
N_EXAMPLES="${N_EXAMPLES:-10000}"
BATCH_SIZE="${BATCH_SIZE:-64}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
SEEDS="${SEEDS:-0}"
USE_LORA="${USE_LORA:-true}"
N_EVAL_EXAMPLES="${N_EVAL_EXAMPLES:-0}"
DO_FIRST_EVAL="${DO_FIRST_EVAL:-false}"
SAVE_EVERY_EXAMPLES="${SAVE_EVERY_EXAMPLES:-10000}"

[[ "${N_EXAMPLES}" == "10000" ]] || { echo "ERROR: this preset requires N_EXAMPLES=10000." >&2; exit 1; }
[[ "${BATCH_SIZE}" == "64" ]] || { echo "ERROR: this preset requires BATCH_SIZE=64." >&2; exit 1; }
[[ "${SEEDS}" == "0" ]] || { echo "ERROR: this launcher requires the single training seed SEEDS=0." >&2; exit 1; }
[[ "${N_EVAL_EXAMPLES}" == "0" ]] || { echo "ERROR: this preset requires N_EVAL_EXAMPLES=0." >&2; exit 1; }
[[ "${SAVE_EVERY_EXAMPLES}" == "10000" ]] || { echo "ERROR: this preset requires SAVE_EVERY_EXAMPLES=10000." >&2; exit 1; }
is_true "${USE_LORA}" || { echo "ERROR: this preset requires USE_LORA=true." >&2; exit 1; }
if is_true "${DO_FIRST_EVAL}"; then
  echo "ERROR: this preset requires DO_FIRST_EVAL=false." >&2
  exit 1
fi
if ! [[ "${GRAD_ACCUM}" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: GRAD_ACCUM must be a positive integer." >&2
  exit 1
fi

DIVISOR=$((GRAD_ACCUM * NUM_GPUS))
if (( BATCH_SIZE % DIVISOR != 0 )); then
  echo "ERROR: BATCH_SIZE=${BATCH_SIZE} must be divisible by GRAD_ACCUM*NUM_GPUS=${DIVISOR}." >&2
  exit 1
fi

NOISE_RATE="${NOISE_RATE:-0.20}"
NOISE_SEED="${NOISE_SEED:-2026}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs/sensitivity/llama3-10k-bs64-2xa100-ga8}"
EXCLUDE_DEFAULT_POINTS="${EXCLUDE_DEFAULT_POINTS:-true}"
HF_REPO_ID="${HF_REPO_ID:-ducthang1703/llama3-arc-bpo-sensitivity-10k-bs64-2xa100-ga8}"
HF_PRIVATE="${HF_PRIVATE:-false}"
HF_UPLOAD_ADAPTER_ONLY="${HF_UPLOAD_ADAPTER_ONLY:-true}"
HF_TOKEN_FILE="${HF_TOKEN_FILE:-}"

export HF_REPO_ID HF_PRIVATE HF_UPLOAD_ADAPTER_ONLY
mkdir -p "${OUTPUT_ROOT}"

echo "[MODE] ${MODE}"
echo "[GPUS] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} (${NUM_GPUS} GPUs)"
echo "[BATCH] global=${BATCH_SIZE} grad_accum=${GRAD_ACCUM} per_gpu_microbatch=$((BATCH_SIZE / DIVISOR))"
echo "[DATA] nominal_examples=${N_EXAMPLES} seeds=${SEEDS} noise_rate=${NOISE_RATE}"
echo "[OUTPUT] ${OUTPUT_ROOT}"

ensure_hf_auth() {
  if [[ -z "${HF_REPO_ID}" ]]; then
    echo "[HF] HF_REPO_ID is empty; checkpoint upload is disabled."
    return
  fi

  if [[ -z "${HF_TOKEN:-}" && -n "${HF_TOKEN_FILE}" ]]; then
    if [[ ! -r "${HF_TOKEN_FILE}" ]]; then
      echo "ERROR: HF_TOKEN_FILE is not readable: ${HF_TOKEN_FILE}" >&2
      exit 1
    fi
    IFS= read -r HF_TOKEN < "${HF_TOKEN_FILE}"
  fi

  if [[ -z "${HF_TOKEN:-}" ]] && command -v hf >/dev/null 2>&1; then
    if hf auth whoami >/dev/null 2>&1; then
      echo "[HF] Using the credential cached by 'hf auth login'."
      return
    fi
  fi

  if [[ -z "${HF_TOKEN:-}" ]]; then
    if [[ ! -t 0 ]]; then
      echo "ERROR: no Hugging Face credential is available." >&2
      echo "Set HF_TOKEN, set HF_TOKEN_FILE, or run 'hf auth login'." >&2
      exit 1
    fi
    read -rsp "Hugging Face token: " HF_TOKEN
    echo
  fi

  if [[ -z "${HF_TOKEN}" ]]; then
    echo "ERROR: Hugging Face token cannot be empty." >&2
    exit 1
  fi
  export HF_TOKEN
  echo "[HF] Using HF_TOKEN without printing it."
}

COMMON_ARGS=(
  --preset llama3-10k-bs64
  --gradient_accumulation_steps "${GRAD_ACCUM}"
  --output_root "${OUTPUT_ROOT}"
  --seeds "${SEEDS}"
  --noise_rate "${NOISE_RATE}"
  --noise_seed "${NOISE_SEED}"
  --expected_gpus "${EXPECTED_GPUS}"
  --expected_gpu_name "${EXPECTED_GPU_NAME}"
)
if is_true "${EXCLUDE_DEFAULT_POINTS}"; then
  COMMON_ARGS+=(--exclude_default_points)
fi

case "${MODE}" in
  audit)
    CMD=("${PYTHON_BIN}" -u run_sensitivity.py "${COMMON_ARGS[@]}")
    ;;
  smoke)
    ensure_hf_auth
    CMD=("${PYTHON_BIN}" -u run_sensitivity.py "${COMMON_ARGS[@]}" --max_runs 1 --execute)
    ;;
  full)
    ensure_hf_auth
    CMD=("${PYTHON_BIN}" -u run_sensitivity.py "${COMMON_ARGS[@]}" --execute)
    ;;
  evaluate)
    CMD=(
      "${PYTHON_BIN}" -u evaluate_sensitivity.py
      --manifest "${OUTPUT_ROOT}/run_manifest.csv"
      --only_status trained,checkpoint_exists
      --tensor_parallel_size "${NUM_GPUS}"
      --dtype bfloat16
      --gpu_memory_utilization 0.90
      --max_model_len 4096
      --batch_size auto:4
      --merge_device auto
      --merge_dtype bfloat16
    )
    ;;
  summarize)
    CMD=(
      "${PYTHON_BIN}" -u summarize_sensitivity.py
      --manifest "${OUTPUT_ROOT}/run_manifest.csv"
      --published_anchors sensitivity/published_anchors.json
      --main_result sensitivity/published_main_result.json
    )
    ;;
esac

LOG_FILE="${LOG_FILE:-${OUTPUT_ROOT}/${MODE}.log}"
printf '[RUN]'
printf ' %q' "${CMD[@]}"
printf '\n'
"${CMD[@]}" 2>&1 | tee "${LOG_FILE}"
