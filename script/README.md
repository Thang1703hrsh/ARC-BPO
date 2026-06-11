# ARC-BPO Script Guide

## 1. Server Setup

```bash
conda create -n arc-bpo python=3.10 -y
conda activate arc-bpo

# Pick the PyTorch CUDA wheel that matches your server. Example for CUDA 12.1:
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

Login to Hugging Face so the server can download models/datasets. This is also
needed if you want the Llama script to upload the final checkpoint.

```bash
huggingface-cli login
```

Alternatively:

```bash
export HF_TOKEN="hf_xxx"
```

## 3. Quick Smoke Tests

Run a small test before launching a long job.

Llama:

```bash
GPU_IDS=0 \
N_EXAMPLES=64 \
BATCH_SIZE=4 \
GRAD_ACCUM=4 \
N_EVAL_EXAMPLES=0 \
DO_FIRST_EVAL=false \
USE_LORA=true \
bash script/train/arc_bpo_llama.sh
```

Mistral:

```bash
GPU_IDS=0 \
N_EXAMPLES=64 \
BATCH_SIZE=4 \
GRAD_ACCUM=4 \
N_EVAL_EXAMPLES=0 \
DO_FIRST_EVAL=false \
USE_LORA=true \
bash script/train/arc_bpo_mistral.sh
```

Qwen2.5:

```bash
GPU_IDS=0 \
N_EXAMPLES=64 \
BATCH_SIZE=4 \
GRAD_ACCUM=4 \
N_EVAL_EXAMPLES=0 \
DO_FIRST_EVAL=false \
USE_LORA=true \
bash script/train/arc_bpo_qwen.sh
```

After the smoke test, check:

```bash
ls output/logs
find output/train_runs -name LATEST
```

## 4. Llama Runs

Full train split:

```bash
GPU_IDS=0,1,2,3 \
BATCH_SIZE=64 \
GRAD_ACCUM=4 \
N_EVAL_EXAMPLES=0 \
DO_FIRST_EVAL=false \
USE_LORA=true \
HF_REPO_ID=ducthang1703/llama3-arc-bpo-uniform-lora-full-bs64 \
HF_PRIVATE=false \
bash script/train/arc_bpo_llama.sh
```

`arc_bpo_llama.sh` can upload to Hugging Face after training. If
`HF_REPO_ID` is empty, it only saves locally. With `USE_LORA=true`, it uploads
`LATEST/adapter` by default and checks that the adapter is not an empty 40-byte
file.

## 5. Mistral Runs

Full train split:

```bash
GPU_IDS=0,1,2,3 \
BATCH_SIZE=64 \
GRAD_ACCUM=4 \
N_EVAL_EXAMPLES=0 \
DO_FIRST_EVAL=false \
USE_LORA=true \
bash script/train/arc_bpo_mistral.sh
```

Mistral uses `HuggingFaceH4/ultrafeedback_binarized` with splits
`train_prefs` and `test_prefs`.

## 6. Qwen2.5 Runs

Full train split:

```bash
GPU_IDS=0,1,2,3 \
BATCH_SIZE=64 \
GRAD_ACCUM=4 \
N_EVAL_EXAMPLES=0 \
DO_FIRST_EVAL=false \
USE_LORA=true \
bash script/train/arc_bpo_qwen.sh
```

Qwen uses a gentler default ARC-BPO target margin:

```text
DELTA_STAR=2.0
```

Llama and Mistral default to:

```text
DELTA_STAR=2.5
```

## 7. Outputs

Training logs are saved to:

```text
output/logs/*.log
```

Final checkpoints are saved under:

```text
output/train_runs/<run_name>/LATEST
```

For LoRA runs, the adapter is saved under:

```text
output/train_runs/<run_name>/LATEST/adapter
```

The final `LATEST` checkpoint is still saved even when:

```bash
N_EVAL_EXAMPLES=0
DO_FIRST_EVAL=false
```

## 8. Common Fixes

If `BATCH_SIZE` is invalid, make it divisible by:

```text
GRAD_ACCUM * number_of_visible_gpus
```

For example, with 4 GPUs and `GRAD_ACCUM=4`, valid batch sizes include
`16`, `32`, `64`, and `128`.

If you hit OOM, start with:

```bash
BATCH_SIZE=16 GRAD_ACCUM=8 MAX_LENGTH=2048
```

or for a single GPU:

```bash
BATCH_SIZE=4 GRAD_ACCUM=4 MAX_LENGTH=2048
```

If Hugging Face download/upload fails, check:

```bash
huggingface-cli whoami
echo "$HF_TOKEN"
```
