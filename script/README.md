# ARC-BPO Script Guide

This folder contains shell entrypoints for running ARC-BPO training and
evaluation outside Modal, for example on a rented SSH server.

The three main ARC-BPO training scripts are:

| Script | Model config | Base model | Default dataset |
| --- | --- | --- | --- |
| `train/arc_bpo_llama.sh` | `llama_8b` | `RLHFlow/LLaMA3-SFT-v2` | `princeton-nlp/llama3-ultrafeedback-armorm` |
| `train/arc_bpo_mistral.sh` | `mistral_7b` | `HuggingFaceH4/mistral-7b-sft-alpha` | `HuggingFaceH4/ultrafeedback_binarized` |
| `train/arc_bpo_qwen.sh` | `qwen2_5_7b_instruct` | `Qwen/Qwen2.5-7B-Instruct` | `HuggingFaceH4/ultrafeedback_binarized` |

All three scripts run ARC-BPO with LoRA by default, W&B disabled, FSDPTrainer,
activation checkpointing, response length `MAX_LENGTH=2048`, and uniform
ARC-BPO chunk shape (`USE_ADVANTAGE_SHAPE=false`).

## 1. Server Setup

On a new Linux GPU server:

```bash
git clone <your-repo-url> ARC-BPO
cd ARC-BPO

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

## 2. Important Variables

All scripts are configured through environment variables.

| Variable | Default | Meaning |
| --- | --- | --- |
| `GPU_IDS` | `0` | Comma-separated GPUs, for example `0,1,2,3`. |
| `N_EXAMPLES` | empty | If set, train on exactly this many examples instead of full epochs. |
| `N_EPOCHS` | `1` | Number of epochs when `N_EXAMPLES` is empty. |
| `BATCH_SIZE` | auto | Global batch size. Must divide by `GRAD_ACCUM * NUM_GPUS`. |
| `GRAD_ACCUM` | `4` | Gradient accumulation steps. Higher value lowers per-GPU memory. |
| `PER_GPU_MICROBATCH` | `2` | Used only when `BATCH_SIZE` is not set. |
| `N_EVAL_EXAMPLES` | `64` | Set to `0` to disable internal train-time eval metrics. |
| `DO_FIRST_EVAL` | `true` | Set to `false` to skip eval before training. |
| `USE_LORA` | `true` | Use LoRA. Set `false` for full fine-tuning. |
| `OUTPUT_DIR` | `<repo>/output` | Local training outputs and checkpoints. |
| `LOG_DIR` | `<output>/logs` | Training logs saved with `tee`. |

The real memory knob is:

```text
per_gpu_microbatch = BATCH_SIZE / GRAD_ACCUM / NUM_GPUS
```

If you hit CUDA OOM, lower `BATCH_SIZE`, raise `GRAD_ACCUM`, or lower
`MAX_LENGTH`.

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

10k examples, matching the Modal-style run:

```bash
GPU_IDS=0,1,2,3 \
N_EXAMPLES=10000 \
BATCH_SIZE=64 \
GRAD_ACCUM=4 \
N_EVAL_EXAMPLES=0 \
DO_FIRST_EVAL=false \
USE_LORA=true \
HF_REPO_ID=ducthang1703/llama3-arc-bpo-uniform-lora-10k-bs64 \
HF_PRIVATE=false \
bash script/train/arc_bpo_llama.sh
```

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

For full fine-tuning:

```bash
GPU_IDS=0,1,2,3 \
N_EXAMPLES=10000 \
BATCH_SIZE=16 \
GRAD_ACCUM=4 \
N_EVAL_EXAMPLES=0 \
DO_FIRST_EVAL=false \
USE_LORA=false \
HF_REPO_ID=ducthang1703/llama3-arc-bpo-uniform-full-10k \
HF_PRIVATE=false \
HF_UPLOAD_ADAPTER_ONLY=false \
bash script/train/arc_bpo_llama.sh
```

## 5. Mistral Runs

10k examples:

```bash
GPU_IDS=0,1,2,3 \
N_EXAMPLES=10000 \
BATCH_SIZE=64 \
GRAD_ACCUM=4 \
N_EVAL_EXAMPLES=0 \
DO_FIRST_EVAL=false \
USE_LORA=true \
bash script/train/arc_bpo_mistral.sh
```

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

10k examples:

```bash
GPU_IDS=0,1,2,3 \
N_EXAMPLES=10000 \
BATCH_SIZE=64 \
GRAD_ACCUM=4 \
N_EVAL_EXAMPLES=0 \
DO_FIRST_EVAL=false \
USE_LORA=true \
bash script/train/arc_bpo_qwen.sh
```

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
