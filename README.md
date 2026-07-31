# ARC-BPO: Advantage-Referenced Chunk-Level Bregman Preference Optimization

Official implementation of

> Advantage-Referenced Chunk-Level Bregman Preference Optimization

<p align="left">
  <a href="#installation"><img alt="python" src="https://img.shields.io/badge/python-3.10%2B-blue.svg"></a>
  <a href="#installation"><img alt="pytorch" src="https://img.shields.io/badge/pytorch-cuda-ee4c2c.svg"></a>
  <a href="#license"><img alt="license" src="https://img.shields.io/badge/license-Apache%202.0-green.svg"></a>
</p>

ARC-BPO is an offline preference-optimization method for aligning language
models from pairwise feedback. It keeps the response-level ratio-matching
principle used by DPO and BPO, but moves the learning signal to semantic
chunks without comparing winner and loser chunks across different prefix
states.

The method is built around three design choices:

- deterministic semantic chunks, with exact policy/reference likelihood ratios
  induced by the autoregressive factorization;
- one-sided chunk targets, so each chunk is matched at its own prefix and no
  state-dependent correction or value head is needed;
- a calibrated response-level margin, distributed across chunks by an
  admissible target shape and optimized with an SBA Bregman objective.

The public training scripts in this repository use a uniform admissible target
shape by default. This preserves the calibrated response-level target. The loss
also supports shaped allocation through `loss.use_advantage_shape=true` when
per-response score proxies are available in the dataset.

## Contents

- [Repository Layout](#repository-layout)
- [Installation](#installation)
- [Training](#training)
- [Evaluation](#evaluation)
- [Tests](#tests)
- [Citation](#citation)

## Repository Layout

```text
.
|-- train.py                       # Hydra entry point
|-- trainers.py                    # BasicTrainer and FSDPTrainer
|-- arc_bpo_chunking.py            # deterministic semantic chunking
|-- preference_datasets.py         # HF preference-data loading and tokenization
|-- loss/
|   |-- loss.py                    # ARC-BPO, BPO, DPO-style losses
|   |-- loss_utils.py              # log-prob, KL, and chunk-ratio utilities
|   `-- h_function.py              # Bregman generators
|-- config/
|   |-- config.yaml                # top-level Hydra config
|   |-- model/                     # Llama, Mistral, Qwen model presets
|   `-- loss/arc_bpo.yaml          # ARC-BPO loss preset
|-- script/
|   |-- README.md                  # server training guide
|   |-- train/arc_bpo_*.sh         # ARC-BPO launch scripts
|   `-- eval/general/              # lm-eval scripts
|-- winrate_eval/                  # AlpacaEval-style win-rate evaluation
|-- diversity_metrics/             # generation diversity scripts
|-- mtbench/FastChat/              # vendored FastChat for MT-Bench
|-- env/                           # conda environment specs
`-- tests/                         # ARC-BPO unit tests
```

Legacy comparison utilities are kept in the repository, but the ARC-BPO entry
points are `config/loss/arc_bpo.yaml` and the three `script/train/arc_bpo_*.sh`
launchers.

## Installation

Training is intended for Linux GPU servers. The shell scripts use `bash`, CUDA,
and Hugging Face model downloads.

```bash
conda create -n arc-bpo python=3.10 -y
conda activate arc-bpo

# Pick the PyTorch CUDA wheel that matches your server. Example for CUDA 12.1:
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

Login to Hugging Face before training gated or hosted models.

```bash
huggingface-cli login
```

or set a token in the shell:

```bash
export HF_TOKEN="hf_xxx"
```

For the general LM evaluation scripts, the pinned evaluation environment is
kept separately:

```bash
conda env create -f env/eval_env.yml
conda activate TokenBPO-eval
```

## Training

The main ARC-BPO launch scripts are:

```text
script/train/arc_bpo_llama.sh
script/train/arc_bpo_mistral.sh
script/train/arc_bpo_qwen.sh
```

Run scripts from the repository root unless your scheduler already starts jobs
there.

### Quick Smoke Test

Run a small LoRA job before launching a full run.

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

The same pattern works for Mistral and Qwen:

```bash
bash script/train/arc_bpo_mistral.sh
bash script/train/arc_bpo_qwen.sh
```

After a smoke test, check:

```bash
ls output/logs
find output -name LATEST
```

### Full ARC-BPO Runs

Llama-3-8B uses `princeton-nlp/llama3-ultrafeedback-armorm`, split into
`train` and `test`, initialized from `RLHFlow/LLaMA3-SFT-v2`.

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

Mistral uses `HuggingFaceH4/ultrafeedback_binarized`, split into
`train_prefs` and `test_prefs`, initialized from
`HuggingFaceH4/mistral-7b-sft-alpha`.

```bash
GPU_IDS=0,1,2,3 \
BATCH_SIZE=64 \
GRAD_ACCUM=4 \
N_EVAL_EXAMPLES=0 \
DO_FIRST_EVAL=false \
USE_LORA=true \
HF_REPO_ID=ducthang1703/mistral-arc-bpo-uniform-lora-full \
HF_PRIVATE=false \
bash script/train/arc_bpo_mistral.sh
```

Qwen uses `HuggingFaceH4/ultrafeedback_binarized`, split into `train_prefs`
and `test_prefs`, initialized from `Qwen/Qwen2.5-7B-Instruct`.

```bash
GPU_IDS=0,1,2,3 \
BATCH_SIZE=64 \
GRAD_ACCUM=4 \
N_EVAL_EXAMPLES=0 \
DO_FIRST_EVAL=false \
USE_LORA=true \
HF_REPO_ID=ducthang1703/qwen25-7b-instruct-arc-bpo-uniform-lora-full \
HF_PRIVATE=false \
bash script/train/arc_bpo_qwen.sh
```

### Important Overrides

The scripts expose the main ARC-BPO hyperparameters as environment variables.

```text
BETA=0.1
DELTA_STAR=2.5        # Llama and Mistral default
DELTA_STAR=2.0        # Qwen default
ARC_T=2.0
KAPPA=2.0
SBA_LAMBDA=1.0
SBA_SCALE=4.0
MIN_TOKENS_PER_CHUNK=4
MAX_TOKENS_PER_CHUNK=64
USE_ADVANTAGE_SHAPE=false
FALLBACK_TO_UNIFORM_SHAPE=true
```

`BATCH_SIZE` must be divisible by:

```text
GRAD_ACCUM * number_of_visible_gpus
```

For example, with four GPUs and `GRAD_ACCUM=4`, valid batch sizes include
`16`, `32`, `64`, and `128`.

If the job runs out of memory, lower the per-GPU microbatch first:

```bash
BATCH_SIZE=16 GRAD_ACCUM=8 MAX_LENGTH=2048 bash script/train/arc_bpo_llama.sh
```

For a single GPU, start smaller:

```bash
BATCH_SIZE=4 GRAD_ACCUM=4 MAX_LENGTH=2048 bash script/train/arc_bpo_llama.sh
```

### Direct Hydra CLI

All script settings can also be passed directly to `train.py`.

```bash
python train.py \
  model=llama_8b \
  loss=arc_bpo \
  datasets=princeton-nlp/llama3-ultrafeedback-armorm \
  dataset_train_split=train \
  dataset_test_split=test \
  batch_size=32 \
  gradient_accumulation_steps=4 \
  lr=5e-7 \
  loss.delta_star=2.5 \
  loss.beta=0.1 \
  loss.min_tokens_per_chunk=4 \
  loss.max_tokens_per_chunk=64
```

### Outputs and Uploads

Training logs are written to:

```text
output/logs/*.log
```

Final checkpoints are saved under:

```text
output/<run_name>/LATEST
```

For LoRA runs, the adapter is saved under:

```text
output/<run_name>/LATEST/adapter
```

If `HF_REPO_ID` is set, the script uploads the selected checkpoint folder to:

```text
https://huggingface.co/<HF_REPO_ID>
```

With `USE_LORA=true` and `HF_UPLOAD_ADAPTER_ONLY=true`, only the adapter folder
is uploaded. With `USE_LORA=false`, or `HF_UPLOAD_ADAPTER_ONLY=false`, the whole
`LATEST` folder is uploaded.

## Evaluation

The repository keeps evaluation tools separate from training.

### General LM Benchmarks

The `script/eval/general/` scripts evaluate ARC, GSM8K, HellaSwag, MMLU, QA,
and Winogrande through `lm-eval` with a vLLM backend.

Edit `MODEL_NAME`, `TP_SIZE`, `BATCH_SIZE`, and `MAX_LEN` in:

```text
script/eval/general/runall.sh
```

then run:

```bash
conda activate TokenBPO-eval
bash script/eval/general/runall.sh
```

### Win Rate

The win-rate harness is under `winrate_eval/`.

```bash
conda create -n alpaca-eval python=3.11.11 -y
conda activate alpaca-eval
pip install 'alpaca-eval[all]'

bash winrate_eval/eval.sh
```

Judge prompts, model endpoints, and client settings live under
`winrate_eval/{annotators_configs,model_configs,client_configs}/`.

### Diversity

Generation diversity scripts are under `diversity_metrics/`.

```bash
conda create -n diversity-metrics python=3.11 -y
conda activate diversity-metrics
pip install vllm transformers sacrebleu tqdm
```

Generate completions:

```bash
python diversity_metrics/generation_vllm.py \
  --model <merged-model-path> \
  --prompts processed_data/diversity_prompts.jsonl \
  --out diversity_metrics/<run>/diversity_generations.jsonl \
  --k 5 \
  --tensor_parallel_size 2 \
  --batch_size 64 \
  --max_new_tokens 128 \
  --temperature 1.0 \
  --top_p 0.95
```

Compute metrics:

```bash
python diversity_metrics/compute_diversity.py \
  --infile diversity_metrics/<run>/diversity_generations.jsonl \
  --out diversity_metrics/<run>/diversity_metrics.json
```

### MT-Bench

MT-Bench support is provided through the vendored FastChat tree.

```bash
conda create -n mtbench python=3.11 -y
conda activate mtbench
cd mtbench/FastChat
pip install -e ".[model_worker,llm_judge]"
pip install vllm
```

Serve the candidate model with vLLM, generate answers with
`fastchat/llm_judge/gen_api_answer.py`, judge them with
`fastchat/llm_judge/gen_judgment.py`, and summarize results with
`fastchat/llm_judge/show_result.py`. The command sequence used in this repo is
documented in `mtbench/FastChat/notes.md`.

## Tests

ARC-BPO unit tests can be run without launching a training job:

```bash
python -m unittest discover -s tests -v
```

These tests cover chunk-ratio telescoping, calibrated one-sided targets,
detached target construction, gradient direction, variable chunk counts, and
the guard that ARC-BPO does not fall back to token-level matching or a value
head.

## Citation

If you use this repository, please cite:

```bibtex
@article{tran2026arcbpo,
  title={Advantage-Referenced Chunk-Level Bregman Preference Optimization},
  author={Thang Duc Tran and Tien-Phat Nguyen and Truong Nguyen and Duc Anh Nguyen and Linh Ngo Van and Thien Huu Nguyen and Trung Le},
  year={2026}
}
```

## Acknowledgements

The MT-Bench harness under `mtbench/FastChat/` is a vendored copy of LMSYS
FastChat and keeps its original Apache 2.0 license.

## License

Released under the Apache 2.0 License.
