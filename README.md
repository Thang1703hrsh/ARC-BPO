# TBPO: Token-Level Bregman Preference Optimization

Official implementation of

> **TokenRatio: Principled Token-Level Preference Optimization via Ratio Matching**

<p align="left">
  <a href="#installation"><img alt="python" src="https://img.shields.io/badge/python-3.11-blue.svg"></a>
  <a href="#installation"><img alt="pytorch" src="https://img.shields.io/badge/pytorch-cu124-ee4c2c.svg"></a>
  <a href="#license"><img alt="license" src="https://img.shields.io/badge/license-Apache%202.0-green.svg"></a>
</p>

---

## Contents

- [Overview](#overview)
- [Repository layout](#repository-layout)
- [Installation](#installation)
- [Training](#training)
- [Evaluation](#evaluation)
- [Citation](#citation)

---

## Overview

TBPO recasts token-level preference optimization as a **density-ratio matching** problem under a **Bregman divergence**. Token-DPO–style methods re-weight a fixed sequence-level loss; TBPO instead *directly fits the policy/reference log-ratio at the token level* using a family of pluggable divergences. This yields a single objective in which the choice of Bregman generator `h(·)` recovers and generalizes prior work.

Two principal variants are studied in the paper:

| Variant   | Idea                                              | Loss preset    |
|-----------|---------------------------------------------------|----------------|
| A-version | Token-level **A**dvantage-weighted ratio matching | `loss=A_tbpo`  |
| Q-version | **Q**-style token-level objective                 | `loss=Q_tbpo`  |

---

## Repository layout

```
.
├── train.py                    # Hydra entry point
├── trainers.py                 # BasicTrainer / FSDPTrainer + per-loss forward passes
├── preference_datasets.py      # HF preference-data loading & tokenization
├── merge.py                    # LoRA adapter → base-model merge utility
├── baseline_head.py            # Optional learned baseline head
├── utils.py                    # Distributed init, logging, run-dir helpers
│
├── loss/
│   ├── loss.py                 # preference / TDPO / TIS-DPO / Bregman losses
│   ├── loss_utils.py           # Per-method token-level log-prob computations
│   └── h_function.py           # Bregman h-functions: SBA, BA, LSIF, KLIEP, LR_DPO
│
├── config/
│   ├── config.yaml             # Top-level training config (Hydra)
│   ├── model/                  # llama_8b.yaml, mistral_7b.yaml, blank.yaml
│   └── loss/                   # A_tbpo.yaml, Q_tbpo.yaml
│
├── script/
│   ├── train/{A_version,Q_version}/   # LoRA + full-FT launchers (Llama / Mistral)
│   └── eval/general/                  # ARC, GSM8K, HellaSwag, MMLU, QA, Winogrande
│
├── winrate_eval/               # AlpacaEval-style win-rate harness
├── diversity_metrics/          # Generation + distinct-1 / self-BLEU / entropy
├── mtbench/FastChat/           # Vendored FastChat for MT-Bench judging
└── env/                        # Conda environment specs
```

---

## Installation

The project uses two isolated conda environments — one for training, one for general LM evaluation. The remaining benchmark suites (AlpacaEval, MT-Bench, diversity) bring their own envs and are documented inline.

**Training**

```bash
conda env create -f env/train_env.yml
conda activate TokenBPO-train
```

PyTorch is installed from the official CUDA 12.4 wheel index. Edit `env/train_env.yml` for other CUDA versions.

**Evaluation (general LM benchmarks)**

```bash
conda env create -f env/eval_env.yml
conda activate TokenBPO-eval
```

Pinned to `lm-eval==0.4.9.2` with the `vllm`, `ifeval`, and `math` extras.

> Tested on Linux + 4× NVIDIA A100/H100 with CUDA 12.4. Single-GPU runs work on consumer cards under LoRA but will require lowering `batch_size` and increasing `gradient_accumulation_steps`.

---

## Training

### Configuration (Hydra)

Configs are split into three composable layers; any field can be overridden on the command line.

| File | Purpose | Key options |
|---|---|---|
| `config/config.yaml`     | Top-level run            | `batch_size`, `lr`, `optimizer`, `trainer`, `n_epochs`, `max_length`, `wandb.*`, `datasets` |
| `config/model/*.yaml`    | Base model + PEFT        | `name_or_path`, `block_name`, `use_lora`, `lora_r`, `lora_alpha`, `lora_target_modules`     |
| `config/loss/*.yaml`     | Loss & Bregman generator | `name`, `bregman_loss.{name,lam,s}`, `beta`, `label_smoothing`                              |

Available presets:

- **Models** — `llama_8b` ([RLHFlow/LLaMA3-SFT-v2](https://huggingface.co/RLHFlow/LLaMA3-SFT-v2)), `mistral_7b`, `blank`
- **Losses** — `A_tbpo`, `Q_tbpo`
- **Bregman generators** — `sba`, `ba`, `lsif`, `kliep`, `lr` (DPO-equivalent)

Defaults that match the paper:

- LoRA enabled (`use_lora=true`, r=32, α=64, dropout=0.05)
- `FSDPTrainer` for multi-GPU runs
- RMSprop optimizer, cosine schedule, 5 % warmup
- `max_length=2048`, `batch_size=64`, `lr=5e-7`, `n_epochs=1`
- `beta=0.1`, `label_smoothing=0`

### Launch scripts

Two regimes per model and per variant. Both share the *same* loss, dataset, optimizer, and schedule — the only difference is the parameterization (`use_lora`).

**LoRA (matches the paper)** — low-rank adapters on a frozen base; lower memory, faster.

```bash
# A-version (advantage-weighted)
bash script/train/A_version/llama_general.sh
bash script/train/A_version/mistral_general.sh

# Q-version
bash script/train/Q_version/llama_general.sh
bash script/train/Q_version/mistral_general.sh
```

**Full fine-tune (maximum quality)** — updates all parameters; higher quality at substantially higher GPU cost.

```bash
bash script/train/A_version/llama_general_full.sh
bash script/train/A_version/mistral_general_full.sh
bash script/train/Q_version/llama_general_full.sh
bash script/train/Q_version/mistral_general_full.sh
```

Set `WANDB_API_KEY` and `CUDA_VISIBLE_DEVICES` at the top of each script before launching.

### Direct CLI

Any setting can be overridden directly:

```bash
python train.py \
    model=llama_8b \
    loss=A_tbpo \
    loss.bregman_loss.name=sba \
    loss.bregman_loss.s=4.0 \
    datasets=princeton-nlp/llama3-ultrafeedback-armorm \
    batch_size=32 \
    lr=5e-7
```

### Merging LoRA adapters

Required **only for LoRA runs** — full fine-tunes already save a complete model under `LATEST/` and can skip this step. Edit the paths in `merge.py` and run:

```bash
python merge.py
```

The script supports both local output and optional push-to-Hub.

---

## Evaluation

Five evaluation tracks are provided; they are independent and can be run in any order. Track 2 ("Pre-processed prompts") only downloads shared data used by tracks 3–5.

### 1 · General LM benchmarks

Six benchmarks (ARC, GSM8K, HellaSwag, MMLU, QA, Winogrande) via `lm-eval` with a vLLM backend.

1. Edit `MODEL_NAME`, `TP_SIZE`, `BATCH_SIZE`, and `MAX_LEN` at the top of `script/eval/general/runall.sh`.
2. Run:

   ```bash
   conda activate TokenBPO-eval
   bash script/eval/general/runall.sh
   ```

`runall.sh` cleans up vLLM processes between tasks to release GPU memory.

### 2 · Pre-processed prompts

Prompts and supporting data for tracks 3–5 are released on the Hub:

```bash
hf download tonyshelby/processed_data --repo-type dataset --local-dir ./processed_data
```

### 3 · Win rate (AlpacaEval)

```bash
conda create -n alpaca-eval python=3.11.11 -y
conda activate alpaca-eval
pip install 'alpaca-eval[all]'

bash winrate_eval/eval.sh
```

Judge model, model endpoints, and prompt templates live under `winrate_eval/{annotators_configs,model_configs,client_configs}/`. See `winrate_eval/notes.md` for the exact wiring used in the paper.

### 4 · Diversity (distinct-1, self-BLEU, predictive entropy)

```bash
conda create -n diversity-metrics python=3.11 -y
conda activate diversity-metrics
pip install vllm transformers sacrebleu tqdm
```

Two-stage pipeline (full reference: `diversity_metrics/notes.md`):

```bash
# (a) sample k completions per prompt
python diversity_metrics/generation_vllm.py \
    --model <merged-model-path> \
    --prompts processed_data/diversity_prompts.jsonl \
    --out diversity_metrics/<run>/diversity_generations.jsonl \
    --k 5 --tensor_parallel_size 2 --batch_size 64 \
    --max_new_tokens 128 --temperature 1.0 --top_p 0.95

# (b) compute distinct-1 / self-BLEU / predictive entropy
python diversity_metrics/compute_diversity.py \
    --infile diversity_metrics/<run>/diversity_generations.jsonl \
    --out    diversity_metrics/<run>/diversity_metrics.json
```

### 5 · MT-Bench

```bash
conda create -n mtbench python=3.11 -y
conda activate mtbench
cd mtbench/FastChat
pip install -e ".[model_worker,llm_judge]"
pip install vllm
```

Serve each candidate with vLLM, generate answers with `fastchat/llm_judge/gen_api_answer.py`, then score with `gen_judgment.py` (mode `pairwise-all`) and summarize with `show_result.py`. Full command sequence: `mtbench/FastChat/notes.md`.

---

## Citation

If you use this repository or find the paper useful, please cite:

```bibtex
@article{nguyen2026tokenration,
  title={TokenRatio: Principled Token-Level Preference Optimization via Ratio Matching},
  author={Truong Nguyen and Tien-Phat Nguyen and Linh Ngo Van and Duy Minh Ho Nguyen and Khoa D. Doan and Trung Le},
  journal={International Conference on Machine Learning},
  year={2026},
  url={https://arxiv.org/abs/2605.12288},
}
```

---

## Acknowledgements

The MT-Bench harness under `mtbench/FastChat/` is a vendored copy of [LMSYS FastChat](https://github.com/lm-sys/FastChat) and retains its original Apache 2.0 license.

## License

Released under the Apache 2.0 License.
