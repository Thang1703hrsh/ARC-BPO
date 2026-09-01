# ARC-BPO Credit-Conditioned Policy Drift

`analyze_credit_drift.py` implements the mechanism-oriented reviewer analysis
described in `ARC_BPO_Generation_Diversity_Reviewer_Codex_Plan.md`. It performs
inference only; it never trains or updates either model.

The analysis:

1. loads an existing ARC-BPO policy and its exact frozen reference model;
2. reads held-out preference pairs using the repository's two-turn dataset
   schema;
3. reuses `tokenize_batch_element` and the exact semantic chunker used during
   training;
4. computes observed-token ARC-BPO chunk log-ratios;
5. computes full-vocabulary token KL in the `policy || reference` direction;
6. averages token KL inside each semantic chunk;
7. creates fixed Low/Medium/High credit tertiles;
8. reports preference-pair cluster-bootstrap confidence intervals and
   Spearman correlations;
9. repeats grouped results for winner and loser chunks.

## 1. Environment

Run on a Linux CUDA server from the repository root:

```bash
conda create -n arc-bpo-credit-drift python=3.11 -y
conda activate arc-bpo-credit-drift
pip install -r diversity_metrics/requirements_credit_drift.txt
```

Hugging Face authentication may be required:

```bash
huggingface-cli login
```

## 2. Choose the credit source deliberately

The script supports three allocation modes:

| Mode | Meaning |
|---|---|
| `logratio` | The reviewer's proposed post-hoc proxy: winsorized detached checkpoint chunk log-ratios. |
| `uniform` | The actual default used by the public ARC-BPO launchers (`loss.use_advantage_shape=false`). |
| `dataset_score` | The repository's training-time score proxy derived from `score_chosen` and `score_rejected`. |

`logratio` is the mode specified by the reviewer experiment plan. It must be
described as a **post-hoc credit proxy** unless the checkpoint was actually
trained using that same allocation rule. Do not describe it as the historical
training target of a checkpoint trained with uniform allocation.

For a public Llama ARC-BPO run, it is useful to report both `logratio` (the
reviewer diagnostic) and `uniform` (the exact training-target reconstruction).

For the ready-to-run single-A100 Modal pipeline using
`ducthang1703/llama3-arc-bpo-uniform-lora-10k-bs64`, see
[`MODAL.md`](MODAL.md). It runs both credit modes, creates held-out generation
prompts, computes output-level diversity metrics, pins Hub revisions, and saves
all artifacts to a Modal Volume.

## 3. Smoke test with a local LoRA checkpoint

The public train scripts save LoRA checkpoints under
`output/<run>/LATEST/adapter`. The analyzer auto-detects either the `LATEST`
directory or the nested `adapter` directory, reads the base model from
`adapter_config.json`, and disables the adapter for reference logits when the
reference equals that base model. This avoids loading two copies of Llama-3-8B.

Replace `<run>` with the real run directory:

```bash
python analyze_credit_drift.py \
  --policy_model "output/<run>/LATEST" \
  --reference_model "RLHFlow/LLaMA3-SFT-v2" \
  --dataset "princeton-nlp/llama3-ultrafeedback-armorm" \
  --split test \
  --output_dir "outputs/credit_drift_smoke" \
  --allocation_mode logratio \
  --beta 0.1 \
  --delta0 2.5 \
  --temperature 2.0 \
  --kappa 2.0 \
  --min_tokens_per_chunk 4 \
  --max_tokens_per_chunk 64 \
  --max_length 2048 \
  --max_examples 8 \
  --bootstrap_iterations 100 \
  --dtype bfloat16 \
  --device_map auto \
  --kl_device cuda:0 \
  --seed 42
```

Inspect `outputs/credit_drift_smoke/summary.md` and
`chunk_level_metrics.csv` before starting the full run.

If the adapter base stored in `adapter_config.json` is stale or unavailable,
override it explicitly:

```bash
--policy_base_model "RLHFlow/LLaMA3-SFT-v2"
```

## 4. Full reviewer run

After the smoke test succeeds, remove the example limit and use 2,000
preference-pair bootstrap replicates:

```bash
python analyze_credit_drift.py \
  --policy_model "output/<run>/LATEST" \
  --reference_model "RLHFlow/LLaMA3-SFT-v2" \
  --dataset "princeton-nlp/llama3-ultrafeedback-armorm" \
  --split test \
  --output_dir "outputs/credit_drift/llama_logratio" \
  --allocation_mode logratio \
  --beta 0.1 \
  --delta0 2.5 \
  --temperature 2.0 \
  --kappa 2.0 \
  --min_tokens_per_chunk 4 \
  --max_tokens_per_chunk 64 \
  --max_length 2048 \
  --bootstrap_iterations 2000 \
  --confidence_level 0.95 \
  --dtype bfloat16 \
  --device_map auto \
  --kl_device cuda:0 \
  --kl_token_batch_size 8 \
  --seed 42
```

To reconstruct the actual uniform targets of the public launcher, run the same
command with a different output directory and:

```bash
--allocation_mode uniform
```

## 5. Merged policy checkpoint

For a fully merged model, point `--policy_model` directly at the model and the
script will load a separate reference model:

```bash
python analyze_credit_drift.py \
  --policy_model "PATH_OR_HF_REPO_TO_MERGED_ARC_BPO" \
  --reference_model "RLHFlow/LLaMA3-SFT-v2" \
  --dataset "princeton-nlp/llama3-ultrafeedback-armorm" \
  --split test \
  --allocation_mode logratio \
  --output_dir "outputs/credit_drift/merged_policy" \
  --beta 0.1 \
  --delta0 2.5
```

This usually needs enough memory for both 8B models. Prefer the LoRA command
when the policy is available as an adapter over the reference base.

## 6. Outputs

Each run writes:

```text
outputs/credit_drift/<run>/
|-- chunk_level_metrics.csv
|-- grouped_credit_drift.csv
|-- grouped_credit_drift_winner.csv
|-- grouped_credit_drift_loser.csv
|-- correlation_results.json
|-- config.json
|-- credit_group_kl.pdf
|-- credit_vs_kl.pdf
`-- summary.md
```

The script stops on failed chunk coverage, target calibration error over
`1e-5`, materially negative KL, vocabulary mismatch, or accidental use of a
split containing `train` without explicit approval.

## 7. Interpretation

Only claim support for the localized-update mechanism if the actual results
show the expected ordering and the continuous analysis is consistent with it.
The generated `summary.md` reports whether the strict group ordering was
observed. Even a supportive result is associative evidence, not a theoretical
or causal guarantee that chunk-level optimization produces diversity.
