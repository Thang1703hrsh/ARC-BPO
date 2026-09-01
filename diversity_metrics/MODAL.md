# Run the Generation-Diversity Reviewer Analysis on Modal

[`modal_generation_diversity.py`](../modal_generation_diversity.py) runs the
reviewer analysis with exactly one Modal A100. Its defaults are already set to:

```text
ARC-BPO LoRA: ducthang1703/llama3-arc-bpo-uniform-lora-10k-bs64
Reference:    RLHFlow/LLaMA3-SFT-v2
Dataset:      princeton-nlp/llama3-ultrafeedback-armorm
Split:        test
GPU:          A100 (one GPU)
```

The job resolves and records immutable Hugging Face commit SHAs. It validates
that the adapter's recorded base is the requested reference model before
running any analysis.

## What the job runs

In one sequential GPU job it:

1. downloads/caches the LoRA adapter and exact reference snapshot;
2. runs the reviewer-requested post-hoc log-ratio credit analysis;
3. runs the uniform-allocation control that reconstructs this checkpoint's
   actual training target shape;
4. creates a fixed prompt set from the same held-out split;
5. merges the LoRA once for vLLM;
6. generates five samples per prompt and computes Predictive Entropy,
   Distinct-1, and Self-BLEU;
7. saves logs, raw generations, CSV/JSON, PDFs, Markdown, resolved revisions,
   and a downloadable `.tar.gz` archive.

The credit-drift steps do not merge the adapter or load a second 8B model.
They load one PEFT base and use `disable_adapter()` for the reference forward,
which is the memory-safe path for one A100.

Because this checkpoint is named and configured as **uniform allocation**, the
log-ratio analysis must be described as a post-hoc diagnostic. The
`credit_drift_uniform` result is the exact historical allocation control. Do
not claim that the post-hoc log-ratio values were its training targets.

## 1. Install and authenticate Modal

From the repository root in PowerShell:

```powershell
py -m pip install -r diversity_metrics/requirements_modal.txt
modal setup
```

Create a Modal secret containing a Hugging Face token. Using the dashboard is
preferred because it avoids putting the token into shell history. If
`HF_TOKEN` is already set locally, PowerShell can create the secret with:

```powershell
modal secret create huggingface-secret HF_TOKEN="$env:HF_TOKEN"
```

The secret name and key must be exactly `huggingface-secret` and `HF_TOKEN`.

## 2. Run the smoke test first

```powershell
modal run modal_generation_diversity.py --mode smoke
```

Smoke mode uses eight preference pairs, 100 bootstrap iterations, and the same
eight held-out prompts for generation. The prompt set is selected using the
exact dataset-row indices that survived credit-drift tokenization. It still
exercises adapter loading, shared-base reference logits, both allocation modes,
LoRA merge, vLLM generation, metric computation, plotting, Volume commits, and
archiving.

Expected result location inside the results Volume:

```text
llama3-arc-bpo-uniform-lora-10k-bs64/smoke/
llama3-arc-bpo-uniform-lora-10k-bs64-smoke.tar.gz
```

Download it:

```powershell
modal volume get arc-bpo-generation-diversity-results `
  llama3-arc-bpo-uniform-lora-10k-bs64-smoke.tar.gz `
  .\llama3-arc-bpo-uniform-lora-10k-bs64-smoke.tar.gz
```

Inspect these files before the full run:

```text
credit_drift_logratio/summary.md
credit_drift_logratio/grouped_credit_drift.csv
credit_drift_logratio/correlation_results.json
credit_drift_uniform/summary.md
diversity/diversity_metrics.json
modal_run_config.json
```

## 3. Run the full held-out analysis

```powershell
modal run modal_generation_diversity.py --mode full
```

Full mode uses all valid examples/prompts in the pinned held-out `test` split
and 2,000 preference-pair bootstrap iterations. It may be a long job. Completed
substeps are detected and reused after a retry, and model snapshots plus the
merged model remain in `arc-bpo-hf-cache`.

Download the full archive:

```powershell
modal volume get arc-bpo-generation-diversity-results `
  llama3-arc-bpo-uniform-lora-10k-bs64-full.tar.gz `
  .\llama3-arc-bpo-uniform-lora-10k-bs64-full.tar.gz
```

## Useful controlled overrides

Run a bounded reviewer job, for example 256 credit-drift pairs and 256 prompts:

```powershell
modal run modal_generation_diversity.py `
  --mode full `
  --max-examples 256 `
  --generation-prompts 256
```

Pin the adapter to a known commit:

```powershell
modal run modal_generation_diversity.py `
  --mode full `
  --checkpoint-revision <COMMIT_SHA>
```

Rerun and replace the exact smoke/full output directory explicitly:

```powershell
modal run modal_generation_diversity.py --mode smoke --force
```

`--force` deletes only that checkpoint's exact `smoke` or `full` result
directory before rerunning. It does not delete the shared Hugging Face cache.

The scientific defaults exposed by the CLI are:

```text
beta=0.1
delta0=2.5
temperature=2.0
kappa=2.0
min_tokens_per_chunk=4
max_tokens_per_chunk=64
max_length=2048
seed=42
generation_seed=1234
```

These match the repository's public Llama uniform launcher defaults. Since a
PEFT adapter normally does not contain the complete training config, compare
`modal_run_config.json` against the `config.yaml` from the original 10k run
before using values in the rebuttal. Override any mismatch on the Modal CLI;
for example, `--delta0 2.0`.

## Output interpretation

Only use the localized-drift explanation if the actual log-ratio diagnostic
supports it. The generated summary reports the observed group ordering and
Spearman result without forcing the expected trend. Even a monotonic result is
associative mechanism evidence, not proof that ARC-BPO causally guarantees
diversity.

The generation script estimates entropy from the renormalized top-20 token
log-probabilities returned by vLLM. This estimator is recorded in both the raw
generation metadata and `modal_run_config.json`; it should not be described as
exact full-vocabulary entropy. Credit drift, in contrast, uses exact
full-vocabulary `KL(policy || reference)` in FP32 blocks.
