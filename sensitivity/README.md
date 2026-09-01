# ARC-BPO Hyperparameter Sensitivity

This pipeline implements the controlled one-factor-at-a-time study in
`ARC_BPO_Hyperparameter_Sensitivity_Codex_Spec.md`. It creates and audits the
training configs, evaluates every final checkpoint with one frozen six-task
protocol, and generates run-level/aggregated data, PDF plots, LaTeX tables,
and a factual Markdown summary.

## Important prerequisite: the actual main config

`run_sensitivity.py` requires the **resolved `config.yaml` saved by the final,
advantage-enabled ARC-BPO main run**. It deliberately rejects the repository's
uncomposed Hydra defaults and any config with:

```yaml
loss:
  use_advantage_shape: false
```

The public launchers currently default to the uniform allocation above. That
configuration cannot identify effects of `T` or `kappa`; do not silently use it
as the sensitivity baseline. Point `--base_config` to the final main run's
`output/<run>/config.yaml`. If that main run does not exist yet, first run the
intended advantage-enabled main experiment, for example by setting
`USE_ADVANTAGE_SHAPE=true` on the Llama launcher after confirming that its
dataset has the required response scores.

The pipeline copies that exact resolved config to:

```text
outputs/sensitivity/default_config.json
outputs/sensitivity/default_config.yaml
```

## 1. Install

Training is designed for a Linux CUDA server:

```bash
conda create -n arc-bpo python=3.10 -y
conda activate arc-bpo
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

Evaluation uses the same pinned harness generation as the repository's general
evaluation scripts:

```bash
pip install -r sensitivity/requirements.txt
```

Login to Hugging Face if the base model or dataset is gated.

## 2. Audit the sweep before training

Use matched seeds for every point. This command creates the full three-seed
grid without starting a GPU job:

```bash
python run_sensitivity.py \
  --base_config "output/<final-main-run>/config.yaml" \
  --output_root outputs/sensitivity \
  --seeds 42,123,2026
```

For a quick launcher smoke test:

```bash
python run_sensitivity.py \
  --base_config "output/<final-main-run>/config.yaml" \
  --output_root outputs/sensitivity-smoke \
  --seeds 42 \
  --max_runs 2
```

Inspect `config_audit.json`, `run_manifest.csv`, and each run's
`config_diff.json`. The audit fails if a run changes anything beyond its one
swept control, the matched seed, label-noise controls, and output identity.

The current Spec grids are:

- `T`: 4, 2, 1, 0.5 on clean labels;
- `kappa`: 3, 2, 1.5, 1 on clean and 20% noisy labels;
- `delta0`: 0.5, 1, 2, 4 on clean labels;
- `lambda`: 0.5, 1, 2 on clean labels only.

The uniform/no-winsorization/zero-lambda limits from the older Plan are not
part of the current experiment. Pass `--exclude_default_points` when the exact
clean and noisy default rows from the Spec are being reused; this produces the
14 missing training runs rather than retraining duplicate anchors.

The optional Hard/RM calibration strategies are not generated because this
repository does not define validated implementations of them. They are not
numeric `Delta0` values and should not be emulated with invented constants.

## 3. Train

After inspecting the audit, add `--execute`:

```bash
python run_sensitivity.py \
  --base_config "output/<final-main-run>/config.yaml" \
  --output_root outputs/sensitivity \
  --seeds 42,123,2026 \
  --execute
```

For a compute-limited controlled study, use `--seeds 42` and state that the
study is single-seed. Runs are sequential and use the trainer, GPU visibility,
schedule, LoRA setting, initialization, and data splits recorded by the main
config. Existing `LATEST` checkpoints are skipped unless `--force` is passed.

Every noisy run points to the same
`outputs/sensitivity/noise20_indices.json`. The first noisy training job samples
exactly `round(0.20 * N)` pair indices without replacement and writes the
deterministic manifest; subsequent jobs validate and replay that exact set.
Evaluation iterators remain clean.

## Full Modal run: Llama-3-8B / 10k / global batch size 64

The Modal launcher pins one seed (`0`), fresh LoRA initialization, 10,000
training examples, global batch size 64, gradient accumulation 4, and four
A100-80GB GPUs. It trains all 14 missing points, evaluates all six tasks, and
generates the CSV/JSON/PDF/LaTeX/Markdown artifacts:

```powershell
$env:PYTHONUTF8 = "1"
modal run --detach --quiet modal_sensitivity.py --mode full
```

The launcher uses a durable `spawn()` invocation, commits the Modal Volume
after every trained/evaluated setting, and resumes completed checkpoints and
evaluation JSON files on a later invocation.

## 4. Evaluate final checkpoints

The evaluator uses only each run's final `LATEST` checkpoint. LoRA adapters are
automatically merged with the base model recorded in that run's config and the
temporary merged model is removed after all six tasks for the run finish.

```bash
python evaluate_sensitivity.py \
  --manifest outputs/sensitivity/run_manifest.csv \
  --tensor_parallel_size 2 \
  --dtype bfloat16 \
  --gpu_memory_utilization 0.90 \
  --max_model_len 4096 \
  --batch_size auto:4 \
  --log_samples
```

Print the commands without requiring checkpoints or GPUs:

```bash
python evaluate_sensitivity.py \
  --manifest outputs/sensitivity/run_manifest.csv \
  --max_runs 1 \
  --dry_run
```

The frozen protocol is HellaSwag 10-shot, ARC-Challenge 25-shot, MMLU 5-shot,
TruthfulQA MC2 0-shot, WinoGrande 5-shot, and GSM8K 5-shot. Task scores and the
arithmetic six-task average are stored on a 0–100 scale. The exact harness
version, approved metric keys, few-shot settings, model backend, and inference
arguments are hashed in `evaluation_protocol.json`; the summarizer rejects
mixed hashes.

If using a newer harness CLI, override the command prefix, for example:

```bash
python evaluate_sensitivity.py \
  --manifest outputs/sensitivity/run_manifest.csv \
  --lm_eval_command "lm-eval run"
```

## 5. Aggregate, plot, and check the main result

Prepare a small JSON file with the published/main-run six-task average and the
variance tolerance justified by the paper's seed/evaluation protocol:

```json
{
  "average": 63.42,
  "expected_variance": 0.30
}
```

Then run:

```bash
python summarize_sensitivity.py \
  --manifest outputs/sensitivity/run_manifest.csv \
  --main_result path/to/main_result.json
```

You can instead pass `--expected_variance 0.30`. The script stops after writing
artifacts if the exact default sensitivity point differs from the main average
by more than this declared tolerance. Do not interpret the trends until that
check passes. Without `--main_result`, artifacts are produced but the Markdown
summary marks reproduction as `not_checked`.

Generated files include:

```text
outputs/sensitivity/
  sensitivity_all_runs.csv
  sensitivity_summary.csv
  sensitivity_summary.json
  sensitivity_T.pdf
  sensitivity_kappa.pdf
  sensitivity_delta0.pdf
  sensitivity_lambda.pdf
  sensitivity_tables.tex
  summary.md
```

`sensitivity_all_runs.csv` is regenerated from immutable per-run harness JSON,
and all reported averages are recomputed from the six task-level scores. Use
`--allow_missing` only for a clearly labeled partial diagnostic; the default
requires all manifested runs.

## Test without training

```bash
python -m unittest discover -s tests -v
```

The test suite covers the exact `lambda -> 0` generator, no-winsorization
behavior, one-factor config auditing, persisted label-noise replay, result
parsing, and aggregation in addition to the ARC-BPO loss tests.
