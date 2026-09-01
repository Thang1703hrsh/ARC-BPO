# ARC-BPO Hyperparameter Sensitivity — Codex Implementation Specification

## 1. Purpose

This document specifies exactly how to implement and run the hyperparameter-sensitivity experiments for ARC-BPO on **Llama-3-8B**.

The goal is to produce the missing entries for the following four sensitivity studies:

1. allocation temperature \(T\);
2. calibration margin \(\Delta_0\);
3. winsorization threshold \(\kappa\);
4. SBA exponent \(\lambda\).

Each experiment must vary **exactly one hyperparameter at a time** while keeping all remaining ARC-BPO settings identical to the final main-run configuration.

The resulting task-level scores must be reported on:

- HellaSwag
- ARC
- MMLU
- TruthfulQA
- Winogrande
- GSM8K
- Average over the six tasks

The default ARC-BPO configuration is:

```yaml
T: 2.0
delta0: 2.0
kappa: 2.0
lambda: 1.0
```

The main clean-data result at this default configuration is:

| HellaSwag | ARC | MMLU | TruthfulQA | Winogrande | GSM8K | Avg. |
|---:|---:|---:|---:|---:|---:|---:|
| 82.26 | 65.36 | 65.20 | 55.10 | 79.24 | 79.83 | 71.17 |

The already-established ARC-BPO result at **20% preference-label noise**, using the same default hyperparameters, is:

| HellaSwag | ARC | MMLU | TruthfulQA | Winogrande | GSM8K | Avg. |
|---:|---:|---:|---:|---:|---:|---:|
| 81.99 | 64.59 | 65.12 | 53.42 | 79.16 | 79.98 | 70.71 |

These default values are anchors and must not be changed.

---

## 2. Critical Consistency Requirement

The sensitivity experiments must reproduce the main ARC-BPO result at the default point.

For clean data:

```text
T = 2
Delta0 = 2
kappa = 2
lambda = 1
```

must correspond to:

```text
HellaSwag   = 82.26
ARC         = 65.36
MMLU        = 65.20
TruthfulQA  = 55.10
Winogrande  = 79.24
GSM8K       = 79.83
Average     = 71.17
```

For the 20% label-noise experiment at the default \(\kappa=2\):

```text
HellaSwag   = 81.99
ARC         = 64.59
MMLU        = 65.12
TruthfulQA  = 53.42
Winogrande  = 79.16
GSM8K       = 79.98
Average     = 70.71
```

If a newly rerun default experiment differs materially from these values, **do not silently overwrite the paper values**.

Instead, stop and diagnose the source of the mismatch, including:

- checkpoint mismatch;
- seed mismatch;
- evaluation harness mismatch;
- different model initialization;
- different training data;
- different number of training steps;
- different optimizer or learning rate;
- different chunking;
- different reference model;
- different label-noise realization.

Generate a diagnostic report before proceeding.

---

# 3. General Experimental Protocol

All sensitivity runs must use the same:

```text
backbone
training dataset
reference model
chunker
optimizer
learning rate
batch size
gradient accumulation
number of epochs
warmup
maximum sequence length
evaluation harness
evaluation task configuration
checkpoint selection rule
seed policy
```

as the final Llama-3-8B ARC-BPO main experiment.

Do not change any parameter that is unrelated to the sensitivity variable being studied.

Use a one-factor-at-a-time protocol:

```text
Experiment A: vary T only
Experiment B: vary Delta0 only
Experiment C: vary kappa only
Experiment D: vary lambda only
```

---

# 4. Existing Training Configuration

Unless the actual repository configuration explicitly differs, the paper currently describes the common training setup as:

```yaml
epochs: 1
optimizer: RMSProp
learning_rate: 5e-7
lr_schedule: cosine
warmup_ratio: 0.05
batch_size: 32
gradient_accumulation: 4
max_sequence_length: 2048
hardware: 4 x H100
```

Codex must first inspect the actual final ARC-BPO Llama-3-8B config and confirm that these values match the implementation.

Do not blindly hard-code these values if the final repository config differs.

---

# 5. Experiment A — Allocation Temperature T

## 5.1 Purpose

The allocation temperature controls the concentration of the softmax credit allocation.

Conceptually:

```text
larger T -> smoother / more uniform credit allocation
smaller T -> sharper / more concentrated credit allocation
```

The allocation is defined by the ARC-BPO credit-shape mechanism:

\[
\pi_i^w
=
\frac{\exp(\widetilde A_i^w/T)}
{\sum_k \exp(\widetilde A_k^w/T)}
\]

and analogously on the loser side.

## 5.2 Sweep

Run:

```text
T in {4, 2, 1, 0.5}
```

The default is:

```text
T = 2
```

The \(T=2\) result is already known from the main experiment and should be reused if the configuration is exactly identical.

Therefore, the missing runs are normally:

```text
T = 4
T = 1
T = 0.5
```

## 5.3 Fixed Parameters

For every T sweep run:

```yaml
delta0: 2.0
kappa: 2.0
lambda: 1.0
```

All other optimization and model settings must equal the main ARC-BPO Llama-3-8B configuration.

## 5.4 Required Output

For each T value store:

```text
HellaSwag
ARC
MMLU
TruthfulQA
Winogrande
GSM8K
Average
```

Recommended run names:

```text
sens_T_4
sens_T_2_default
sens_T_1
sens_T_05
```

---

# 6. Experiment B — Calibration Margin Delta0

## 6.1 Purpose

The finite calibration margin \(\Delta_0\) determines the total signed chunk-level credit budget.

ARC-BPO uses:

\[
\tau_i^w = \frac{\Delta_0}{2}\pi_i^w,
\qquad
\tau_j^l = -\frac{\Delta_0}{2}\rho_j^l.
\]

The signed sum must satisfy:

\[
\sum_i \tau_i^w - \sum_j \tau_j^l = \Delta_0.
\]

## 6.2 Sweep

Run:

```text
Delta0 in {0.5, 1, 2, 4}
```

Do **not** include:

```text
Hard
RM gap
```

in this sensitivity study.

The default is:

```text
Delta0 = 2
```

The \(\Delta_0=2\) point is already available from the main experiment.

Normally the missing runs are:

```text
Delta0 = 0.5
Delta0 = 1
Delta0 = 4
```

## 6.3 Fixed Parameters

For every Delta0 sweep run:

```yaml
T: 2.0
kappa: 2.0
lambda: 1.0
```

## 6.4 Required Output

For each Delta0 value store all six benchmark scores plus the six-task average.

Recommended run names:

```text
sens_delta0_05
sens_delta0_1
sens_delta0_2_default
sens_delta0_4
```

---

# 7. Experiment C — Winsorization Threshold kappa

## 7.1 Purpose

The winsorization threshold \(\kappa\) controls clipping of anomalous detached advantage scores before the allocation softmax.

ARC-BPO uses:

\[
\widetilde A_i
=
\operatorname{clip}
\left(
\widehat A_i;
\bar A-\kappa\widehat\sigma,
\bar A+\kappa\widehat\sigma
\right).
\]

Smaller \(\kappa\):

```text
stronger clipping
```

Larger \(\kappa\):

```text
weaker clipping
```

This is the only hyperparameter sweep in the current plan that must be evaluated under both:

```text
clean preference data
20% preference-label noise
```

## 7.2 Sweep

Run:

```text
kappa in {3, 2, 1.5, 1}
```

Do **not** include:

```text
kappa -> infinity
```

in the current experiment.

Default:

```text
kappa = 2
```

## 7.3 Clean Sweep

Use:

```yaml
T: 2.0
delta0: 2.0
lambda: 1.0
noise_rate: 0.0
```

Known default result:

```text
kappa = 2
Avg = 71.17
```

Normally the missing clean runs are:

```text
kappa = 3
kappa = 1.5
kappa = 1
```

## 7.4 20% Noise Sweep

Use:

```yaml
T: 2.0
delta0: 2.0
lambda: 1.0
noise_rate: 0.20
```

Known default result:

```text
kappa = 2
HellaSwag   = 81.99
ARC         = 64.59
MMLU        = 65.12
TruthfulQA  = 53.42
Winogrande  = 79.16
GSM8K       = 79.98
Average     = 70.71
```

Normally the missing 20%-noise runs are:

```text
kappa = 3
kappa = 1.5
kappa = 1
```

---

# 8. Preference-Label Noise Protocol

For the 20% noisy-training experiments:

1. start from the exact same clean preference-training dataset;
2. sample exactly 20% of training preference pairs;
3. swap winner and loser for those selected pairs;
4. corrupt training data only;
5. leave evaluation data untouched;
6. use one fixed corruption seed;
7. save the exact indices of corrupted examples;
8. reuse the identical corrupted dataset for all \(\kappa\) settings.

Pseudo-code:

```python
rng = Random(NOISE_SEED)

num_noisy = round(0.20 * len(train_pairs))
noise_indices = rng.sample(range(len(train_pairs)), num_noisy)

for idx in noise_indices:
    pair = train_pairs[idx]
    pair.chosen, pair.rejected = pair.rejected, pair.chosen
```

Save the corruption indices:

```text
outputs/sensitivity/noise20_indices.json
```

Do not generate a new 20% corruption pattern for every kappa value.

That would confound parameter sensitivity with different noisy datasets.

---

# 9. Experiment D — SBA Exponent lambda

## 9.1 Purpose

The SBA exponent \(\lambda\) controls ratio-dependent gradient scaling induced by the Bregman generator.

The ARC-BPO derivative contains:

\[
e^{-\lambda a_\theta(c_i)}
\]

so changing \(\lambda\) changes the geometry / scaling of the loss.

## 9.2 Sweep

Run only:

```text
lambda in {0.5, 1, 2}
```

Do **not** run:

```text
lambda -> 0
```

for the current paper sensitivity experiment.

Do **not** add a 20% noise sweep for lambda.

The lambda experiment uses **clean preference data only**.

Default:

```text
lambda = 1
```

The \(\lambda=1\) result is already available from the main experiment.

Normally the only missing runs are:

```text
lambda = 0.5
lambda = 2
```

## 9.3 Fixed Parameters

```yaml
T: 2.0
delta0: 2.0
kappa: 2.0
noise_rate: 0.0
```

Recommended run names:

```text
sens_lambda_05
sens_lambda_1_default
sens_lambda_2
```

---

# 10. Total New Training Runs

If all existing default runs are reusable, the new sensitivity training runs are:

## T

```text
3 new clean runs
T = 4
T = 1
T = 0.5
```

## Delta0

```text
3 new clean runs
Delta0 = 0.5
Delta0 = 1
Delta0 = 4
```

## kappa

Clean:

```text
3 new runs
kappa = 3
kappa = 1.5
kappa = 1
```

20% noise:

```text
3 new runs
kappa = 3
kappa = 1.5
kappa = 1
```

## lambda

```text
2 new clean runs
lambda = 0.5
lambda = 2
```

Total:

```text
3 + 3 + 3 + 3 + 2 = 14 new training runs
```

This assumes that:

```text
default clean ARC-BPO run
default 20%-noise ARC-BPO run
```

can be reused.

If they cannot be reused because the experimental protocol differs, Codex must report why before launching redundant runs.

---

# 11. Seed Policy

Preferred option:

```text
use the same seed policy as the final main experiments
```

If the main experiment uses one fixed seed for this analysis, use exactly that seed for all sensitivity values.

If multiple seeds are used, use the exact same seed set for every value in a sweep.

Never do:

```text
T=4 using seed A
T=1 using seed B
T=0.5 using seed C
```

unless all settings are averaged over the same matched seed set.

---

# 12. Checkpoint Policy

Use one consistent checkpoint-selection rule for every sensitivity experiment.

Preferred:

```text
final checkpoint after the same training schedule
```

Do not cherry-pick the best checkpoint independently for different parameter values.

Store:

```text
run_name
parameter
parameter_value
seed
checkpoint_path
global_step
epoch
```

for every run.

---

# 13. Evaluation Protocol

Use exactly the same evaluation implementation and settings as the Llama-3-8B main ARC-BPO result.

Evaluate:

```text
HellaSwag
ARC
MMLU
TruthfulQA
Winogrande
GSM8K
```

Compute:

\[
\mathrm{Avg}
=
\frac{
\mathrm{HellaSwag}
+\mathrm{ARC}
+\mathrm{MMLU}
+\mathrm{TruthfulQA}
+\mathrm{Winogrande}
+\mathrm{GSM8K}
}{6}.
\]

Round only for final presentation.

Recommended:

```python
average = sum(task_scores) / 6.0
```

Keep full precision internally.

Report two decimals in the paper.

---

# 14. Output Schema

Create:

```text
outputs/
└── sensitivity/
    ├── runs/
    ├── logs/
    ├── configs/
    ├── sensitivity_results.csv
    ├── sensitivity_results.json
    ├── noise20_indices.json
    ├── default_reproduction_check.json
    └── summary.md
```

Recommended CSV schema:

```csv
sweep,parameter,value,noise_rate,seed,checkpoint,hellaswag,arc,mmlu,truthfulqa,winogrande,gsm8k,average
```

Example:

```csv
T,T,2,0.0,42,...,82.26,65.36,65.20,55.10,79.24,79.83,71.17
kappa,kappa,2,0.2,42,...,81.99,64.59,65.12,53.42,79.16,79.98,70.71
```

---

# 15. Suggested Launcher Structure

Create a launcher such as:

```text
run_sensitivity.py
```

Pseudo-code:

```python
DEFAULTS = {
    "T": 2.0,
    "delta0": 2.0,
    "kappa": 2.0,
    "lambda": 1.0,
}

SWEEPS = {
    "T": [4.0, 2.0, 1.0, 0.5],
    "delta0": [0.5, 1.0, 2.0, 4.0],
    "lambda": [0.5, 1.0, 2.0],
}

KAPPA_SWEEP = [3.0, 2.0, 1.5, 1.0]
```

Then:

```python
for sweep_name, values in SWEEPS.items():
    for value in values:
        cfg = load_final_llama3_arcbpo_config()

        # restore exact defaults
        cfg.T = 2.0
        cfg.delta0 = 2.0
        cfg.kappa = 2.0
        cfg.lambda_ = 1.0
        cfg.noise_rate = 0.0

        # vary exactly one parameter
        setattr(cfg, sweep_name, value)

        launch_and_evaluate(cfg)
```

For kappa:

```python
for noise_rate in [0.0, 0.20]:
    for kappa in [3.0, 2.0, 1.5, 1.0]:
        cfg = load_final_llama3_arcbpo_config()

        cfg.T = 2.0
        cfg.delta0 = 2.0
        cfg.kappa = kappa
        cfg.lambda_ = 1.0
        cfg.noise_rate = noise_rate

        if noise_rate == 0.20:
            cfg.noise_indices_path = (
                "outputs/sensitivity/noise20_indices.json"
            )

        launch_and_evaluate(cfg)
```

---

# 16. Reuse Existing Default Runs

Before launching a new job, check whether an exact matching run already exists.

Match on:

```text
model
training dataset
reference model
seed
epochs
optimizer
learning rate
batch size
gradient accumulation
chunker
T
Delta0
kappa
lambda
noise rate
noise indices
evaluation harness
```

If an exact match exists, reuse its result.

In particular, try to reuse:

```text
clean:
T=2, Delta0=2, kappa=2, lambda=1

20% noise:
T=2, Delta0=2, kappa=2, lambda=1
```

---

# 17. Sanity Checks

Before accepting any result, verify:

```text
1. Exactly one hyperparameter changes within each sweep.
2. All non-swept settings match the final main config.
3. Same model initialization protocol.
4. Same training dataset.
5. Same chunker.
6. Same reference model.
7. Same optimizer.
8. Same learning rate.
9. Same training length.
10. Same evaluation harness.
11. Same checkpoint-selection rule.
12. Same seed policy.
13. Same 20% corruption indices for every noisy kappa run.
14. Default clean point matches Avg = 71.17.
15. Default noisy kappa point matches Avg = 70.71.
16. Average is recomputed from all six task scores.
17. No task is silently omitted from the average.
```

---

# 18. Do Not Fabricate Missing Values

The empty cells in the manuscript tables are intentionally empty.

Codex must **never**:

- interpolate values;
- simulate values;
- infer values from nearby settings;
- smooth a sensitivity curve;
- choose values to create a desirable trend;
- modify real results to make the default point best;
- reuse a score from a different hyperparameter setting;
- reuse clean results as noisy results;
- report an average without actual task-level evaluation.

Only actual experiment outputs may populate the missing cells.

---

# 19. Desired Final Tables

## 19.1 Allocation Temperature T

Populate:

```text
T = 4
T = 2       [known anchor]
T = 1
T = 0.5
```

Columns:

```text
Hella.
ARC
MMLU
Truth.
Wino.
GSM8K
Avg.
```

Known row:

```text
T = 2
82.26 | 65.36 | 65.20 | 55.10 | 79.24 | 79.83 | 71.17
```

---

## 19.2 Calibration Margin Delta0

Populate:

```text
Delta0 = 0.5
Delta0 = 1
Delta0 = 2   [known anchor]
Delta0 = 4
```

Known row:

```text
Delta0 = 2
82.26 | 65.36 | 65.20 | 55.10 | 79.24 | 79.83 | 71.17
```

---

## 19.3 Winsorization Threshold kappa

### Clean block

Populate:

```text
kappa = 3
kappa = 2       [known anchor]
kappa = 1.5
kappa = 1
```

Known clean row:

```text
kappa = 2
82.26 | 65.36 | 65.20 | 55.10 | 79.24 | 79.83 | 71.17
```

### 20% noise block

Populate:

```text
kappa = 3
kappa = 2       [known anchor]
kappa = 1.5
kappa = 1
```

Known 20%-noise row:

```text
kappa = 2
81.99 | 64.59 | 65.12 | 53.42 | 79.16 | 79.98 | 70.71
```

---

## 19.4 SBA Exponent lambda

Populate:

```text
lambda = 0.5
lambda = 1      [known anchor]
lambda = 2
```

Known row:

```text
lambda = 1
82.26 | 65.36 | 65.20 | 55.10 | 79.24 | 79.83 | 71.17
```

---

# 20. Generate LaTeX Automatically

After all runs finish, generate the final LaTeX rows automatically from `sensitivity_results.csv`.

Do not manually type experimental scores into the final tables if they can be generated programmatically.

Recommended helper:

```text
generate_sensitivity_tables.py
```

It should output:

```text
outputs/sensitivity/table_temperature.tex
outputs/sensitivity/table_margin.tex
outputs/sensitivity/table_kappa.tex
outputs/sensitivity/table_lambda.tex
```

The default row should be highlighted using:

```latex
\rowcolor{gray!20}
```

and the default average should use:

```latex
\textbf{...}
```

Do not automatically bold the highest score of every non-default row unless the manuscript explicitly adopts that convention.

---

# 21. Final Summary File

Create:

```text
outputs/sensitivity/summary.md
```

It should contain:

```markdown
# ARC-BPO Sensitivity Results

## Default Reproduction
- Main clean average: 71.17
- Reproduced clean average: ...
- Absolute difference: ...

## Allocation Temperature T
| T | HellaSwag | ARC | MMLU | TruthfulQA | Winogrande | GSM8K | Avg |
|---|---:|---:|---:|---:|---:|---:|---:|
...

## Calibration Margin Delta0
...

## Winsorization Threshold kappa — Clean
...

## Winsorization Threshold kappa — 20% Noise
...

## SBA Exponent lambda
...

## Configuration Checks
- Same backbone: yes/no
- Same reference model: yes/no
- Same optimizer: yes/no
- Same seed policy: yes/no
- Same evaluation protocol: yes/no
- Same noisy indices across kappa runs: yes/no

## Notes
...
```

---

# 22. Interpretation Rules

Do not write the paper conclusion before the real sweep values are available.

Only make statements supported by the results.

Examples:

### T

Only say:

```text
moderate concentration is preferable
```

if the real results show that moderate T values outperform both smaller and larger values.

### Delta0

Only claim:

```text
performance is stable across finite calibration margins
```

if the actual scores are indeed stable.

### kappa

Only claim:

```text
moderate winsorization improves robustness
```

if the 20%-noise results support this conclusion.

### lambda

Only claim:

```text
ARC-BPO is insensitive to the Bregman-loss geometry
```

if \(\lambda=0.5,1,2\) perform sufficiently similarly.

If the sensitivity curve is unstable, report that honestly.

---

# 23. Expected Manuscript Structure

The final subsection is intended to support the following structure:

```text
Hyperparameter Sensitivity
|
|-- Allocation temperature T
|   `-- Table: T x six tasks + Avg.
|
|-- Calibration margin Delta0
|   `-- Table: Delta0 x six tasks + Avg.
|
|-- Winsorization threshold kappa
|   `-- Table:
|       |-- clean block
|       `-- 20% noise block
|
`-- SBA exponent lambda
    `-- Table: lambda x six tasks + Avg.
```

The clean default row in every table must correspond to the same ARC-BPO main result:

```text
Avg = 71.17
```

---

# 24. Completion Criteria

The sensitivity implementation is complete only when:

- [ ] T sweep is complete.
- [ ] Delta0 sweep is complete.
- [ ] kappa clean sweep is complete.
- [ ] kappa 20%-noise sweep is complete.
- [ ] lambda clean sweep is complete.
- [ ] every run has six task-level metrics.
- [ ] every average is recomputed from the six task scores.
- [ ] default clean point is consistent with 71.17.
- [ ] default noisy kappa point is consistent with 70.71.
- [ ] all configs are archived.
- [ ] all noisy kappa runs use the same corruption indices.
- [ ] final CSV and JSON are generated.
- [ ] final LaTeX tables are generated.
- [ ] `summary.md` is generated.
- [ ] no missing value is fabricated.
