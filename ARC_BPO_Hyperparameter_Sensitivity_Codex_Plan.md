# ARC-BPO Hyperparameter Sensitivity — Codex Implementation Plan

## Reviewer Comment

> **Comment 2:** Please provide sensitivity analysis for important hyperparameters, like the allocation temperature, winsorization threshold, calibration margin, and Bregman-loss parameter.

---

# 1. Objective

The reviewer asks for a controlled sensitivity analysis of the four ARC-BPO-specific hyperparameters:

- allocation temperature \(T\),
- winsorization threshold \(\kappa\),
- calibration margin \(\Delta_0\),
- Bregman / SBA exponent \(\lambda\).

The goal is not simply to find the best value for each hyperparameter.

The analysis should demonstrate that:

1. ARC-BPO works over a reasonable range of values;
2. the chosen defaults lie in a stable region rather than at an isolated optimum;
3. each parameter has the expected qualitative effect;
4. conclusions do not depend on a narrowly tuned setting.

---

# 2. Critical Experimental Requirement

## The default sensitivity point must be consistent with the main ARC-BPO experiment

Before running any sensitivity study, Codex must identify the exact final configuration used for the main ARC-BPO Llama-3-8B experiment.

Record:

```text
model checkpoint / initialization
reference model
dataset
training split
evaluation split
number of training epochs / steps
learning rate
batch size
gradient accumulation
beta
T
kappa
Delta0
lambda
chunker
random seed(s)
generation/evaluation settings
```

The sensitivity run at the exact default configuration must reproduce the main result within the expected seed/evaluation variance.

If the sensitivity analysis intentionally uses a different protocol, such as:

```text
single seed instead of multi-seed mean
different checkpoint selection
smaller validation subset
shorter training schedule
different evaluation split
```

then this difference must be explicitly documented in the paper and rebuttal.

Do not silently report a default sensitivity point that is inconsistent with the main result.

---

# 3. Experimental Design

Use a one-factor-at-a-time protocol.

For each hyperparameter:

```text
vary only the parameter under study
keep all other ARC-BPO settings fixed at the final default values
```

Recommended structure:

```text
Experiment A: vary T
Experiment B: vary kappa
Experiment C: vary Delta0
Experiment D: vary lambda
```

All experiments should use the same:

```text
backbone
training data
evaluation benchmark
training schedule
optimizer
chunker
reference model
seed policy
evaluation code
```

---

# 4. Recommended Backbone

Use the same backbone already used in the sensitivity analysis / main paper.

Recommended:

```text
Llama-3-8B
```

because:

- it is already a main experimental backbone,
- it is large enough to be representative,
- running all four sweeps on every backbone is unnecessary unless compute is abundant.

The paper can state that sensitivity is evaluated on one representative backbone while the main performance results cover all backbones.

---

# 5. Default Configuration

Codex should read the actual experiment config used for the final ARC-BPO run.

Example only:

```yaml
T: 2.0
kappa: 2.0
Delta0: 2.0
lambda: 1.0
```

Do not assume these values if the final training config differs.

Save the resolved defaults to:

```text
outputs/sensitivity/default_config.json
```

---

# 6. Sweep A — Allocation Temperature T

## Purpose

\(T\) controls how concentrated the chunk-credit allocation is.

Qualitatively:

```text
large T  -> allocation approaches uniform
small T  -> allocation becomes more concentrated
```

Recommended sweep:

```text
T in {infinity, 4, 2, 1, 0.5}
```

Operationally, do not literally set `T = infinity`.

Implement the uniform limit explicitly:

```python
if mode == "uniform":
    allocation = torch.ones_like(scores) / len(scores)
else:
    allocation = torch.softmax(scores / T, dim=-1)
```

For loser chunks use the corresponding sign convention from ARC-BPO.

Recommended command pattern:

```bash
python train_arcbpo.py \
    --config configs/arcbpo_llama3_8b.yaml \
    --allocation_temperature 4.0 \
    --run_name sens_T_4
```

Repeat for all values.

---

# 7. Sweep B — Winsorization Threshold kappa

## Purpose

\(\kappa\) controls how aggressively extreme chunk-credit scores are clipped.

Recommended sweep:

```text
kappa in {infinity, 3, 2, 1.5, 1}
```

Operationally:

```text
kappa = infinity
```

must mean:

```text
disable winsorization
```

rather than attempting to use a numerical infinity in clipping code.

Recommended implementation:

```python
if args.disable_winsorization:
    scores_clipped = scores
else:
    scores_clipped = winsorize(scores, kappa=args.kappa)
```

This experiment should be run on:

```text
clean preference data
20% preference-label noise
```

because winsorization is intended primarily as a robustness mechanism.

Recommended command examples:

```bash
python train_arcbpo.py \
    --config configs/arcbpo_llama3_8b.yaml \
    --disable_winsorization \
    --run_name sens_kappa_inf_clean
```

```bash
python train_arcbpo.py \
    --config configs/arcbpo_llama3_8b.yaml \
    --kappa 2.0 \
    --label_noise 0.20 \
    --run_name sens_kappa_2_noise20
```

---

# 8. Sweep C — Calibration Margin Delta0

## Purpose

\(\Delta_0\) determines the finite response-level margin budget distributed across chunks.

Recommended finite sweep:

```text
Delta0 in {0.5, 1, 2, 4}
```

If the paper also compares:

```text
Hard
Reward-model-derived margin
```

treat these as separate calibration strategies rather than pretending they are numeric values of \(\Delta_0\).

Recommended labels:

```text
Hard-label BT
RM-derived
Delta0 = 0.5
Delta0 = 1
Delta0 = 2
Delta0 = 4
```

Important:

- do not call the finite calibration parameter \(\Delta^\star\);
- use \(\Delta_0\) consistently;
- document exactly how the hard-label setting is implemented;
- document which reward model is used for the RM-derived margin.

Recommended commands:

```bash
python train_arcbpo.py \
    --config configs/arcbpo_llama3_8b.yaml \
    --delta0 0.5 \
    --run_name sens_delta0_05
```

```bash
python train_arcbpo.py \
    --config configs/arcbpo_llama3_8b.yaml \
    --delta0 4.0 \
    --run_name sens_delta0_4
```

---

# 9. Sweep D — Bregman / SBA Exponent lambda

## Purpose

\(\lambda\) controls the geometry / gradient scaling induced by the SBA Bregman generator.

Recommended sweep:

```text
lambda in {lambda -> 0, 0.5, 1, 2}
```

The `lambda -> 0` condition must use the mathematically correct limiting implementation used by ARC-BPO.

Do not approximate the limit with an arbitrary tiny number unless the training code already defines and validates that approximation.

Prefer an explicit mode such as:

```python
if lambda_mode == "limit_zero":
    loss = kliep_limit_loss(...)
else:
    loss = sba_loss(..., lambda_value)
```

Run this sweep on:

```text
clean preference data
20% preference-label noise
```

to show whether the loss-shape parameter behaves differently under corruption.

---

# 10. Label-Noise Protocol

Use the same label-flipping protocol already used elsewhere in the ARC-BPO robustness experiment.

For a noise rate:

```text
p = 0.20
```

randomly select 20% of training preference pairs and swap:

```text
winner <-> loser
```

Important:

- corrupt the training set only;
- keep evaluation data clean;
- use a fixed corruption seed;
- save the exact corrupted-pair indices;
- reuse the same corrupted dataset across all \(\kappa\) and \(\lambda\) runs.

Save:

```text
outputs/sensitivity/noise20_indices.json
```

This prevents different noise realizations from confounding the sensitivity comparison.

---

# 11. Seeds

Best option:

```text
3 seeds per setting
```

Example:

```text
42
123
2026
```

If compute is limited, use:

```text
1 fixed seed for all sensitivity points
```

but explicitly state this in the paper/rebuttal.

Do not use different seeds for different hyperparameter values unless you average over a matched seed set.

For a controlled sweep:

```text
same parameter grid
same seed set
same data
same checkpoint initialization protocol
```

---

# 12. Evaluation Metric

Use exactly the same aggregate evaluation metric as the main ARC-BPO paper.

For Open LLM Leaderboard-style evaluation, store all task-level values and the final average.

Recommended CSV columns:

```text
sweep
parameter
value
seed
hellaswag
arc
mmlu
truthfulqa
winogrande
gsm8k
average
noise_rate
checkpoint
```

Example:

```csv
sweep,parameter,value,seed,hellaswag,arc,mmlu,truthfulqa,winogrande,gsm8k,average,noise_rate
T,T,2,42,...,...,...,...,...,...,...,0.0
```

---

# 13. Output Directory Structure

Recommended:

```text
outputs/sensitivity/
├── default_config.json
├── T/
│   ├── inf/
│   ├── 4/
│   ├── 2/
│   ├── 1/
│   └── 0.5/
├── kappa/
│   ├── clean/
│   └── noise20/
├── delta0/
│   ├── hard/
│   ├── rm/
│   ├── 0.5/
│   ├── 1/
│   ├── 2/
│   └── 4/
├── lambda/
│   ├── clean/
│   └── noise20/
├── sensitivity_all_runs.csv
├── sensitivity_summary.csv
├── sensitivity_summary.json
├── sensitivity_T.pdf
├── sensitivity_kappa.pdf
├── sensitivity_delta0.pdf
├── sensitivity_lambda.pdf
└── summary.md
```

---

# 14. Aggregation

If multiple seeds are used, compute:

```text
mean
standard deviation
```

for each setting.

Recommended final summary format:

```text
mean ± std
```

Do not average only the final averages if the manuscript reports task-level metrics; store both task-level and aggregate values.

---

# 15. Recommended Final Tables

## Table A — T and Delta0

```latex
\begin{table*}[t]
\centering
\small
\renewcommand{\arraystretch}{1.10}

\begin{minipage}[t]{0.47\textwidth}
\vspace{0pt}
\centering
\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}lccccc@{}}
\toprule
$T$ & $\infty$ & $4$ & $\mathbf{2}$ & $1$ & $0.5$ \\
\midrule
Avg. & [V] & [V] & [V] & [V] & [V] \\
\bottomrule
\end{tabular*}

\vspace{4pt}
{\small (a) Allocation temperature $T$.}
\end{minipage}
\hfill
\begin{minipage}[t]{0.47\textwidth}
\vspace{0pt}
\centering
\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}lcccccc@{}}
\toprule
Margin & Hard & RM & $0.5$ & $1$ & $\mathbf{2}$ & $4$ \\
\midrule
Avg. & [V] & [V] & [V] & [V] & [V] & [V] \\
\bottomrule
\end{tabular*}

\vspace{4pt}
{\small (b) Calibration margin $\Delta_0$.}
\end{minipage}

\caption{Sensitivity of ARC-BPO to the allocation temperature $T$ and calibration margin $\Delta_0$ on Llama-3-8B. The remaining ARC-BPO hyperparameters are kept fixed while varying the parameter under study. RM denotes a reward-model-derived calibration margin.}
\label{tab:sensitivity_T_margin}
\end{table*}
```

---

## Table B — kappa and lambda

```latex
\begin{table*}[t]
\centering
\small
\renewcommand{\arraystretch}{1.10}

\begin{minipage}[t]{0.47\textwidth}
\vspace{0pt}
\centering
\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}lccccc@{}}
\toprule
$\kappa$ & $\infty$ & $3$ & $\mathbf{2}$ & $1.5$ & $1$ \\
\midrule
Clean & [V] & [V] & [V] & [V] & [V] \\
20\% noise & [V] & [V] & [V] & [V] & [V] \\
\bottomrule
\end{tabular*}

\vspace{4pt}
{\small (a) Winsorization threshold $\kappa$.}
\end{minipage}
\hfill
\begin{minipage}[t]{0.47\textwidth}
\vspace{0pt}
\centering
\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}lcccc@{}}
\toprule
$\lambda$ & $\to0$ & $0.5$ & $\mathbf{1}$ & $2$ \\
\midrule
Clean & [V] & [V] & [V] & [V] \\
20\% noise & [V] & [V] & [V] & [V] \\
\bottomrule
\end{tabular*}

\vspace{4pt}
{\small (b) SBA exponent $\lambda$.}
\end{minipage}

\caption{Sensitivity of ARC-BPO to the winsorization threshold $\kappa$ and SBA exponent $\lambda$ on Llama-3-8B. For $\kappa$, $\kappa\rightarrow\infty$ denotes no winsorization. We additionally report results under $20\%$ preference-label noise.}
\label{tab:sensitivity_kappa_lambda}
\end{table*}
```

---

# 16. Recommended Plots

Generate four plots:

```text
T vs average score
Delta0 vs average score
kappa vs average score
lambda vs average score
```

For \(\kappa\) and \(\lambda\):

```text
one curve = clean
one curve = 20% noise
```

If multiple seeds are used:

```text
error bars = standard deviation
```

These plots are mainly for internal validation / appendix; the main paper can use compact tables.

---

# 17. Sanity Checks

Codex must verify:

```text
1. exactly one hyperparameter changes within each sweep;
2. all remaining config values are identical;
3. same dataset split across all runs;
4. same corruption indices for all noisy runs;
5. same evaluation script across all runs;
6. same seed set across each parameter grid;
7. same model initialization protocol;
8. default sensitivity point matches the final main configuration;
9. no accidental change in beta, learning rate, batch size, epochs, or chunker;
10. no evaluation checkpoint cherry-picking.
```

Log a config hash for every run.

Example:

```python
import hashlib, json

config_hash = hashlib.sha256(
    json.dumps(config, sort_keys=True).encode()
).hexdigest()
```

Store the hash with each result.

---

# 18. Default-Point Reproduction Check

Before interpreting sensitivity, run:

```text
exact default ARC-BPO configuration
```

and compare against the main paper result.

Output:

```text
main_reported_average
sensitivity_default_average
absolute_difference
```

If:

```text
difference > expected seed/evaluation variance
```

stop and identify the reason.

Possible legitimate causes:

```text
different seed aggregation
different checkpoint
different evaluation split
different number of training steps
different leaderboard harness version
```

Document the actual cause.

Do not simply keep both numbers without explanation.

---

# 19. Suggested Automation Script

Create a launcher:

```text
run_sensitivity.py
```

Pseudo-structure:

```python
SWEEPS = {
    "T": ["uniform", 4.0, 2.0, 1.0, 0.5],
    "kappa": ["disabled", 3.0, 2.0, 1.5, 1.0],
    "delta0": ["hard", "rm", 0.5, 1.0, 2.0, 4.0],
    "lambda": ["limit_zero", 0.5, 1.0, 2.0],
}

for sweep_name, values in SWEEPS.items():
    for value in values:
        for seed in SEEDS:
            cfg = load_default_config()
            cfg = change_exactly_one_parameter(cfg, sweep_name, value)
            launch_training(cfg)
            evaluate_checkpoint(cfg)
            save_result(cfg)
```

For \(\kappa\) and \(\lambda\):

```python
for noise_rate in [0.0, 0.20]:
    ...
```

---

# 20. Result Summary Script

Create:

```text
summarize_sensitivity.py
```

It should:

1. load all run-level CSV/JSON results;
2. verify config consistency;
3. group by hyperparameter value;
4. compute mean/std;
5. produce final tables;
6. produce plots;
7. write a human-readable summary.

Output:

```text
summary.md
```

Example structure:

```markdown
# Sensitivity Summary

## T
Best: ...
Stable region: ...
Uniform limit: ...
Most concentrated setting: ...

## kappa
Clean:
...
20% noise:
...

## Delta0
...

## lambda
...
```

---

# 21. Interpretation Logic

## T

Expected qualitative story:

```text
T -> infinity:
    nearly uniform allocation

moderate T:
    useful credit concentration

very small T:
    excessive concentration
```

Reviewer-facing interpretation:

> Moderate temperatures perform best, while both nearly uniform and overly concentrated allocations are slightly worse, suggesting that ARC-BPO benefits from selective but not excessively sharp credit assignment.

Use only if actual results support it.

---

## kappa

Expected qualitative story:

```text
kappa -> infinity:
    no winsorization

moderate kappa:
    robust to outliers

very small kappa:
    over-clipping
```

Reviewer-facing interpretation:

> Moderate winsorization preserves clean-data performance while improving robustness under corrupted preference labels; overly aggressive clipping begins to remove useful credit variation.

Use only if supported.

---

## Delta0

Expected qualitative story:

```text
too small:
    weak preference separation

moderate:
    good calibration

too large:
    over-aggressive target margin
```

Reviewer-facing interpretation:

> Performance remains stable across a range of finite margins, with a moderate fixed margin providing the best trade-off while avoiding the need for an additional reward model.

Use only if supported.

---

## lambda

Expected qualitative story:

```text
reasonable range:
    stable performance

different lambda:
    different Bregman gradient geometry
```

Reviewer-facing interpretation:

> ARC-BPO remains stable across the tested SBA exponents, indicating that the method is not tied to a narrowly tuned Bregman geometry.

Use only if supported.

---

# 22. Recommended Paper Paragraph

Once actual results are available:

```latex
\subsection{Hyperparameter Sensitivity}
\label{subsec:hyperparameter_sensitivity}

ARC-BPO introduces four principal hyperparameters beyond standard optimization settings: the allocation temperature $T$, winsorization threshold $\kappa$, calibration margin $\Delta_0$, and SBA exponent $\lambda$. We vary each parameter independently on Llama-3-8B while keeping the remaining ARC-BPO configuration fixed. For $\kappa$ and $\lambda$, which directly affect robustness to extreme credit scores and loss geometry, respectively, we additionally evaluate performance under $20\%$ preference-label noise.

For the allocation temperature, [INSERT ACTUAL RESULT AND INTERPRETATION]. For the calibration margin, [INSERT ACTUAL RESULT AND INTERPRETATION]. For the winsorization threshold, [INSERT ACTUAL CLEAN/NOISE RESULT]. For the SBA exponent, [INSERT ACTUAL CLEAN/NOISE RESULT].

Overall, the sensitivity results show that ARC-BPO remains competitive over a broad range of parameter values rather than relying on a single sharply tuned configuration. The selected defaults lie in stable regions of the corresponding sweeps.
```

---

# 23. Recommended Rebuttal Response

```latex
\reviewer{\textbf{Comment 2:} Please provide sensitivity analysis for important hyperparameters, like the allocation temperature, winsorization threshold, calibration margin, and Bregman-loss parameter.}

\noindent\textbf{Response.}
We thank the reviewer for this helpful suggestion. We have added a dedicated sensitivity analysis for all four ARC-BPO-specific hyperparameters highlighted in the comment: the allocation temperature $T$, winsorization threshold $\kappa$, calibration margin $\Delta_0$, and Bregman/SBA exponent $\lambda$. We vary each parameter independently on Llama-3-8B while keeping the remaining ARC-BPO configuration fixed. For the robustness-related parameters $\kappa$ and $\lambda$, we additionally report results under $20\%$ preference-label noise.

[INSERT FINAL TABLE]

The allocation-temperature results show that [INSERT]. The calibration-margin analysis shows that [INSERT]. For winsorization, [INSERT]. Finally, the SBA-exponent sweep shows that [INSERT].

Overall, these results indicate that ARC-BPO does not depend on a narrowly tuned hyperparameter configuration. The final default values lie within stable performance regions of the corresponding sensitivity analyses.
```

---

# 24. What Not to Do

Do not:

```text
- fabricate or smooth sensitivity values;
- choose different seeds for different values;
- cherry-pick the best checkpoint for some settings but final checkpoint for others;
- change more than one hyperparameter at once;
- use different noise realizations across kappa/lambda values;
- silently report a sensitivity default inconsistent with the main result;
- call the calibration margin Delta* if the implementation uses finite Delta0;
- claim hyperparameter robustness if the actual curves are highly unstable.
```

---

# 25. Minimum Version That Is Sufficient for the Reviewer

If compute is limited, run:

```text
T:
    infinity, 4, 2, 1, 0.5

kappa:
    infinity, 3, 2, 1.5, 1
    clean + 20% noise

Delta0:
    0.5, 1, 2, 4
    optionally Hard and RM-derived if already implemented

lambda:
    limit-zero, 0.5, 1, 2
    clean + 20% noise
```

Use:

```text
one fixed seed
same evaluation protocol
same training schedule
```

and explicitly state that the sensitivity study is a controlled single-seed analysis if that is the case.

---

# 26. Stronger Version

If compute permits:

```text
3 matched seeds per setting
mean ± std
same corruption realization per seed
```

This is the strongest reviewer-facing version.

---

# 27. Final Acceptance Checklist

Before using the results in the paper, verify:

```text
[ ] T sweep completed
[ ] kappa clean sweep completed
[ ] kappa 20% noise sweep completed
[ ] Delta0 sweep completed
[ ] lambda clean sweep completed
[ ] lambda 20% noise sweep completed
[ ] exact default config reproduced
[ ] main-vs-sensitivity default consistency checked
[ ] all runs use same evaluation code
[ ] all results saved at task level
[ ] aggregate means recomputed automatically
[ ] no manually edited result values
[ ] plots and tables generated from raw output files
[ ] paper wording matches actual trends
[ ] rebuttal wording matches actual trends
```

---

# 28. Recommended Final Deliverables from Codex

Codex should finish by producing:

```text
1. train/run launcher for all sweeps
2. evaluation launcher
3. aggregated CSV
4. summary JSON
5. sensitivity plots
6. LaTeX-ready tables
7. summary.md
8. exact config files for reproducibility
```

The final analysis should make it possible to regenerate every number appearing in the sensitivity table directly from saved run outputs.
