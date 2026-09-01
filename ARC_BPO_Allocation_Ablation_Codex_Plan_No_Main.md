# ARC-BPO Ablation Study — Codex Implementation Plan

## Reviewer Comment

> **Comment 1:** Please provide proper ablations:
>
> - uniform chunk allocation,
> - advantage-based allocation,
> - advantage allocation with and without winsorization.

---

# 1. Goal

The reviewer asks for a controlled ablation that isolates the contribution of:

1. uniform versus advantage-based chunk allocation;
2. the advantage-based allocation mechanism itself;
3. winsorization applied to the advantage scores.

This implementation plan deliberately **excludes the full ARC-BPO `main` settings**.

The existing ARC-BPO `main` result should be treated as the reference configuration already available in the repository / experiment logs.

Do not duplicate the full `main` configuration in new config files.

Do not retrain the ARC-BPO `main` model unless verification is necessary.

---

# 2. Required Ablation Variants

The recommended reviewer-facing table contains four rows:

```text
1. Uniform chunk allocation
2. Advantage allocation
3. Advantage + SBA, without winsorization
4. ARC-BPO main = Advantage + SBA, with winsorization
```

Only the first three rows require new ablation runs.

The fourth row should reuse the already available ARC-BPO `main` result if it was trained and evaluated under exactly the same backbone, data split, training schedule, and evaluation protocol.

Therefore, Codex should implement and launch only:

```text
Ablation A — Uniform chunk allocation
Ablation B — Advantage allocation
Ablation C — Advantage + SBA without winsorization
```

Do not create a separate new configuration for the ARC-BPO `main` method.

---

# 3. Critical Controlled-Ablation Principle

All ablation variants must inherit the same non-ablated settings from the existing ARC-BPO experiment.

The ablation code should load the existing base configuration and modify **only the component being ablated**.

Conceptually:

```python
cfg = load_existing_arcbpo_config()

cfg = apply_ablation_patch(cfg, ablation_name)
```

Do not manually copy the entire `main` configuration into three new files.

This reduces the risk of accidental differences in:

```text
backbone
reference model
dataset
data split
optimizer
learning rate
batch size
gradient accumulation
training steps / epochs
beta
semantic chunker
calibration mechanism
evaluation harness
random seed policy
```

The only differences should be the ablated components described below.

---

# 4. Ablation A — Uniform Chunk Allocation

## Purpose

This variant tests whether ARC-BPO benefits from assigning different amounts of the fixed credit budget to different chunks.

Instead of using the detached advantage-referenced scores, distribute the calibrated response-level margin uniformly across chunks.

For a preferred response with `m` chunks:

```text
winner allocation weight_i = 1 / m
```

For a rejected response with `n` chunks:

```text
loser allocation weight_j = 1 / n
```

The corresponding targets are:

```text
tau_i^w = +(Delta0 / 2) * (1 / m)
tau_j^l = -(Delta0 / 2) * (1 / n)
```

## Important

In this variant:

```text
do not compute advantage scores for allocation
do not apply softmax over advantage scores
winsorization is not applicable
```

The semantic chunker must remain unchanged.

The calibrated response-level margin must remain unchanged.

Sanity check:

```text
sum_i tau_i^w - sum_j tau_j^l ~= Delta0
```

---

# 5. Ablation B — Advantage-Based Allocation

## Purpose

This variant tests whether replacing uniform credit with the detached advantage-referenced allocation improves performance.

Use the existing ARC-BPO detached advantage proxy and the existing allocation-temperature mechanism.

Conceptually:

```text
A_hat_i = detach(a_theta(c_i))
```

Then apply the same advantage-based allocation rule already implemented in the codebase.

For the winner:

```text
allocation_i = softmax(A_i / T)
```

For the loser:

```text
allocation_j = softmax(-A_j / T)
```

and construct the calibrated targets from the fixed response-level budget.

## Important

This row should correspond to the repository's existing definition of:

```text
"Advantage allocation"
```

If the current codebase has a distinct pre-SBA / base-Bregman variant used for this ablation, use that existing implementation.

Do **not** invent a new loss definition merely to create this row.

The goal is to reproduce the exact intended component progression:

```text
Uniform allocation
    ->
Advantage allocation
    ->
Advantage + SBA
    ->
Advantage + SBA + winsorization
```

If the repository does not currently distinguish `Advantage allocation` from `Advantage + SBA`, Codex should stop and report this rather than silently inventing a distinction.

---

# 6. Ablation C — Advantage + SBA Without Winsorization

## Purpose

This is the most important controlled comparison for isolating winsorization.

Keep:

```text
advantage-based allocation
SBA / Bregman generator
semantic chunking
calibration
all training settings
```

exactly the same as the existing ARC-BPO pipeline.

Disable only the winsorization step.

Conceptually:

```python
scores_for_allocation = detached_advantage_scores
```

instead of:

```python
scores_for_allocation = winsorize(
    detached_advantage_scores,
    kappa=kappa
)
```

Then form the allocation using the unchanged temperature:

```text
winner:
softmax(scores / T)

loser:
softmax(-scores / T)
```

This creates the cleanest comparison:

```text
Advantage + SBA without winsorization
vs.
existing ARC-BPO main
```

where the intended difference is only:

```text
winsorization OFF
vs.
winsorization ON
```

---

# 7. Existing ARC-BPO Main Result

Do not create a new main-method configuration in this ablation plan.

The existing ARC-BPO main result should be inserted into the final table only after verifying that it uses the same:

```text
backbone
dataset
training data
evaluation tasks
training schedule
seed aggregation
checkpoint-selection rule
evaluation script
```

as the new ablation runs.

If the existing main result is not directly comparable, Codex should report the mismatch.

Do not silently mix incompatible runs.

---

# 8. Recommended Code Structure

Create an ablation enum or command-line flag.

Example:

```python
ABLATATIONS = {
    "uniform",
    "advantage",
    "advantage_sba_no_winsor",
}
```

Suggested CLI:

```bash
python train_arcbpo.py \
    --config PATH_TO_EXISTING_BASE_CONFIG \
    --ablation uniform \
    --run_name ablation_uniform
```

```bash
python train_arcbpo.py \
    --config PATH_TO_EXISTING_BASE_CONFIG \
    --ablation advantage \
    --run_name ablation_advantage
```

```bash
python train_arcbpo.py \
    --config PATH_TO_EXISTING_BASE_CONFIG \
    --ablation advantage_sba_no_winsor \
    --run_name ablation_advantage_sba_no_winsor
```

No command for `main` should be added to this plan.

---

# 9. Recommended Implementation Pattern

Pseudo-code:

```python
def build_chunk_targets(
    winner_scores,
    loser_scores,
    num_winner_chunks,
    num_loser_chunks,
    delta0,
    temperature,
    ablation,
):
    if ablation == "uniform":
        winner_weights = torch.full(
            (num_winner_chunks,),
            1.0 / num_winner_chunks,
            device=winner_scores.device,
        )

        loser_weights = torch.full(
            (num_loser_chunks,),
            1.0 / num_loser_chunks,
            device=loser_scores.device,
        )

    elif ablation == "advantage":
        winner_used = winner_scores.detach()
        loser_used = loser_scores.detach()

        # Use the repository's exact existing implementation
        # of the "Advantage allocation" ablation.
        winner_weights = advantage_allocation(
            winner_used,
            temperature=temperature,
            side="winner",
        )

        loser_weights = advantage_allocation(
            loser_used,
            temperature=temperature,
            side="loser",
        )

    elif ablation == "advantage_sba_no_winsor":
        winner_used = winner_scores.detach()
        loser_used = loser_scores.detach()

        winner_weights = torch.softmax(
            winner_used / temperature,
            dim=-1,
        )

        loser_weights = torch.softmax(
            -loser_used / temperature,
            dim=-1,
        )

    else:
        raise ValueError(f"Unsupported ablation: {ablation}")

    tau_w = +(delta0 / 2.0) * winner_weights
    tau_l = -(delta0 / 2.0) * loser_weights

    return tau_w, tau_l
```

Important:

The actual training code should reuse existing ARC-BPO functions whenever possible.

Do not duplicate target-construction logic if the repository already has:

```text
compute_advantage_proxy()
winsorize_scores()
compute_allocation_weights()
compute_chunk_targets()
compute_sba_loss()
```

Patch or configure those functions instead.

---

# 10. Recommended Configuration Patches

Do not create full standalone configs.

Create minimal override files if the codebase supports configuration inheritance.

Example:

```text
configs/ablations/
├── uniform.yaml
├── advantage.yaml
└── advantage_sba_no_winsor.yaml
```

Example `uniform.yaml`:

```yaml
ablation:
  name: uniform
  allocation: uniform
  winsorization: false
```

Example `advantage.yaml`:

```yaml
ablation:
  name: advantage
  allocation: advantage
```

Example `advantage_sba_no_winsor.yaml`:

```yaml
ablation:
  name: advantage_sba_no_winsor
  allocation: advantage
  use_sba: true
  winsorization: false
```

These files should override the existing experiment configuration rather than restating it.

---

# 11. What Must Remain Fixed

Codex should automatically compare configs and verify that all non-ablated settings are identical.

The following should not change across the three new runs:

```text
model backbone
reference model
training dataset
validation/test data
semantic chunker
chunk-size safeguards
optimizer
learning rate
scheduler
batch size
gradient accumulation
number of epochs / steps
beta
calibration margin mechanism
allocation temperature, except where uniform makes it irrelevant
evaluation tasks
evaluation harness
checkpoint-selection rule
random seed(s)
```

---

# 12. Seed Protocol

Preferred:

```text
use the same seed set as the existing main ARC-BPO result
```

If the main table reports multiple-seed averages, use the same seed set for all new ablations.

If only a single seed is feasible:

```text
use exactly the same fixed seed for all three ablation variants
```

Do not use different seeds for different variants.

---

# 13. Evaluation

Evaluate the new variants using the exact same evaluation pipeline as the existing ARC-BPO main experiment.

For the Mistral-7B-v0.1 ablation table, store task-level scores for:

```text
HellaSwag
ARC
MMLU
TruthfulQA
Winogrande
GSM8K
Average
```

Do not manually enter the final average.

Compute it automatically from the task scores using the same aggregation rule as the main paper.

---

# 14. Required Output CSV

Generate:

```text
outputs/ablations/allocation_ablation_results.csv
```

Recommended columns:

```text
variant
seed
hellaswag
arc
mmlu
truthfulqa
winogrande
gsm8k
average
checkpoint
config_hash
```

Rows should include only the newly run variants.

Do not copy the main ARC-BPO row into the raw ablation-run CSV unless it is loaded programmatically from the existing main result.

---

# 15. Final Reviewer Table

The final paper/rebuttal table should have this conceptual structure:

| Variant | Hella. | ARC | MMLU | Truth. | Wino. | GSM8K | Avg. |
|---|---:|---:|---:|---:|---:|---:|---:|
| Uniform chunk allocation | ... | ... | ... | ... | ... | ... | ... |
| Advantage allocation | ... | ... | ... | ... | ... | ... | ... |
| Advantage + SBA (without winsorization) | ... | ... | ... | ... | ... | ... | ... |
| ARC-BPO main / Advantage + SBA (with winsorization) | EXISTING | EXISTING | EXISTING | EXISTING | EXISTING | EXISTING | EXISTING |

The last row must come from the already existing main result.

Do not retrain or redefine `main` in this implementation plan.

---

# 16. Comparison Logic

The reviewer asks for three distinct conclusions.

## Comparison 1 — Uniform vs Advantage

Compare:

```text
Uniform chunk allocation
vs.
Advantage allocation
```

This tests whether advantage-referenced credit assignment is preferable to equal allocation.

Interpret only the actual observed difference.

---

## Comparison 2 — Contribution of SBA

If the codebase has a distinct `Advantage allocation` row and `Advantage + SBA` row, compare:

```text
Advantage allocation
vs.
Advantage + SBA without winsorization
```

This isolates the additional SBA/Bregman component.

This comparison is useful for the paper but is not the central requirement of the reviewer comment.

---

## Comparison 3 — Winsorization

Compare:

```text
Advantage + SBA without winsorization
vs.
existing ARC-BPO main
```

This is the controlled winsorization comparison.

The only intended difference should be:

```text
winsorization OFF
vs.
winsorization ON
```

This comparison must not also change the loss, allocation temperature, chunker, or calibration.

---

# 17. Sanity Checks

Codex must verify all of the following.

## Uniform allocation

For every winner response:

```text
all winner weights are equal
sum winner weights ~= 1
```

For every loser response:

```text
all loser weights are equal
sum loser weights ~= 1
```

---

## Advantage allocation

Verify:

```text
allocation weights sum to 1
weights are non-negative
targets are detached from the allocation-score computation
```

---

## Calibration

For every preference pair and every variant:

```text
sum(tau_w) - sum(tau_l) ~= Delta0
```

Suggested tolerance:

```text
1e-5
```

---

## No-winsorization variant

Verify that:

```text
winsorization function is not called
```

or:

```text
the input scores pass through unchanged
```

Add a debug assertion if possible.

---

# 18. Config-Difference Audit

For every pair of ablation runs, write a config diff.

Example output:

```text
outputs/ablations/config_diff_uniform_vs_advantage.txt
outputs/ablations/config_diff_advantage_vs_sba_no_winsor.txt
```

The diff should contain only expected ablation-specific fields.

For the winsorization comparison against the existing main result, the expected difference should be only the winsorization setting, assuming the main configuration is otherwise identical.

If unexpected config differences are detected, flag them before using results in the paper.

---

# 19. Suggested Launcher Script

Create:

```text
run_allocation_ablations.py
```

Pseudo-code:

```python
ABLATIONS = [
    "uniform",
    "advantage",
    "advantage_sba_no_winsor",
]

for ablation in ABLATIONS:
    for seed in SEEDS:
        cfg = load_existing_base_config()
        cfg = apply_ablation(cfg, ablation)
        assert_only_expected_fields_changed(cfg, ablation)

        run_training(cfg)
        run_evaluation(cfg)
        save_result(cfg)
```

Again:

```text
do not include "main" in ABLATIONS
```

---

# 20. Suggested Result Aggregator

Create:

```text
summarize_allocation_ablations.py
```

It should:

1. load all run-level results;
2. group by variant;
3. compute mean and standard deviation if multiple seeds are used;
4. compute the task average automatically;
5. load the existing ARC-BPO main result separately;
6. verify comparability;
7. produce the final LaTeX table;
8. produce a Markdown summary.

Recommended outputs:

```text
outputs/ablations/
├── allocation_ablation_results.csv
├── allocation_ablation_summary.csv
├── allocation_ablation_summary.json
├── allocation_ablation_table.tex
├── config_audit.json
└── summary.md
```

---

# 21. Automatic Main-Result Import

If the repository stores the existing main result in JSON/CSV, the summarizer should read it automatically.

Example:

```python
main_result = load_existing_result(
    path=args.main_result_file
)
```

Use it only when:

```text
main backbone == ablation backbone
main data == ablation data
main eval protocol == ablation eval protocol
main seed policy == ablation seed policy
```

The code should not contain hard-coded main scores.

---

# 22. Recommended Paper Interpretation

If the results support the intended trend, the paper can say:

```text
Replacing uniform allocation with advantage-based allocation improves the
average score, showing that non-uniform credit assignment is useful beyond
semantic chunking alone. When the advantage allocation and SBA loss are held
fixed, enabling winsorization provides an additional improvement, isolating
the benefit of clipping extreme allocation scores. These controlled ablations
show that the gains do not arise solely from chunking, but from the proposed
credit-allocation and robustness components.
```

Use actual values only after the experiments finish.

Do not state that winsorization is strongly beneficial on clean data if the observed difference is small.

---

# 23. Recommended Rebuttal Structure

```latex
\reviewer{\textbf{Comment 1:} Please provide proper ablations:
\begin{itemize}
    \item uniform chunk allocation,
    \item advantage-based allocation,
    \item advantage allocation with and without winsorization.
\end{itemize}}

\noindent\textbf{Response.}
We thank the reviewer for this helpful suggestion. We have added a controlled
ablation in which the one-sided semantic-chunk construction and the remaining
training protocol are kept fixed while the credit-allocation components are
varied.

[INSERT TABLE]

Replacing uniform chunk allocation with advantage-based allocation changes the
average score from [VALUE] to [VALUE], showing [INTERPRET ACTUAL RESULT].
To isolate winsorization, we keep the advantage allocation and SBA generator
fixed and compare the variant without winsorization against the existing
ARC-BPO configuration with winsorization. This changes the average from
[VALUE] to [VALUE]. The latter comparison differs only in whether the
advantage scores are winsorized before forming the allocation, thereby
directly isolating the contribution of winsorization.
```

---

# 24. Important Interpretation Constraint

Do not use this misleading comparison:

```text
Advantage allocation
vs.
ARC-BPO main
```

to quantify the effect of winsorization.

That comparison may change more than one component.

The winsorization effect must be isolated using:

```text
Advantage + SBA without winsorization
vs.
Advantage + SBA with winsorization
```

where the latter is the already existing ARC-BPO main configuration.

---

# 25. What Not to Do

Do not:

```text
- create a new ARC-BPO main configuration in this ablation plan;
- rerun main unnecessarily;
- manually copy all main hyperparameters into every ablation config;
- alter the semantic chunker;
- alter Delta0;
- alter T except where uniform allocation makes T irrelevant;
- alter the evaluation protocol;
- use different seeds across variants;
- compare runs from incompatible checkpoints or data splits;
- treat "Advantage allocation vs main" as the winsorization-only comparison;
- invent an "Advantage allocation" loss definition if the codebase does not already define it;
- manually edit scores to create a monotonic progression.
```

---

# 26. Minimum Additional Runs Required

Assuming the existing ARC-BPO main result is directly reusable, the minimum new training runs are:

```text
1. Uniform chunk allocation
2. Advantage allocation
3. Advantage + SBA without winsorization
```

If using one seed:

```text
3 new runs
```

If using three matched seeds:

```text
9 new runs
```

The existing main result supplies:

```text
Advantage + SBA with winsorization
```

for the final comparison table.

---

# 27. Preferred Strong Version

Use:

```text
3 matched seeds
same base checkpoint / initialization protocol
same data
same training schedule
same evaluation harness
```

Report:

```text
mean ± std
```

for every task and overall average.

If the main result is already a matched multi-seed result, use the same seeds for the new ablations.

---

# 28. Final Codex Checklist

Before completing the task, verify:

```text
[ ] no new ARC-BPO main config was created
[ ] uniform allocation implemented
[ ] advantage allocation implemented using existing repository logic
[ ] advantage + SBA without winsorization implemented
[ ] winsorization is truly disabled in the no-winsor run
[ ] same semantic chunker used in all runs
[ ] same calibration mechanism used in all runs
[ ] same T used in all advantage-based runs
[ ] same training schedule used
[ ] same evaluation script used
[ ] same seed set used
[ ] target calibration check passes
[ ] allocation simplex checks pass
[ ] config diffs contain only expected changes
[ ] task-level metrics saved
[ ] averages computed automatically
[ ] existing main result imported rather than hard-coded
[ ] final LaTeX table generated automatically
[ ] no manually modified experimental values
```

---

# 29. Expected Final Deliverables

Codex should produce:

```text
1. ablation-mode implementation / flags
2. three minimal ablation override configs
3. run_allocation_ablations.py
4. summarize_allocation_ablations.py
5. allocation_ablation_results.csv
6. allocation_ablation_summary.csv
7. allocation_ablation_table.tex
8. config_audit.json
9. summary.md
```

The ARC-BPO `main` configuration is intentionally excluded from the implementation plan and should only be imported as an existing comparison result after compatibility checks.
