# Codex Implementation Plan — Reviewer Comment on Generation Diversity

## Reviewer Comment

> **Comment 2:** The authors state that the proposed method leads to better generation diversity. However, what is the underlying mechanism behind this improvement? Fine-grained chunk-level preference optimization does not necessarily lead to greater diversity, and the paper should provide further analysis or empirical evidence to explain this phenomenon.

---

# 1. What the Reviewer Is Asking

The reviewer is not simply asking for another diversity metric.

The paper already has output-level diversity evidence such as:

- Predictive Entropy,
- Distinct-1,
- Self-BLEU.

These metrics show **that ARC-BPO produces more diverse generations**, but they do not explain **why**.

The new experiment should therefore test a mechanism:

> ARC-BPO localizes preference-induced updates to high-credit chunks, while low-credit chunks remain closer to the reference policy.

The empirical prediction is:

\[
\mathrm{KL}_{\text{Low Credit}}
<
\mathrm{KL}_{\text{Medium Credit}}
<
\mathrm{KL}_{\text{High Credit}}.
\]

If this pattern appears, it supports the interpretation that ARC-BPO avoids unnecessary policy drift on low-credit content while concentrating stronger updates on preference-relevant chunks.

This is **mechanism-oriented evidence**, not a theoretical or causal proof that ARC-BPO guarantees diversity.

---

# 2. Recommended Experiment

## Main experiment: Credit-conditioned policy drift

Use the existing trained ARC-BPO checkpoint and its frozen reference model.

Do **not** retrain the model for this experiment unless the required checkpoint is unavailable.

For held-out examples:

1. apply the exact same semantic chunker used during ARC-BPO training;
2. recompute the ARC-BPO chunk credit / target magnitude `|tau_i|`;
3. compute policy-to-reference KL divergence at each token prefix;
4. average token-level KL within each chunk;
5. group chunks into Low / Medium / High credit;
6. compare the average KL of the three groups.

The desired mechanism-consistent pattern is:

```text
Low-credit chunks    -> smallest policy-reference KL
Medium-credit chunks -> intermediate KL
High-credit chunks   -> largest KL
```

Do not force this pattern. Report the actual results.

---

# 3. Checkpoints and Data

Use:

```text
ARC-BPO checkpoint:
    the exact Llama-3-8B ARC-BPO checkpoint used for the diversity analysis

Reference checkpoint:
    the frozen reference/SFT checkpoint corresponding to that ARC-BPO run

Evaluation split:
    held-out preference data, preferably the same held-out prompts used for
    the generation-diversity experiment
```

Recommended command interface:

```bash
python analyze_credit_drift.py \
    --policy_model PATH_TO_ARCBPO \
    --reference_model PATH_TO_REFERENCE \
    --dataset DATASET_OR_PATH \
    --split test \
    --output_dir outputs/credit_drift \
    --beta BETA \
    --delta0 DELTA0 \
    --temperature T \
    --kappa KAPPA \
    --max_length 2048 \
    --seed 42
```

Replace all hyperparameters with the actual final ARC-BPO configuration.

---

# 4. Reuse the Exact ARC-BPO Chunker

Do not create a new chunking implementation if the training repository already contains one.

For each response:

```text
y = (c_1, c_2, ..., c_m)
```

Store for every chunk:

```text
example_id
side              # winner / loser
chunk_id
start_token
end_token
num_tokens
chunk_text
```

The chunk boundaries must match training exactly.

Sanity check:

```text
all response tokens should be covered exactly once by the chunk spans
```

---

# 5. Compute Chunk Log-Ratios

For each chunk `c_i` at prefix state `s_i`, compute:

```text
a_theta(c_i)
=
beta * [
    log pi_theta(c_i | s_i)
    -
    log pi_ref(c_i | s_i)
]
```

Operationally:

```text
chunk_logratio =
beta * sum_t(
    policy_logprob_observed_token_t
    -
    reference_logprob_observed_token_t
)
```

Save:

```text
chunk_logratio
```

The detached credit proxy is simply:

```text
A_hat_i = detach(chunk_logratio)
```

No gradients should be computed.

---

# 6. Recompute the ARC-BPO Allocation

## Winner side

Apply the same winsorization used during training:

```text
A_tilde_i =
clip(
    A_hat_i,
    median(A_hat) - kappa * MAD(A_hat),
    median(A_hat) + kappa * MAD(A_hat)
)
```

Then:

```text
pi_i = softmax(A_tilde_i / T)
```

and:

```text
tau_i = +(Delta0 / 2) * pi_i
```

## Loser side

Use:

```text
rho_j = softmax(-A_tilde_j / T)
```

and:

```text
tau_j = -(Delta0 / 2) * rho_j
```

For analysis define:

```text
credit_magnitude = abs(tau)
```

Also store:

```text
allocation_weight
```

because this can later be used as a normalization/control variable.

---

# 7. Calibration Sanity Check

For every preference pair verify:

```text
sum(tau_winner) - sum(tau_loser) ~= Delta0
```

Recommended tolerance:

```text
1e-5
```

Log:

```text
max_calibration_error
mean_calibration_error
```

If this sanity check fails, stop the analysis and fix the implementation.

---

# 8. Compute Token-Level Policy-to-Reference KL

This is the key diagnostic.

For every response token position `t`, compute the full next-token distributions:

```text
p = pi_theta(. | s_t)
q = pi_ref(. | s_t)
```

Then compute:

```text
KL_t = D_KL(p || q)
```

Use the same direction throughout:

```text
policy || reference
```

Recommended PyTorch implementation:

```python
with torch.no_grad():
    policy_logits = policy(**batch).logits
    reference_logits = reference(**batch).logits

    logp = torch.log_softmax(policy_logits.float(), dim=-1)
    logq = torch.log_softmax(reference_logits.float(), dim=-1)

    p = logp.exp()
    token_kl = torch.sum(p * (logp - logq), dim=-1)
```

Important:

- compute KL over the **full vocabulary distribution**;
- do not use only the probability of the observed token;
- use FP32 for `log_softmax` if necessary for stability;
- both models must be in `.eval()` mode;
- use `torch.no_grad()`.

---

# 9. Aggregate Token KL to Chunk-Level Policy Drift

For each chunk:

```text
chunk_policy_reference_kl =
mean(token_kl over all token positions in the chunk)
```

Save one row per chunk.

Recommended CSV:

```text
outputs/credit_drift/chunk_level_metrics.csv
```

Columns:

```text
example_id
side
chunk_id
num_tokens
chunk_text
chunk_logratio
allocation_weight
tau
credit_magnitude
policy_reference_kl
```

---

# 10. Primary Low / Medium / High Credit Analysis

Pool all held-out chunks and split by empirical tertiles of:

```text
credit_magnitude = |tau|
```

Groups:

```text
Low    = bottom 1/3
Medium = middle 1/3
High   = top 1/3
```

For each group report:

```text
number of chunks
mean |tau|
mean policy-reference KL
bootstrap 95% confidence interval
```

Recommended final table:

| Credit group | # chunks | Mean \(|\tau_i|\) | Policy--reference KL |
|---|---:|---:|---:|
| Low | ... | ... | ... |
| Medium | ... | ... | ... |
| High | ... | ... | ... |

The expected mechanism-consistent ordering is:

```text
KL_low < KL_medium < KL_high
```

Again: do not manipulate results to obtain this pattern.

---

# 11. Strongly Recommended: Continuous Correlation

The three-bin analysis is intuitive for reviewers, but it depends on binning.

Also compute the continuous association between:

```text
|tau_i|
```

and:

```text
policy_reference_kl
```

Use:

```text
Spearman rank correlation
```

Report:

```text
rho
p-value
N
```

Recommended output:

```json
{
  "spearman_rho": ...,
  "p_value": ...,
  "num_chunks": ...
}
```

A positive and significant Spearman correlation supports the same mechanism without relying on arbitrary Low/Medium/High boundaries.

---

# 12. Important Control: Use Allocation Weight Too

Because:

```text
tau_i = Delta0 / 2 * allocation_weight_i
```

the target magnitude may be influenced by the number of chunks in a response.

Therefore repeat the analysis using:

```text
allocation_weight
```

instead of `|tau_i|`.

Compute:

```text
Spearman(allocation_weight, policy_reference_kl)
```

and optionally Low / Medium / High groups by allocation weight.

If both analyses show the same trend, the result is stronger.

---

# 13. Winner / Loser Sanity Check

Repeat the Low / Medium / High analysis separately for:

```text
winner chunks
loser chunks
```

Use:

```text
abs(tau)
```

for both.

Output:

```text
grouped_credit_drift_winner.csv
grouped_credit_drift_loser.csv
```

This checks that the overall pattern is not driven only by one side of the preference pair.

---

# 14. Bootstrap Confidence Intervals

Use bootstrap confidence intervals for group-level KL.

Prefer resampling by:

```text
example / preference pair
```

not individual chunks, because chunks from the same response are dependent.

Recommended:

```text
bootstrap iterations: 2000
confidence level: 95%
seed: 42
```

Each bootstrap replicate should:

1. sample preference examples with replacement;
2. include all chunks belonging to each sampled example;
3. recompute the group mean.

Report either:

```text
mean [95% CI]
```

or:

```text
mean +/- bootstrap standard error
```

Make the caption explicit.

---

# 15. Optional Stronger Analysis: ARC-BPO vs DPO

If a trained DPO checkpoint already exists, add this analysis.

Do not retrain DPO unless necessary.

## Goal

Test whether ARC-BPO changes low-credit states less than a sequence-level preference method.

Use the exact same held-out prefix states.

Define Low / Medium / High groups using ARC-BPO credit.

For every state compute:

```text
KL_ARC = D_KL(pi_ARC || pi_ref)
KL_DPO = D_KL(pi_DPO || pi_ref)
```

Then report:

| ARC-BPO credit group | ARC-BPO KL | DPO KL |
|---|---:|---:|
| Low | ... | ... |
| Medium | ... | ... |
| High | ... | ... |

The strongest result would be:

```text
Low-credit:
    ARC-BPO KL << DPO KL

High-credit:
    ARC-BPO still has substantial KL
```

This would support:

> ARC-BPO is not merely making smaller updates globally; it is redistributing policy change toward high-credit content.

Do not make this claim unless the numbers support it.

---

# 16. Optional Additional Baselines

If checkpoints already exist:

```text
BPO
TBPO-A
TDPO
```

can also be measured at the same prefix states.

This is useful for an appendix or supplementary plot, but not required for the minimum reviewer response.

---

# 17. Recommended Outputs

The script should produce:

```text
outputs/credit_drift/
├── chunk_level_metrics.csv
├── grouped_credit_drift.csv
├── grouped_credit_drift_winner.csv
├── grouped_credit_drift_loser.csv
├── correlation_results.json
├── config.json
├── credit_group_kl.pdf
├── credit_vs_kl.pdf
└── summary.md
```

If DPO comparison is included:

```text
├── arc_vs_dpo_grouped_kl.csv
└── arc_vs_dpo_grouped_kl.pdf
```

---

# 18. Recommended Plot

## Plot 1: Low / Medium / High Credit vs KL

```text
x-axis:
    Low
    Medium
    High

y-axis:
    Mean policy-reference KL

error bars:
    bootstrap 95% CI
```

Expected visual pattern:

```text
Low < Medium < High
```

Use the actual result.

---

## Plot 2: Continuous Credit vs KL

Either:

- scatter with low alpha,
- hexbin,
- or decile-binned mean curve.

```text
x-axis:
    |tau_i|

y-axis:
    policy-reference KL
```

Include Spearman rho in the caption.

---

# 19. Minimum Experiment Needed

If time or compute is limited, run at least:

1. existing ARC-BPO checkpoint;
2. existing reference checkpoint;
3. held-out preference data;
4. exact training chunker;
5. recompute `|tau_i|`;
6. full-vocabulary token KL `D_KL(pi_ARC || pi_ref)`;
7. average KL per chunk;
8. Low / Medium / High groups;
9. bootstrap CI;
10. Spearman correlation.

This requires no retraining.

Combined with the existing Predictive Entropy / Distinct-1 / Self-BLEU figure, this should be sufficient to answer the reviewer's mechanism question if the results are supportive.

---

# 20. What the New Experiment Establishes

If the expected trend is observed:

```text
credit magnitude
      ↓
localized policy drift
      ↓
low-credit content remains closer to reference
```

The existing diversity figure already provides:

```text
ARC-BPO
      ↓
higher Predictive Entropy
higher Distinct-1
competitive Self-BLEU
```

Together, the two analyses support the interpretation:

```text
localized updates are associated with reduced unnecessary distributional drift,
which is consistent with better preservation of alternative continuations
```

Do not claim:

```text
localized updates mathematically guarantee diversity
```

and do not claim:

```text
this experiment proves causality
```

---

# 21. Paper Wording If the Results Support the Hypothesis

Use a concise Experimental paragraph:

```latex
\paragraph{Generation diversity and localized policy drift.}
Fine-grained chunk-level optimization does not inherently guarantee greater
generation diversity. We instead interpret the empirical behavior of ARC-BPO
through the localization of its preference-induced updates. ARC-BPO distributes
the fixed calibrated margin across semantic chunks, so low-credit chunks
receive little pressure to move their observed policy-to-reference likelihood
ratios, whereas larger updates are concentrated on chunks carrying more of the
preference signal.

To examine this mechanism directly, we stratify held-out chunks by the
magnitude of their allocated credit $|\tau_i|$. For each chunk, we measure
policy drift as the average token-level KL divergence between ARC-BPO and the
reference policy over its prefix states. Chunks are divided into low-, medium-,
and high-credit groups using empirical tertiles. Table~\ref{tab:credit_drift}
shows that [INSERT ACTUAL RESULT]. The continuous Spearman analysis shows the
same qualitative relationship. These results support the interpretation that
ARC-BPO localizes preference-induced policy changes according to its
chunk-level allocation.

We complement this mechanism-oriented diagnostic with generation-level
diversity metrics. ARC-BPO achieves the highest predictive entropy and
Distinct-1 among the compared methods while maintaining competitive Self-BLEU.
Together, these results are consistent with localized policy drift being
associated with reduced diversity collapse, although we do not claim a
theoretical guarantee of diversity.
```

---

# 22. Rebuttal Table

Recommended compact table:

```latex
\begin{table}[t]
\centering
\small
\setlength{\tabcolsep}{7pt}
\renewcommand{\arraystretch}{1.10}
\begin{tabular}{lccc}
\toprule
\textbf{Credit group} &
\textbf{\# Chunks} &
\textbf{Mean $|\tau_i|$} &
\textbf{Policy--reference KL} \\
\midrule
Low    & [VALUE] & [VALUE] & [VALUE] \\
Medium & [VALUE] & [VALUE] & [VALUE] \\
High   & [VALUE] & [VALUE] & [VALUE] \\
\bottomrule
\end{tabular}
\caption{Credit-conditioned policy drift on held-out generations. Chunks are
grouped by the magnitude of their allocated target $|\tau_i|$.
Policy--reference KL denotes the average token-level KL divergence between
ARC-BPO and the reference policy over prefix states within each chunk.}
\label{tab:credit_drift_rebuttal}
\end{table}
```

---

# 23. Suggested Rebuttal Wording

If the results support the mechanism:

```text
We thank the reviewer for this important observation. We agree that
fine-grained chunk-level optimization does not inherently imply greater
generation diversity. The diversity figure in the original manuscript
establishes the empirical output-level effect, but it does not identify the
underlying mechanism.

We therefore add a credit-conditioned policy-drift analysis. ARC-BPO
distributes a fixed calibrated margin across semantic chunks, so chunks
receiving little credit have targets close to zero and receive little direct
pressure to move their observed policy-to-reference likelihood ratios, while
larger target magnitudes are concentrated on higher-credit chunks.

On held-out data, we group chunks by |tau_i| and measure their average
token-level policy-to-reference KL divergence. [INSERT ACTUAL RESULTS.] We
also observe a positive Spearman association between allocated credit and
policy drift. These results provide direct empirical evidence that ARC-BPO
localizes preference-induced policy changes according to its chunk-level
credit allocation.

Combined with the higher predictive entropy and Distinct-1 and competitive
Self-BLEU reported in our diversity analysis, these findings support the
interpretation that localized policy updates are associated with reduced
unnecessary distributional drift and better preservation of alternative
continuations. We emphasize that this is an empirically supported mechanism
rather than a theoretical or causal guarantee of diversity.
```

---

# 24. Important Sanity Checks

Codex must verify:

```text
1. policy.eval()
2. reference.eval()
3. torch.no_grad()
4. same tokenizer
5. same chunker as training
6. chunk spans cover response tokens exactly once
7. sum winner targets - sum loser targets ~= Delta0
8. winner allocation sums to 1
9. loser allocation sums to 1
10. policy-reference KL >= 0 up to numerical tolerance
11. no training examples in the primary held-out analysis
12. all reported group boundaries are fixed before interpreting results
```

---

# 25. Important Interpretation Rules

If results show:

```text
KL_low < KL_medium < KL_high
```

then the mechanism is supported.

If the differences are very small, report that honestly.

If the ordering is not monotonic:

```text
do not claim that ARC-BPO localizes policy drift according to credit
```

In that case, possible follow-up diagnostics are:

- analyze winner and loser sides separately;
- use normalized allocation weight instead of `|tau|`;
- compare against DPO;
- investigate chunk length as a confound.

Never alter bins or remove examples simply to obtain the expected ordering.

---

# 26. Best Practical Version

For a strong but efficient reviewer response, run:

### Required
- Credit-conditioned Low / Medium / High KL
- Bootstrap 95% CI
- Spearman correlation
- Winner / loser sanity check

### Strong optional control
- ARC-BPO vs DPO KL at the same ARC-defined credit groups

### Already available
- Predictive Entropy
- Distinct-1
- Self-BLEU

This gives a complete reviewer-facing story:

```text
ARC-BPO credit allocation
        ↓
policy drift is localized
        ↓
low-credit states are less unnecessarily altered
        ↓
generation-level diversity remains higher empirically
```

The final wording should remain:

> The evidence supports a localized-update mechanism associated with improved
> generation diversity; it does not establish a theoretical or causal
> guarantee of diversity.
