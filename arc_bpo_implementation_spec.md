# ARC-BPO Implementation Specification

This document summarizes ARC-BPO in a code-oriented form. It is intended for an implementation agent such as Codex to implement the ARC-BPO loss from the background and methodology.

## 1. Goal

ARC-BPO is a chunk-level preference optimization objective. It extends response-level DPO/BPO ratio matching to semantic chunks while avoiding TBPO-style cross-state token comparisons.

Core idea:

- Compute exact policy-to-reference log-ratios for each chunk.
- Assign each chunk a finite one-sided target at its own prefix.
- Match the chunk model ratio to the chunk data ratio using a Bregman divergence with the SBA generator.
- Ensure that if all chunk targets are matched, the response-level ratio also matches the data preference ratio.

ARC-BPO should not implement token-to-token matching, rejected-to-chosen chunk pairing, or a TBPO correction term `w_t`.

## 2. Notation

For each preference example:

- Prompt: `x`
- Chosen response: `y_w`
- Rejected response: `y_l`
- Trainable policy: `pi_theta`
- Frozen reference policy: `pi_ref`
- KL temperature: `beta > 0`
- Stop-gradient operator: `sg`

A deterministic chunker partitions the responses:

```text
y_w = (c_1^w, ..., c_m^w)
y_l = (c_1^l, ..., c_n^l)
```

Each chunk is a contiguous span of response tokens. The prefix of chunk `c_i` is:

```text
s_i = (x, y_<t_i)
```

where `t_i` is the first token position of the chunk.

## 3. Required Inputs for the Loss

For each batch, the implementation needs:

```python
batch = {
    "prompt_input_ids": ...,          # optional, depending on tokenizer pipeline
    "chosen_input_ids": ...,          # full prompt + chosen or chosen response tokens
    "rejected_input_ids": ...,        # full prompt + rejected or rejected response tokens
    "chosen_response_mask": ...,      # mask selecting response tokens only
    "rejected_response_mask": ...,    # mask selecting response tokens only
    "chosen_chunk_spans": ...,        # list of (start, end) spans over response-token indices
    "rejected_chunk_spans": ...,      # list of (start, end) spans over response-token indices
    "delta_star": ...,                # finite preference margin, scalar per pair
    "chosen_adv_proxy": ...,          # optional detached advantage proxy per chosen chunk
    "rejected_adv_proxy": ...,        # optional detached advantage proxy per rejected chunk
}
```

Important implementation convention:

- Chunk spans should index response tokens only, not prompt tokens.
- If log-prob arrays include prompt tokens, map chunk spans to the correct response-token offset.
- Chunk boundaries must align with tokenizer boundaries.

## 4. Exact Chunk Log-Ratio

For a chunk `c_i`, define:

```math
a_\theta(c_i)
=
\beta \log \frac{\pi_\theta(c_i \mid s_i)}{\pi_{\mathrm{ref}}(c_i \mid s_i)}
=
\sum_{t \in c_i}
\beta
\left[
\log \pi_\theta(y_t \mid x,y_{<t})
-
\log \pi_{\mathrm{ref}}(y_t \mid x,y_{<t})
\right].
```

Implementation:

```python
token_log_ratio = beta * (policy_token_logps - ref_token_logps)
chunk_a_i = token_log_ratio[start:end].sum()
```

The chunk log-ratios must telescope:

```math
\sum_i a_\theta(c_i)
=
\beta \log \frac{\pi_\theta(y \mid x)}{\pi_{\mathrm{ref}}(y \mid x)}.
```

For a preference pair:

```math
\Delta_\theta
=
\sum_i a_\theta(c_i^w) - \sum_j a_\theta(c_j^l).
```

Then:

```math
R_\theta = e^{-\Delta_\theta}.
```

## 5. Why ARC-BPO Avoids `w_t`

TBPO compares tokens or units generated from different prefix states, for example:

```math
S(s_i^w,c_i^w) - S(s_j^l,c_j^l), \quad s_i^w \neq s_j^l.
```

Because the states differ, state-only value terms do not cancel. This creates a correction term such as `w_t` or `w_ij`.

ARC-BPO avoids this by never comparing a chosen chunk directly against a rejected chunk from another prefix. Each chunk is matched to its own one-sided target at its own prefix:

```math
a_\theta(c_i) \rightarrow \tau_i.
```

Therefore:

- no token-to-token alignment,
- no chunk-to-chunk pairing,
- no length-mismatch problem,
- no `w_t` correction,
- no online value head for `w_t`,
- no per-step Monte Carlo KL estimator.

## 6. Data-Anchored Margin

Pairwise BT data identify only the response-level margin:

```math
\Delta^\star
=
\operatorname{logit} P(y^w \succ y^l \mid x)
=
-\log R_{\mathrm{data}}.
```

For hard binary labels, the exact BT margin is infinite. In code, use a finite margin.

Common options:

```python
# Option 1: fixed IPO-style margin
delta_star = tau_0  # e.g. 1.0 or 2.0

# Option 2: reward-model gap
delta_star = reward_model(x, y_w) - reward_model(x, y_l)
delta_star = clamp(delta_star, min_delta, max_delta)
```

Do not use an infinite hard-label margin in the ARC-BPO loss.

## 7. Advantage-Based Credit Shape

The total margin `Delta_star` must be distributed across chunks. ARC-BPO uses a detached advantage-based shape.

Let `A_hat_i` be a frozen or detached reference-side advantage proxy for chunk `c_i`.

Possible implementations:

1. Frozen critic proxy for `A_ref`.
2. Reward-model proxy.
3. Cheap surrogate score.
4. Uniform shape if no proxy is available.

The proxy must be detached:

```python
adv = adv.detach()
```

No gradient should flow through the advantage proxy or through the target construction.

## 8. Winsorization of the Advantage Proxy

For stability, clip the advantage proxy before softmax.

For a vector of chunk advantages `A` within one response side:

```math
\widetilde A_i
=
\operatorname{clip}
\left(
\operatorname{sg}(\widehat A_i),
\bar A - \kappa \hat\sigma,
\bar A + \kappa \hat\sigma
\right),
```

where:

```math
\bar A = \operatorname{sg}(\operatorname{median}_k \widehat A_k),
\qquad
\hat\sigma = \operatorname{sg}(\operatorname{MAD}_k \widehat A_k).
```

Recommended code:

```python
def winsorize_advantages(A, kappa=2.0, eps=1e-8):
    A = A.detach()
    med = A.median()
    mad = (A - med).abs().median()
    scale = mad.clamp_min(eps)
    lo = med - kappa * scale
    hi = med + kappa * scale
    return A.clamp(lo, hi)
```

Use winsorization separately for chosen chunks and rejected chunks.

## 9. Winner and Loser Shapes

For chosen chunks:

```math
\pi_i^w
=
\frac{\exp(\widetilde A_i^w / T)}
{\sum_k \exp(\widetilde A_k^w / T)}.
```

For rejected chunks:

```math
\rho_j^l
=
\frac{\exp(-\widetilde A_j^l / T)}
{\sum_k \exp(-\widetilde A_k^l / T)}.
```

Implementation:

```python
pi_w = softmax(A_tilde_w / T, dim=0)
rho_l = softmax(-A_tilde_l / T, dim=0)
```

Interpretation:

- Chosen chunks with higher advantage receive more positive credit.
- Rejected chunks with lower advantage receive more negative credit.
- Large `T` approaches uniform credit.
- Small `T` concentrates credit on fewer chunks.

Uniform fallback:

```python
pi_w = torch.ones(m, device=device) / m
rho_l = torch.ones(n, device=device) / n
```

## 10. One-Sided Chunk Targets

ARC-BPO defines:

```math
\tau_i^w = \frac{\Delta^\star}{2}\pi_i^w,
\qquad
\tau_j^l = -\frac{\Delta^\star}{2}\rho_j^l.
```

These targets satisfy the calibration constraint automatically:

```math
\sum_i \tau_i^w - \sum_j \tau_j^l = \Delta^\star.
```

Implementation:

```python
tau_w = 0.5 * delta_star * pi_w
tau_l = -0.5 * delta_star * rho_l
```

Targets must be detached:

```python
tau_w = tau_w.detach()
tau_l = tau_l.detach()
```

## 11. Per-Chunk Ratios

For every chunk:

```math
R_{\mathrm{data}}^{(i)} = e^{-\tau_i},
\qquad
R_\theta^{(i)} = e^{-a_\theta(c_i)}.
```

Implementation:

```python
b_i = torch.exp(-tau_i)       # target ratio
R_i = torch.exp(-a_i)         # model ratio
```

Numerical stability:

```python
a_i_safe = a_i.clamp(min=-clip_value, max=clip_value)
tau_i_safe = tau_i.clamp(min=-clip_value, max=clip_value)
R_i = torch.exp(-a_i_safe)
b_i = torch.exp(-tau_i_safe)
```

Typical `clip_value`: 20 to 50, depending on precision.

## 12. SBA Bregman Generator

ARC-BPO uses the SBA generator:

```math
h_\lambda(R)
=
\frac{R^{1+\lambda}-R}{s\lambda(\lambda+1)},
\qquad
h_\lambda''(R)=\frac{R^{\lambda-1}}{s},
\qquad
\lambda>0,\ s>0.
```

Its derivative is:

```math
h_\lambda'(R)
=
\frac{(1+\lambda)R^\lambda - 1}{s\lambda(1+\lambda)}.
```

Bregman divergence:

```math
D_h(b,R)=h(b)-h(R)-h'(R)(b-R).
```

Implementation:

```python
def sba_h(R, lam=1.0, s=4.0):
    return (R.pow(1.0 + lam) - R) / (s * lam * (1.0 + lam))

def sba_h_prime(R, lam=1.0, s=4.0):
    return (((1.0 + lam) * R.pow(lam)) - 1.0) / (s * lam * (1.0 + lam))

def bregman_sba(b, R, lam=1.0, s=4.0):
    return sba_h(b, lam, s) - sba_h(R, lam, s) - sba_h_prime(R, lam, s) * (b - R)
```

Important:

- `lambda -> 0` is the KLIEP/KL limit, not DPO.
- DPO corresponds to a separate logistic-regression Bregman generator.
- ARC-BPO with SBA should use `lambda > 0`.

## 13. ARC-BPO Loss

For one pair:

```math
\mathcal{L}_{\mathrm{ARC}}
=
\frac{1}{m}\sum_{i=1}^{m}
D_{h_\lambda}\left(e^{-\tau_i^w}, e^{-a_\theta(c_i^w)}\right)
+
\frac{1}{n}\sum_{j=1}^{n}
D_{h_\lambda}\left(e^{-\tau_j^l}, e^{-a_\theta(c_j^l)}\right).
```

For a batch, average the pair losses:

```python
loss = pair_losses.mean()
```

## 14. Gradient Interpretation

For a chunk with target `tau_i` and log-ratio `a_i`:

```math
\frac{\partial}{\partial a_i}
D_{h_\lambda}(e^{-\tau_i},e^{-a_i})
=
\frac{1}{s}e^{-\lambda a_i}
\left(e^{-\tau_i}-e^{-a_i}\right).
```

Sign behavior:

- If `a_i < tau_i`, derivative is negative, so gradient descent increases `a_i`.
- If `a_i > tau_i`, derivative is positive, so gradient descent decreases `a_i`.
- Therefore the chunk log-ratio is pulled toward its target.

## 15. Pseudocode for One Pair

```python
def arc_bpo_pair_loss(
    logp_theta_w, logp_ref_w,
    logp_theta_l, logp_ref_l,
    spans_w, spans_l,
    delta_star,
    adv_w=None, adv_l=None,
    beta=0.1,
    T=2.0,
    kappa=2.0,
    lam=1.0,
    s=4.0,
    exp_clip=30.0,
):
    # 1. Token-level beta-scaled log-ratios over response tokens only.
    r_w = beta * (logp_theta_w - logp_ref_w.detach())
    r_l = beta * (logp_theta_l - logp_ref_l.detach())

    # 2. Exact chunk log-ratios.
    a_w = torch.stack([r_w[start:end].sum() for (start, end) in spans_w])
    a_l = torch.stack([r_l[start:end].sum() for (start, end) in spans_l])

    m = a_w.numel()
    n = a_l.numel()

    # 3. Detached shape.
    if adv_w is None:
        pi_w = torch.ones(m, device=a_w.device, dtype=a_w.dtype) / m
    else:
        A_w = winsorize_advantages(adv_w.to(a_w.device, a_w.dtype), kappa=kappa)
        pi_w = torch.softmax(A_w / T, dim=0).detach()

    if adv_l is None:
        rho_l = torch.ones(n, device=a_l.device, dtype=a_l.dtype) / n
    else:
        A_l = winsorize_advantages(adv_l.to(a_l.device, a_l.dtype), kappa=kappa)
        rho_l = torch.softmax(-A_l / T, dim=0).detach()

    # 4. One-sided targets.
    delta_star = torch.as_tensor(delta_star, device=a_w.device, dtype=a_w.dtype).detach()
    tau_w = (0.5 * delta_star * pi_w).detach()
    tau_l = (-0.5 * delta_star * rho_l).detach()

    # 5. Convert to ratios.
    R_w = torch.exp(-a_w.clamp(-exp_clip, exp_clip))
    R_l = torch.exp(-a_l.clamp(-exp_clip, exp_clip))
    b_w = torch.exp(-tau_w.clamp(-exp_clip, exp_clip))
    b_l = torch.exp(-tau_l.clamp(-exp_clip, exp_clip))

    # 6. Bregman SBA matching.
    loss_w = bregman_sba(b_w, R_w, lam=lam, s=s).mean()
    loss_l = bregman_sba(b_l, R_l, lam=lam, s=s).mean()

    return loss_w + loss_l
```

## 16. Pseudocode for a Batch

```python
def arc_bpo_batch_loss(policy_model, ref_model, batch, tokenizer, cfg):
    # Compute token log-probs for chosen and rejected responses.
    # Reference computation should be no_grad or precomputed.
    logp_theta_w = compute_response_token_logps(policy_model, batch["chosen"], batch["chosen_response_mask"])
    logp_theta_l = compute_response_token_logps(policy_model, batch["rejected"], batch["rejected_response_mask"])

    with torch.no_grad():
        logp_ref_w = compute_response_token_logps(ref_model, batch["chosen"], batch["chosen_response_mask"])
        logp_ref_l = compute_response_token_logps(ref_model, batch["rejected"], batch["rejected_response_mask"])

    pair_losses = []
    for b in range(batch_size):
        pair_loss = arc_bpo_pair_loss(
            logp_theta_w[b], logp_ref_w[b],
            logp_theta_l[b], logp_ref_l[b],
            batch["chosen_chunk_spans"][b],
            batch["rejected_chunk_spans"][b],
            batch["delta_star"][b],
            adv_w=batch.get("chosen_adv_proxy", [None] * batch_size)[b],
            adv_l=batch.get("rejected_adv_proxy", [None] * batch_size)[b],
            beta=cfg.beta,
            T=cfg.T,
            kappa=cfg.kappa,
            lam=cfg.sba_lambda,
            s=cfg.sba_scale,
            exp_clip=cfg.exp_clip,
        )
        pair_losses.append(pair_loss)

    return torch.stack(pair_losses).mean()
```

## 17. Reference Log-Probability Computation

A standard implementation should compute log-probability of the target response tokens under teacher forcing.

Given full sequence input IDs:

```python
logits = model(input_ids, attention_mask=attention_mask).logits
shift_logits = logits[:, :-1, :]
shift_labels = input_ids[:, 1:]
log_probs = torch.log_softmax(shift_logits, dim=-1)
token_logps = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
```

Then select only response positions using `response_mask[:, 1:]` or the correct shifted mask.

Critical checks:

- Do not include prompt tokens in ARC-BPO chunk sums.
- Chosen and rejected responses can have different numbers of chunks.
- Padding tokens must be masked out.
- Reference log-probs must not receive gradients.

## 18. Recommended Hyperparameters

Initial defaults:

```yaml
beta: 0.1
sba_lambda: 1.0
sba_scale: 4.0
T: 2.0
kappa: 2.0
delta_star: 2.0
exp_clip: 30.0
use_winsorization: true
use_advantage_shape: true
fallback_to_uniform_shape: true
```

Important ablations:

- Granularity: token, fixed window, discourse-aware chunk, sequence.
- Geometry: one-sided ARC-BPO versus cross-state variant.
- Shape: uniform, shuffled advantage, reward-model proxy, frozen critic.
- Temperature `T`: large, 4.0, 2.0, 1.0, 0.5.
- Winsorization `kappa`: infinity, 3, 2, 1.5, 1.
- SBA exponent `lambda`: near 0, 0.5, 1.0, 2.0.
- Margin `Delta_star`: reward-model gap, fixed tau0, hard-label limit for diagnostic only.

## 19. Invariants and Unit Tests

### Test 1: Chunk telescoping

For each response:

```python
sum_chunk_a = sum(a_chunks)
seq_a = beta * (logp_theta_response.sum() - logp_ref_response.sum())
assert torch.allclose(sum_chunk_a, seq_a, atol=1e-5)
```

### Test 2: Target calibration

```python
assert torch.allclose(tau_w.sum() - tau_l.sum(), delta_star, atol=1e-5)
```

### Test 3: Uniform limit

With no advantage proxy or very large `T`:

```python
tau_w == delta_star / (2 * m)
tau_l == -delta_star / (2 * n)
```

### Test 4: Loss zero at target

If `a_w = tau_w` and `a_l = tau_l`, then:

```python
loss ≈ 0
```

### Test 5: Response-level recovery

If all chunks are matched:

```python
delta_theta = a_w.sum() - a_l.sum()
assert torch.allclose(delta_theta, delta_star, atol=1e-5)
```

### Test 6: No gradient through targets

```python
assert not tau_w.requires_grad
assert not tau_l.requires_grad
assert not pi_w.requires_grad
assert not rho_l.requires_grad
```

### Test 7: Different lengths are allowed

Chosen and rejected can have different chunk counts:

```python
m != n
loss still works
```

## 20. What Not to Implement

Do not implement the following as part of ARC-BPO:

- Same-index token-to-token chosen versus rejected matching.
- Chunk-to-chunk matching across chosen and rejected responses.
- Cross-state BT comparison as the main objective.
- TBPO correction term `w_t` or `w_ij`.
- Online value head for estimating `w_t`.
- Per-step Monte Carlo KL estimator for TBPO-A.
- ChunkRPO cross-response baseline or trimmed opposite-side baseline.
- Hard-label infinite margin as the default training target.

## 21. Minimal Implementation Checklist

A correct ARC-BPO implementation should have:

- A deterministic chunker producing response-token spans.
- Policy and reference token log-probs under teacher forcing.
- Exact chunk log-ratio sums.
- A finite `delta_star` for every pair.
- Detached advantage-based or uniform shapes.
- Winsorization before the shape softmax.
- One-sided targets `tau_w` and `tau_l` satisfying calibration.
- Per-chunk ratios `R_model = exp(-a)` and `R_data = exp(-tau)`.
- SBA Bregman divergence loss.
- Batch mean reduction.
- No gradient through reference model, advantage proxy, shape, or targets.

## 22. Compact Mathematical Summary

For each pair `(x, y_w, y_l)`:

```math
a_\theta(c_i)
=
\sum_{t\in c_i}\beta
\log\frac{\pi_\theta(y_t\mid x,y_{<t})}{\pi_{\mathrm{ref}}(y_t\mid x,y_{<t})}.
```

```math
\pi_i^w=\operatorname{softmax}(\widetilde A_i^w/T),
\qquad
\rho_j^l=\operatorname{softmax}(-\widetilde A_j^l/T).
```

```math
\tau_i^w=\frac{\Delta^\star}{2}\pi_i^w,
\qquad
\tau_j^l=-\frac{\Delta^\star}{2}\rho_j^l.
```

```math
\mathcal{L}_{\mathrm{ARC}}
=
\frac{1}{m}\sum_i D_{h_\lambda}\left(e^{-\tau_i^w},e^{-a_\theta(c_i^w)}\right)
+
\frac{1}{n}\sum_j D_{h_\lambda}\left(e^{-\tau_j^l},e^{-a_\theta(c_j^l)}\right).
```

```math
h_\lambda(R)=\frac{R^{1+\lambda}-R}{s\lambda(1+\lambda)}.
```

If all chunk targets are matched:

```math
\Delta_\theta
=
\sum_i a_\theta(c_i^w)-\sum_j a_\theta(c_j^l)
=
\sum_i \tau_i^w-\sum_j \tau_j^l
=
\Delta^\star,
```

so:

```math
R_\theta = R_{\mathrm{data}}.
```

## 23. Suggested Codex Prompt

Use the following prompt to ask Codex to implement ARC-BPO:

```text
Implement ARC-BPO loss in PyTorch. Use the specification in this markdown. The loss must compute beta-scaled token log-ratio between policy and frozen reference, sum log-ratios over deterministic chunk spans, construct detached one-sided chunk targets from a finite delta_star and an optional winsorized advantage-based shape, and match exp(-a_theta) to exp(-tau) using the SBA Bregman divergence. Do not implement TBPO token matching, cross-state chunk pairing, w_t correction, online value head, or ChunkRPO cross-response baseline. Include unit tests for chunk telescoping, target calibration, zero loss at a=tau, no target gradients, and different numbers of chosen and rejected chunks.
```
