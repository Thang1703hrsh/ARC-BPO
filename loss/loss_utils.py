from typing import Optional, Sequence, Tuple

import torch


Span = Tuple[int, int]


def compute_entropy(
    logits: torch.FloatTensor,
    labels: torch.LongTensor,
) -> torch.FloatTensor:
    """Compute the entropy of the policy distribution over tokens.

    Args:
        logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
        labels: Labels to determine which positions to compute entropy for.
                Label tokens with a value of -100 are ignored. Shape: (batch_size, sequence_length)

    Returns:
        A tensor of shape (batch_size, sequence_length-1) containing the entropy at each token position.
        Higher entropy = more uncertain/uniform distribution over vocabulary.
    """
    assert logits.shape[:-1] == labels.shape

    labels = labels[:, 1:].clone()
    logits = logits[:, :-1, :]
    loss_mask = labels != -100

    # Compute probabilities
    probs = logits.softmax(-1)  # Shape: (batch_size, seq_len, vocab_size)
    log_probs = logits.log_softmax(-1)

    # Entropy = -sum(p * log(p))
    entropy = -(probs * log_probs).sum(-1)  # Shape: (batch_size, seq_len)

    # Apply mask to ignore padding/special tokens
    entropy = entropy * loss_mask

    return entropy, loss_mask


def compute_kl(
    logits: torch.FloatTensor,
    reference_logits: torch.FloatTensor,
    labels: torch.LongTensor,
    direction: str = "ref_to_policy",
) -> tuple[torch.FloatTensor, torch.BoolTensor]:
    """Compute per-token KL divergence between policy and reference distributions.

    Args:
        logits: Policy logits. Shape: (batch_size, sequence_length, vocab_size)
        reference_logits: Reference logits. Shape: (batch_size, sequence_length, vocab_size)
        labels: Labels to determine which positions to compute KL for.
                Label tokens with a value of -100 are ignored. Shape: (batch_size, sequence_length)
        direction: Which KL to compute.
            - "ref_to_policy": KL(reference || policy)
            - "policy_to_ref": KL(policy || reference)

    Returns:
        A tuple (kl, loss_mask) where:
          - kl has shape (batch_size, sequence_length-1) and is zero at masked positions.
          - loss_mask has shape (batch_size, sequence_length-1) and is True where labels != -100.
    """
    assert logits.shape[:-1] == labels.shape
    assert reference_logits.shape[:-1] == labels.shape

    labels = labels[:, 1:].clone()
    logits = logits[:, :-1, :]
    reference_logits = reference_logits[:, :-1, :]
    loss_mask = labels != -100

    policy_log_probs = logits.log_softmax(-1)
    reference_log_probs = reference_logits.log_softmax(-1)

    direction = direction.lower()
    if direction in {"ref_to_policy", "reference_to_policy", "ref"}:
        reference_probs = reference_log_probs.exp()
        kl = (reference_probs * (reference_log_probs - policy_log_probs)).sum(-1)
    elif direction in {"policy_to_ref", "policy"}:
        policy_probs = policy_log_probs.exp()
        kl = (policy_probs * (policy_log_probs - reference_log_probs)).sum(-1)
    else:
        raise ValueError(
            f"Unknown KL direction: {direction}. Use 'ref_to_policy' or 'policy_to_ref'."
        )

    kl = kl * loss_mask
    return kl, loss_mask


def compute_response_token_logps(
    logits: torch.FloatTensor,
    labels: torch.LongTensor,
    response_mask: torch.LongTensor,
) -> tuple[torch.FloatTensor, torch.BoolTensor]:
    """Gather teacher-forced token log-probs over assistant response tokens.

    The returned tensors are shifted to match next-token prediction positions:
    logits[:, t - 1] scores labels[:, t]. The mask is therefore
    response_mask[:, 1:] intersected with labels[:, 1:] != -100.
    """
    assert logits.shape[:-1] == labels.shape
    assert response_mask.shape == labels.shape

    shifted_labels = labels[:, 1:].clone()
    shifted_logits = logits[:, :-1, :]
    shifted_response_mask = response_mask[:, 1:].bool()
    valid_mask = shifted_response_mask & (shifted_labels != -100)

    shifted_labels[~valid_mask] = 0
    token_logps = torch.gather(
        shifted_logits.log_softmax(-1),
        dim=2,
        index=shifted_labels.unsqueeze(2),
    ).squeeze(2)

    return token_logps * valid_mask, valid_mask


def _validate_chunk_spans(chunk_spans: Sequence[Span], n_tokens: int):
    if n_tokens == 0:
        if chunk_spans:
            raise ValueError("Chunk spans must be empty for an empty response.")
        return
    if not chunk_spans:
        raise ValueError("Chunk spans must cover a non-empty response.")

    expected_start = 0
    for start, end in chunk_spans:
        if start != expected_start:
            raise ValueError(
                f"Chunk spans must be contiguous; expected start {expected_start}, got {start}."
            )
        if end <= start:
            raise ValueError(f"Invalid empty or reversed chunk span: {(start, end)}.")
        if end > n_tokens:
            raise ValueError(
                f"Chunk span {(start, end)} exceeds response length {n_tokens}."
            )
        expected_start = end

    if expected_start != n_tokens:
        raise ValueError(
            f"Chunk spans must cover the full response; ended at {expected_start}, "
            f"expected {n_tokens}."
        )


def compute_exact_chunk_log_ratios(
    policy_response_logps: torch.FloatTensor,
    reference_response_logps: torch.FloatTensor,
    chunk_spans: Sequence[Span],
    beta: float,
    verify_telescope: bool = True,
    atol: float = 1e-5,
) -> tuple[torch.FloatTensor, torch.FloatTensor]:
    """Compute ARC-BPO exact chunk log-ratios for one response.

    Args:
        policy_response_logps: Per-token log p_theta over response tokens only.
        reference_response_logps: Per-token log p_ref over the same response tokens.
        chunk_spans: Contiguous [start, end) spans over response-token indices.
        beta: KL temperature multiplying the policy/reference log-ratio.
        verify_telescope: If true, assert sum_i a_theta(c_i) equals the full
            sequence beta-scaled policy/reference log-ratio.
        atol: Absolute tolerance for the telescoping check.

    Returns:
        A tuple (chunk_log_ratios, sequence_log_ratio).
    """
    if policy_response_logps.dim() != 1 or reference_response_logps.dim() != 1:
        raise ValueError("ARC-BPO chunk log-ratios expect 1D response-token log-probs.")
    if policy_response_logps.shape != reference_response_logps.shape:
        raise ValueError(
            "Policy and reference response log-probs must have the same shape."
        )

    n_tokens = policy_response_logps.numel()
    _validate_chunk_spans(chunk_spans, n_tokens)

    token_log_ratio = beta * (policy_response_logps - reference_response_logps.detach())
    sequence_log_ratio = token_log_ratio.sum()

    if n_tokens == 0:
        chunk_log_ratios = token_log_ratio.new_empty((0,))
    else:
        chunk_log_ratios = torch.stack(
            [token_log_ratio[start:end].sum() for start, end in chunk_spans]
        )

    if verify_telescope:
        chunk_sum = chunk_log_ratios.sum()
        if not torch.allclose(chunk_sum, sequence_log_ratio, atol=atol, rtol=0.0):
            raise ValueError(
                "Chunk log-ratios do not telescope to the sequence log-ratio: "
                f"chunks={chunk_sum.detach().item():.8f}, "
                f"sequence={sequence_log_ratio.detach().item():.8f}."
            )

    return chunk_log_ratios, sequence_log_ratio


def winsorize_advantages(
    advantages: torch.FloatTensor,
    kappa: float = 2.0,
    eps: float = 1e-8,
) -> torch.FloatTensor:
    """Detach and robustly clip chunk advantages before shape softmax."""
    advantages = advantages.detach()
    if advantages.dim() != 1:
        raise ValueError("ARC-BPO advantage proxies must be 1D per response.")
    if advantages.numel() == 0:
        return advantages

    median = advantages.median()
    mad = (advantages - median).abs().median()
    scale = mad.clamp_min(eps)
    return advantages.clamp(median - kappa * scale, median + kappa * scale)


def _as_scalar_delta(
    delta_star,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    delta = torch.as_tensor(delta_star, device=device, dtype=dtype).detach()
    if delta.numel() != 1:
        raise ValueError("ARC-BPO delta_star must be a finite scalar per preference pair.")
    if not torch.isfinite(delta).all():
        raise ValueError("ARC-BPO delta_star must be finite.")
    return delta.reshape(())


def _uniform_shape(n_chunks: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if n_chunks <= 0:
        raise ValueError("ARC-BPO target construction requires at least one chunk.")
    return torch.full((n_chunks,), 1.0 / n_chunks, device=device, dtype=dtype)


def _advantage_shape(
    advantages: Optional[torch.FloatTensor],
    n_chunks: int,
    device: torch.device,
    dtype: torch.dtype,
    temperature: float,
    kappa: float,
    sign: float,
    fallback_to_uniform: bool,
    winsorize: bool,
) -> torch.Tensor:
    if advantages is None:
        if fallback_to_uniform:
            return _uniform_shape(n_chunks, device, dtype)
        raise ValueError("Missing ARC-BPO advantage proxy and uniform fallback is disabled.")

    advantages = torch.as_tensor(advantages, device=device, dtype=dtype).flatten()
    if advantages.numel() != n_chunks:
        raise ValueError(
            f"Expected {n_chunks} ARC-BPO advantage values, got {advantages.numel()}."
        )
    if temperature <= 0:
        raise ValueError("ARC-BPO advantage-shape temperature must be positive.")

    scores = winsorize_advantages(advantages, kappa=kappa) if winsorize else advantages.detach()
    return torch.softmax(sign * scores / temperature, dim=0).detach()


def construct_arc_bpo_one_sided_targets(
    n_chosen_chunks: int,
    n_rejected_chunks: int,
    delta_star,
    chosen_advantages: Optional[torch.FloatTensor] = None,
    rejected_advantages: Optional[torch.FloatTensor] = None,
    temperature: float = 2.0,
    kappa: float = 2.0,
    use_advantage_shape: bool = False,
    fallback_to_uniform: bool = True,
    winsorize: bool = True,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
    verify_calibration: bool = True,
    atol: float = 1e-5,
) -> tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
    """Construct detached one-sided ARC-BPO chunk targets for one pair.

    Uniform shapes are used by default. If use_advantage_shape is true, chosen
    chunks use softmax(A / T), while rejected chunks use softmax(-A / T).
    Returns (tau_w, tau_l, pi_w, rho_l).
    """
    if device is None:
        if chosen_advantages is not None:
            device = torch.as_tensor(chosen_advantages).device
        elif rejected_advantages is not None:
            device = torch.as_tensor(rejected_advantages).device
        else:
            device = torch.device("cpu")

    delta = _as_scalar_delta(delta_star, device=device, dtype=dtype)

    if use_advantage_shape:
        pi_w = _advantage_shape(
            chosen_advantages,
            n_chosen_chunks,
            device,
            dtype,
            temperature,
            kappa,
            sign=1.0,
            fallback_to_uniform=fallback_to_uniform,
            winsorize=winsorize,
        )
        rho_l = _advantage_shape(
            rejected_advantages,
            n_rejected_chunks,
            device,
            dtype,
            temperature,
            kappa,
            sign=-1.0,
            fallback_to_uniform=fallback_to_uniform,
            winsorize=winsorize,
        )
    else:
        pi_w = _uniform_shape(n_chosen_chunks, device, dtype)
        rho_l = _uniform_shape(n_rejected_chunks, device, dtype)

    pi_w = pi_w.detach()
    rho_l = rho_l.detach()
    tau_w = (0.5 * delta * pi_w).detach()
    tau_l = (-0.5 * delta * rho_l).detach()

    if verify_calibration:
        target_margin = tau_w.sum() - tau_l.sum()
        if not torch.allclose(target_margin, delta, atol=atol, rtol=0.0):
            raise ValueError(
                "ARC-BPO targets are not calibrated: "
                f"target_margin={target_margin.detach().item():.8f}, "
                f"delta_star={delta.detach().item():.8f}."
            )

    return tau_w, tau_l, pi_w, rho_l


def Q_tbpo_get_batch_logps(
    logits: torch.FloatTensor,
    reference_logits: torch.FloatTensor,
    labels: torch.LongTensor,
):
    assert logits.shape[:-1] == labels.shape
    assert reference_logits.shape[:-1] == labels.shape

    labels = labels[:, 1:].clone()
    logits = logits[:, :-1, :]
    reference_logits = reference_logits[:, :-1, :]

    loss_mask = labels != -100

    # dummy token; we'll ignore the losses on these tokens later
    labels[labels == -100] = 0

    per_token_logps = torch.gather(
        logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)
    ).squeeze(2)
    per_reference_token_logps = torch.gather(
        reference_logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)
    ).squeeze(2)

    logps_margin = per_token_logps - per_reference_token_logps

    return (
        logps_margin * loss_mask,
        per_token_logps * loss_mask,
    )


def A_tbpo_get_batch_logps(
    logits: torch.FloatTensor,
    reference_logits: torch.FloatTensor,
    labels: torch.LongTensor,
    average_log_prob: bool = False,
):
    """also return token KL"""
    pass


def _tdpo_get_batch_logps(
    logits: torch.FloatTensor,
    reference_logits: torch.FloatTensor,
    labels: torch.LongTensor,
    average_log_prob: bool = False,
):
    """Compute the kl divergence/log probabilities of the given labels under the given logits.

    Args:
        logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
        reference_logits: Logits of the reference model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
        labels: Labels for which to compute the log probabilities. Label tokens with a value of -100 are ignored. Shape: (batch_size, sequence_length)
        average_log_prob: If True, return the average log probability per (non-masked) token. Otherwise, return the sum of the log probabilities of the (non-masked) tokens.

    Returns:
        Several tensors of shape (batch_size,) containing the average/sum kl divergence/log probabilities of the given labels under the given logits.
    """
    assert logits.shape[:-1] == labels.shape
    assert reference_logits.shape[:-1] == labels.shape

    labels = labels[:, 1:].clone()
    logits = logits[:, :-1, :]
    reference_logits = reference_logits[:, :-1, :]

    loss_mask = labels != -100

    # dummy token; we'll ignore the losses on these tokens later
    labels[labels == -100] = 0

    vocab_logps = logits.log_softmax(-1)

    reference_vocab_ps = reference_logits.softmax(-1)
    reference_vocab_logps = reference_vocab_ps.log()

    per_position_kl = (reference_vocab_ps * (reference_vocab_logps - vocab_logps)).sum(-1)
    per_token_logps = torch.gather(vocab_logps, dim=2, index=labels.unsqueeze(2)).squeeze(2)
    per_reference_token_logps = torch.gather(
        reference_vocab_logps, dim=2, index=labels.unsqueeze(2)
    ).squeeze(2)

    logps_margin = per_token_logps - per_reference_token_logps

    if average_log_prob:
        return (
            (logps_margin * loss_mask).sum(-1) / loss_mask.sum(-1),
            (per_position_kl * loss_mask).sum(-1) / loss_mask.sum(-1),
            (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1),
        )
    else:
        return (
            (logps_margin * loss_mask).sum(-1),
            (per_position_kl * loss_mask).sum(-1),
            (per_token_logps * loss_mask).sum(-1),
        )


def _get_batch_logps(
    logits: torch.FloatTensor,
    labels: torch.LongTensor,
    average_log_prob: bool = False,
    token_level: bool = False,
) -> torch.FloatTensor:
    """Compute the log probabilities of the given labels under the given logits.

    Args:
        logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
        labels: Labels for which to compute the log probabilities. Label tokens with a value of -100 are ignored. Shape: (batch_size, sequence_length)
        average_log_prob: If True, return the average log probability per (non-masked) token. Otherwise, return the sum of the log probabilities of the (non-masked) tokens.

    Returns:
        A tensor of shape (batch_size,) containing the average/sum log probabilities of the given labels under the given logits.
    """
    assert logits.shape[:-1] == labels.shape

    labels = labels[:, 1:].clone()
    logits = logits[:, :-1, :]
    loss_mask = labels != -100

    # dummy token; we'll ignore the losses on these tokens later
    labels[labels == -100] = 0

    per_token_logps = torch.gather(
        logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)
    ).squeeze(2)
    # import ipdb; ipdb.set_trace()
    batch_logps = (per_token_logps * loss_mask).sum(-1)

    if average_log_prob:
        return batch_logps / loss_mask.sum(-1)
    else:
        return batch_logps


def _get_batch_logps_tisdpo(
    logits: torch.FloatTensor,
    reference_logits: torch.FloatTensor,
    labels: torch.LongTensor,
    weights: torch.FloatTensor = None,
    average_log_prob: bool = False,
) -> torch.FloatTensor:
    """Compute the log probabilities of the given labels under the given logits.

    Args:
        logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
        reference_logits: Logits of the reference model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
        labels: Labels for which to compute the log probabilities. Label tokens with a value of -100 are ignored. Shape: (batch_size, sequence_length)
        weights: Weights for each token. Shape: (batch_size, sequence_length)
        average_log_prob: If True, return the average log probability per
        (non-masked) token. Otherwise, return the sum of the log probabilities of the (non-masked) tokens.
        token_level: If True, return the log probability per (non-masked) token. Otherwise, return the sum of the log probabilities of the (non-masked) tokens.

    Returns:
        A tensor of shape (batch_size,) containing the average/sum log probabilities of the given labels under the given logits.
    """
    assert logits.shape[:-1] == labels.shape

    labels = labels[:, 1:].clone()
    logits = logits[:, :-1, :]
    reference_logits = reference_logits[:, :-1, :]

    loss_mask = labels != -100

    labels[labels == -100] = 0

    vocab_ps = logits.softmax(-1)
    vocab_logps = vocab_ps.log()

    reference_vocab_ps = reference_logits.softmax(-1)
    reference_vocab_logps = reference_vocab_ps.log()

    per_position_kl = (vocab_ps * (vocab_logps - reference_vocab_logps)).sum(-1)

    per_token_logps = torch.gather(vocab_logps, dim=2, index=labels.unsqueeze(2)).squeeze(2)
    per_reference_token_logps = torch.gather(
        reference_vocab_logps, dim=2, index=labels.unsqueeze(2)
    ).squeeze(2)

    logps_margin = per_token_logps - per_reference_token_logps
    weights = weights[:, 1:].clone()

    if average_log_prob:
        return (
            (logps_margin * weights * loss_mask).sum(-1) / loss_mask.sum(-1),
            (per_position_kl * weights * loss_mask).sum(-1) / loss_mask.sum(-1),
            (per_token_logps * weights * loss_mask).sum(-1) / loss_mask.sum(-1),
        )
    else:
        return (
            (logps_margin * weights * loss_mask).sum(-1),
            (per_position_kl * weights * loss_mask).sum(-1),
            (per_token_logps * weights * loss_mask).sum(-1),
        )
