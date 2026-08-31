from typing import Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from loss.h_function import HFunc
from loss.loss_utils import construct_arc_bpo_one_sided_targets


def bregman_loss(
    log_R: torch.Tensor,
    loss_mask: torch.Tensor,
    h_func: HFunc,
    label_smoothing: float = 0.0,
):
    per_token = h_func.loss_from_logR(log_R)
    if label_smoothing:
        per_token_flipped = h_func.loss_from_logR(-log_R)
        per_token = per_token * (1 - label_smoothing) + per_token_flipped * label_smoothing
    per_token_loss = per_token * loss_mask

    denom = loss_mask.sum(dim=1).clamp_min(1.0)
    return per_token_loss.sum(dim=1) / denom, per_token_loss


def sba_h(R: torch.Tensor, lam: float = 1.0, s: float = 4.0) -> torch.Tensor:
    """Scaled Basu's power-divergence generator for ARC-BPO."""
    if lam < 0:
        raise ValueError("ARC-BPO SBA requires lambda >= 0.")
    if s <= 0:
        raise ValueError("ARC-BPO SBA requires scale s > 0.")
    if lam == 0:
        # Exact lambda -> 0 limit of
        # (R^(1+lambda) - R) / (s*lambda*(1+lambda)).
        return R * torch.log(R) / s
    return (R.pow(1.0 + lam) - R) / (s * lam * (1.0 + lam))


def sba_h_prime(R: torch.Tensor, lam: float = 1.0, s: float = 4.0) -> torch.Tensor:
    if lam < 0:
        raise ValueError("ARC-BPO SBA requires lambda >= 0.")
    if s <= 0:
        raise ValueError("ARC-BPO SBA requires scale s > 0.")
    if lam == 0:
        return (torch.log(R) + 1.0) / s
    return (((1.0 + lam) * R.pow(lam)) - 1.0) / (s * lam * (1.0 + lam))


def bregman_sba(
    target_ratio: torch.Tensor,
    model_ratio: torch.Tensor,
    lam: float = 1.0,
    s: float = 4.0,
) -> torch.Tensor:
    """Compute D_h(target_ratio, model_ratio) using the SBA generator."""
    if target_ratio.shape != model_ratio.shape:
        raise ValueError("ARC-BPO target and model ratios must have the same shape.")
    return (
        sba_h(target_ratio, lam=lam, s=s)
        - sba_h(model_ratio, lam=lam, s=s)
        - sba_h_prime(model_ratio, lam=lam, s=s) * (target_ratio - model_ratio)
    )


def arc_bpo_pair_loss(
    chosen_chunk_log_ratios: torch.FloatTensor,
    rejected_chunk_log_ratios: torch.FloatTensor,
    delta_star,
    chosen_advantages: Optional[torch.FloatTensor] = None,
    rejected_advantages: Optional[torch.FloatTensor] = None,
    temperature: float = 2.0,
    kappa: float = 2.0,
    use_advantage_shape: bool = False,
    fallback_to_uniform: bool = True,
    winsorize_advantages: bool = True,
    lam: float = 1.0,
    s: float = 4.0,
    exp_clip: float = 30.0,
    verify_calibration: bool = True,
) -> tuple[torch.FloatTensor, dict]:
    """ARC-BPO loss for one preference pair.

    The chosen and rejected chunk sets are matched only to their own one-sided
    targets. This function does not perform token-to-token matching, cross-side
    chunk pairing, correction terms, or value-head estimation.
    """
    if chosen_chunk_log_ratios.dim() != 1 or rejected_chunk_log_ratios.dim() != 1:
        raise ValueError("ARC-BPO chunk log-ratios must be 1D tensors per response.")
    if chosen_chunk_log_ratios.numel() == 0 or rejected_chunk_log_ratios.numel() == 0:
        raise ValueError("ARC-BPO requires at least one chosen and rejected chunk.")
    if exp_clip <= 0:
        raise ValueError("ARC-BPO exp_clip must be positive.")

    device = chosen_chunk_log_ratios.device
    dtype = chosen_chunk_log_ratios.dtype
    rejected_chunk_log_ratios = rejected_chunk_log_ratios.to(device=device, dtype=dtype)

    tau_w, tau_l, pi_w, rho_l = construct_arc_bpo_one_sided_targets(
        chosen_chunk_log_ratios.numel(),
        rejected_chunk_log_ratios.numel(),
        delta_star,
        chosen_advantages=chosen_advantages,
        rejected_advantages=rejected_advantages,
        temperature=temperature,
        kappa=kappa,
        use_advantage_shape=use_advantage_shape,
        fallback_to_uniform=fallback_to_uniform,
        winsorize=winsorize_advantages,
        device=device,
        dtype=dtype,
        verify_calibration=verify_calibration,
    )

    chosen_model_ratio = torch.exp(-chosen_chunk_log_ratios.clamp(-exp_clip, exp_clip))
    rejected_model_ratio = torch.exp(-rejected_chunk_log_ratios.clamp(-exp_clip, exp_clip))
    chosen_target_ratio = torch.exp(-tau_w.clamp(-exp_clip, exp_clip))
    rejected_target_ratio = torch.exp(-tau_l.clamp(-exp_clip, exp_clip))

    chosen_losses = bregman_sba(
        chosen_target_ratio,
        chosen_model_ratio,
        lam=lam,
        s=s,
    )
    rejected_losses = bregman_sba(
        rejected_target_ratio,
        rejected_model_ratio,
        lam=lam,
        s=s,
    )
    loss = chosen_losses.mean() + rejected_losses.mean()

    outputs = {
        "tau_w": tau_w,
        "tau_l": tau_l,
        "pi_w": pi_w,
        "rho_l": rho_l,
        "chosen_model_ratio": chosen_model_ratio,
        "rejected_model_ratio": rejected_model_ratio,
        "chosen_target_ratio": chosen_target_ratio,
        "rejected_target_ratio": rejected_target_ratio,
        "chosen_losses": chosen_losses,
        "rejected_losses": rejected_losses,
        "target_margin": tau_w.sum() - tau_l.sum(),
        "model_margin": chosen_chunk_log_ratios.sum() - rejected_chunk_log_ratios.sum(),
    }
    return loss, outputs


def arc_bpo_batch_loss(
    chosen_chunk_log_ratios: Sequence[torch.FloatTensor],
    rejected_chunk_log_ratios: Sequence[torch.FloatTensor],
    delta_star,
    chosen_advantages: Optional[Sequence[Optional[torch.FloatTensor]]] = None,
    rejected_advantages: Optional[Sequence[Optional[torch.FloatTensor]]] = None,
    **kwargs,
) -> tuple[torch.FloatTensor, list[dict]]:
    """Average ARC-BPO pair losses over a variable-chunk batch."""
    batch_size = len(chosen_chunk_log_ratios)
    if len(rejected_chunk_log_ratios) != batch_size:
        raise ValueError("Chosen and rejected ARC-BPO batches must have the same length.")

    if torch.is_tensor(delta_star) and delta_star.numel() > 1:
        deltas = list(delta_star)
    elif isinstance(delta_star, Sequence) and not isinstance(delta_star, (str, bytes)):
        deltas = list(delta_star)
    else:
        deltas = [delta_star] * batch_size
    if len(deltas) != batch_size:
        raise ValueError("ARC-BPO delta_star must be scalar or one value per pair.")

    if chosen_advantages is None:
        chosen_advantages = [None] * batch_size
    if rejected_advantages is None:
        rejected_advantages = [None] * batch_size
    if len(chosen_advantages) != batch_size or len(rejected_advantages) != batch_size:
        raise ValueError("ARC-BPO advantage batches must match batch size.")

    losses = []
    outputs = []
    for idx in range(batch_size):
        pair_loss, pair_outputs = arc_bpo_pair_loss(
            chosen_chunk_log_ratios[idx],
            rejected_chunk_log_ratios[idx],
            deltas[idx],
            chosen_advantages=chosen_advantages[idx],
            rejected_advantages=rejected_advantages[idx],
            **kwargs,
        )
        losses.append(pair_loss)
        outputs.append(pair_outputs)

    return torch.stack(losses).mean(), outputs


def tdpo_loss(
    chosen_logps_margin: torch.FloatTensor,
    rejected_logps_margin: torch.FloatTensor,
    chosen_position_kl: torch.FloatTensor,
    rejected_position_kl: torch.FloatTensor,
    beta: float,
    alpha: float = 0.5,
    if_tdpo2: bool = True,
) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
    """Compute the TDPO loss for a batch of policy and reference model log probabilities.

    Args:
        chosen_logps_margin: The difference of log probabilities between the policy model and the reference model for the chosen responses. Shape: (batch_size,)
        rejected_logps_margin: The difference of log probabilities between the policy model and the reference model for the rejected responses. Shape: (batch_size,)
        chosen_position_kl: The difference of sequential kl divergence between the policy model and the reference model for the chosen responses. Shape: (batch_size,)
        rejected_position_kl: The difference of sequential kl divergence between the policy model and the reference model for the rejected responses. Shape: (batch_size,)
        beta: Temperature parameter for the TDPO loss, typically something in the range of 0.1 to 0.5. We ignore the reference model as beta -> 0.
        alpha: Temperature parameter for the TDPO loss, used to adjust the impact of sequential kl divergence.
        if_tdpo2: Determine whether to use method TDPO2, default is True; if False, then use method TDPO1.

    Returns:
        A tuple of two tensors: (losses, rewards).
        The losses tensor contains the TDPO loss for each example in the batch.
        The rewards tensors contain the rewards for response pair.
    """

    chosen_values = chosen_logps_margin + chosen_position_kl
    rejected_values = rejected_logps_margin + rejected_position_kl

    chosen_rejected_logps_margin = chosen_logps_margin - rejected_logps_margin

    if not if_tdpo2:
        logits = chosen_rejected_logps_margin - (rejected_position_kl - chosen_position_kl)  # tdpo1
    else:
        logits = chosen_rejected_logps_margin - alpha * (
            rejected_position_kl - chosen_position_kl.detach()
        )  # tdpo2
    losses = -F.logsigmoid(beta * logits)

    chosen_rewards = beta * chosen_values.detach()
    rejected_rewards = beta * rejected_values.detach()

    return losses, chosen_rewards, rejected_rewards


def tisdpo_loss(
    chosen_logps_margin: torch.FloatTensor,
    rejected_logps_margin: torch.FloatTensor,
    chosen_position_kl: torch.FloatTensor,
    rejected_position_kl: torch.FloatTensor,
    beta: float,
    alpha: float = 0.5,
    token_level: bool = False,
) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
    if token_level:
        chosen_values = chosen_logps_margin - chosen_position_kl
        rejected_values = rejected_logps_margin - rejected_position_kl
    else:
        chosen_values = chosen_logps_margin
        rejected_values = rejected_logps_margin

    chosen_rejected_logps_margin = chosen_logps_margin - rejected_logps_margin

    if token_level:
        logits = chosen_rejected_logps_margin - alpha * (chosen_position_kl - rejected_position_kl)
    else:
        logits = chosen_rejected_logps_margin

    losses = -F.logsigmoid(beta * logits)

    chosen_rewards = beta * chosen_values.detach()
    rejected_rewards = beta * rejected_values.detach()

    return losses, chosen_rewards, rejected_rewards


def preference_loss(
    policy_chosen_logps: torch.FloatTensor,
    policy_rejected_logps: torch.FloatTensor,
    reference_chosen_logps: torch.FloatTensor,
    reference_rejected_logps: torch.FloatTensor,
    beta: float,
    label_smoothing: float = 0.0,
    ipo: bool = False,
    reference_free: bool = False,
) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
    """Compute the DPO loss for a batch of policy and reference model log probabilities.

    Args:
        policy_chosen_logps: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
        policy_rejected_logps: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)
        reference_chosen_logps: Log probabilities of the reference model for the chosen responses. Shape: (batch_size,)
        reference_rejected_logps: Log probabilities of the reference model for the rejected responses. Shape: (batch_size,)
        beta: Temperature parameter for the DPO loss, typically something in the range of 0.1 to 0.5. We ignore the reference model as beta -> 0.
        label_smoothing: conservativeness for DPO loss, which assumes that preferences are noisy (flipped with probability label_smoothing)
        ipo: If True, use the IPO loss instead of the DPO loss.
        reference_free: If True, we ignore the _provided_ reference model and implicitly use a reference model that assigns equal probability to all responses.

    Returns:
        A tuple of three tensors: (losses, chosen_rewards, rejected_rewards).
        The losses tensor contains the DPO loss for each example in the batch.
        The chosen_rewards and rejected_rewards tensors contain the rewards for the chosen and rejected responses, respectively.
    """
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = reference_chosen_logps - reference_rejected_logps

    if reference_free:
        ref_logratios = 0

    logits = pi_logratios - ref_logratios  # also known as h_{\pi_\theta}^{y_w,y_l}

    if ipo:
        losses = (logits - 1 / (2 * beta)) ** 2  # Eq. 17 of https://arxiv.org/pdf/2310.12036v2.pdf
    else:
        # Eq. 3 https://ericmitchell.ai/cdpo.pdf; label_smoothing=0 gives original DPO (Eq. 7 of https://arxiv.org/pdf/2305.18290.pdf)
        losses = (
            -F.logsigmoid(beta * logits) * (1 - label_smoothing)
            - F.logsigmoid(-beta * logits) * label_smoothing
        )

    chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps).detach()
    rejected_rewards = beta * (policy_rejected_logps - reference_rejected_logps).detach()

    return losses, chosen_rewards, rejected_rewards
