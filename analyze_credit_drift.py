#!/usr/bin/env python3
"""Credit-conditioned policy-drift analysis for ARC-BPO.

The script evaluates an existing ARC-BPO checkpoint on held-out preference
pairs.  It reuses the training tokenizer/chunker, computes exact observed-token
chunk log-ratios and full-vocabulary KL(policy || reference), then produces the
reviewer-facing grouped, bootstrap, correlation, and plotting artifacts.

No training or gradient computation is performed.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import math
import os
import random
import statistics
import warnings
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch

from loss.loss_utils import (
    compute_exact_chunk_log_ratios,
    construct_arc_bpo_one_sided_targets,
)


GROUP_ORDER = ("Low", "Medium", "High")
CHUNK_FIELDS = (
    "example_id",
    "example_index",
    "dataset_index",
    "side",
    "chunk_id",
    "start_token",
    "end_token",
    "num_tokens",
    "chunk_text",
    "chunk_logratio",
    "allocation_weight",
    "tau",
    "credit_magnitude",
    "policy_reference_kl",
    "credit_group",
    "allocation_group",
    "response_num_chunks",
    "credit_source",
)
GROUP_FIELDS = (
    "credit_group",
    "num_chunks",
    "num_examples",
    "mean_credit_magnitude",
    "mean_allocation_weight",
    "mean_policy_reference_kl",
    "kl_ci_lower",
    "kl_ci_upper",
    "bootstrap_valid_replicates",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze ARC-BPO credit-conditioned policy drift on held-out data."
    )
    parser.add_argument(
        "--policy_model",
        required=True,
        help=(
            "Merged ARC-BPO model, a local PEFT adapter/LATEST directory, or the "
            "base model when --policy_adapter is supplied."
        ),
    )
    parser.add_argument(
        "--policy_adapter",
        default=None,
        help="Optional explicit local/HF PEFT adapter. --policy_model is then the base model.",
    )
    parser.add_argument(
        "--policy_base_model",
        default=None,
        help="Override the base model recorded in an auto-detected adapter config.",
    )
    parser.add_argument(
        "--reference_model",
        required=True,
        help="Exact frozen SFT/reference checkpoint used by the ARC-BPO run.",
    )
    parser.add_argument(
        "--tokenizer",
        default=None,
        help="Tokenizer path/repo. Defaults to the adapter base or policy model.",
    )
    parser.add_argument("--dataset", required=True, help="HF dataset id or local JSON/JSONL file.")
    parser.add_argument("--split", default="test", help="Held-out dataset split.")
    parser.add_argument(
        "--dataset_revision",
        default=None,
        help="Optional immutable Hugging Face dataset revision/commit SHA.",
    )
    parser.add_argument(
        "--allow_train_split",
        action="store_true",
        help="Explicitly allow a split containing 'train' (not recommended for reviewer analysis).",
    )
    parser.add_argument("--output_dir", default="outputs/credit_drift")

    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--delta0", type=float, default=2.5)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--kappa", type=float, default=2.0)
    parser.add_argument(
        "--allocation_mode",
        choices=("logratio", "uniform", "dataset_score"),
        default="logratio",
        help=(
            "Credit source: the reviewer plan's post-hoc chunk log-ratio, the public "
            "training scripts' uniform target, or the training data score proxy."
        ),
    )
    parser.add_argument(
        "--fallback_to_uniform_shape",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="For dataset_score mode, fall back to uniform when scores are unavailable.",
    )
    parser.add_argument("--min_tokens_per_chunk", type=int, default=4)
    parser.add_argument("--max_tokens_per_chunk", type=int, default=64)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument(
        "--max_examples",
        type=int,
        default=0,
        help="Maximum valid held-out pairs; 0 means all available pairs.",
    )
    parser.add_argument("--shuffle", action="store_true", help="Shuffle held-out rows first.")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--dtype",
        choices=("auto", "float16", "bfloat16", "float32"),
        default="auto",
    )
    parser.add_argument(
        "--device_map",
        default="auto",
        help="Transformers device map: auto/balanced/sequential/cpu/cuda[:N].",
    )
    parser.add_argument(
        "--attn_implementation",
        choices=("auto", "eager", "sdpa", "flash_attention_2"),
        default="auto",
    )
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument(
        "--reuse_base_for_reference",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "For a PEFT policy whose base equals --reference_model, disable the adapter "
            "for reference logits instead of loading a second 8B model."
        ),
    )
    parser.add_argument(
        "--kl_device",
        default="auto",
        help="Device for blockwise full-vocabulary KL; auto chooses CUDA when available.",
    )
    parser.add_argument(
        "--kl_token_batch_size",
        type=int,
        default=8,
        help="Number of response positions per FP32 full-vocabulary KL block.",
    )
    parser.add_argument(
        "--kl_negative_tolerance",
        type=float,
        default=1e-6,
        help="Abort if numerical full-vocabulary KL falls below this negative tolerance.",
    )

    parser.add_argument("--bootstrap_iterations", type=int, default=2000)
    parser.add_argument("--confidence_level", type=float, default=0.95)
    parser.add_argument(
        "--scatter_max_points",
        type=int,
        default=20000,
        help="Maximum points rendered in the continuous-credit scatter plot.",
    )
    return parser.parse_args()


def _json_default(value: Any):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    raise TypeError(f"Cannot JSON-serialize {type(value).__name__}")


def _write_json(path: Path, payload: Dict[str, Any]):
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, default=_json_default)
        handle.write("\n")


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]], fields: Sequence[str]):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _mean(values: Iterable[float]) -> Optional[float]:
    values = list(values)
    return statistics.fmean(values) if values else None


def empirical_quantile(values: Sequence[float], probability: float) -> float:
    """Linear empirical quantile, matching the usual (n-1)*q convention."""
    if not values:
        raise ValueError("Cannot compute a quantile of an empty sequence.")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("Quantile probability must be in [0, 1].")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * probability
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def assign_tertile_groups(
    rows: Sequence[Dict[str, Any]],
    value_field: str,
    group_field: str,
) -> Dict[str, float]:
    """Assign fixed value-threshold tertiles without forcing tied values apart."""
    values = [float(row[value_field]) for row in rows]
    q1 = empirical_quantile(values, 1.0 / 3.0)
    q2 = empirical_quantile(values, 2.0 / 3.0)
    for row in rows:
        value = float(row[value_field])
        if value <= q1:
            group = "Low"
        elif value <= q2:
            group = "Medium"
        else:
            group = "High"
        row[group_field] = group
    return {"lower_tertile": q1, "upper_tertile": q2}


def bootstrap_group_kl(
    rows: Sequence[Dict[str, Any]],
    iterations: int,
    confidence_level: float,
    seed: int,
    group_field: str = "credit_group",
) -> Dict[str, Dict[str, Any]]:
    """Cluster bootstrap group means by preference-pair example_index."""
    if iterations <= 0:
        return {
            group: {"lower": None, "upper": None, "valid_replicates": 0}
            for group in GROUP_ORDER
        }
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be strictly between 0 and 1.")

    cluster_summaries: Dict[int, Dict[str, Tuple[float, int]]] = {}
    per_cluster: Dict[int, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        per_cluster[int(row["example_index"])][str(row[group_field])].append(
            float(row["policy_reference_kl"])
        )
    for cluster, grouped in per_cluster.items():
        cluster_summaries[cluster] = {
            group: (sum(values), len(values)) for group, values in grouped.items()
        }

    cluster_ids = sorted(cluster_summaries)
    if not cluster_ids:
        raise ValueError("Cannot bootstrap an empty chunk table.")

    rng = random.Random(seed)
    replicates: Dict[str, List[float]] = {group: [] for group in GROUP_ORDER}
    for _ in range(iterations):
        sampled = [rng.choice(cluster_ids) for _ in cluster_ids]
        sums = {group: 0.0 for group in GROUP_ORDER}
        counts = {group: 0 for group in GROUP_ORDER}
        for cluster in sampled:
            for group, (group_sum, group_count) in cluster_summaries[cluster].items():
                if group in sums:
                    sums[group] += group_sum
                    counts[group] += group_count
        for group in GROUP_ORDER:
            if counts[group] > 0:
                replicates[group].append(sums[group] / counts[group])

    alpha = (1.0 - confidence_level) / 2.0
    output: Dict[str, Dict[str, Any]] = {}
    for group in GROUP_ORDER:
        values = replicates[group]
        output[group] = {
            "lower": empirical_quantile(values, alpha) if values else None,
            "upper": empirical_quantile(values, 1.0 - alpha) if values else None,
            "valid_replicates": len(values),
        }
    return output


def grouped_credit_summary(
    rows: Sequence[Dict[str, Any]],
    bootstrap_iterations: int,
    confidence_level: float,
    seed: int,
) -> List[Dict[str, Any]]:
    bootstrap = bootstrap_group_kl(
        rows,
        iterations=bootstrap_iterations,
        confidence_level=confidence_level,
        seed=seed,
    )
    result = []
    for group in GROUP_ORDER:
        selected = [row for row in rows if row.get("credit_group") == group]
        ci = bootstrap[group]
        result.append(
            {
                "credit_group": group,
                "num_chunks": len(selected),
                "num_examples": len({int(row["example_index"]) for row in selected}),
                "mean_credit_magnitude": _mean(
                    float(row["credit_magnitude"]) for row in selected
                ),
                "mean_allocation_weight": _mean(
                    float(row["allocation_weight"]) for row in selected
                ),
                "mean_policy_reference_kl": _mean(
                    float(row["policy_reference_kl"]) for row in selected
                ),
                "kl_ci_lower": ci["lower"],
                "kl_ci_upper": ci["upper"],
                "bootstrap_valid_replicates": ci["valid_replicates"],
            }
        )
    return result


def spearman_result(rows: Sequence[Dict[str, Any]], x_field: str) -> Dict[str, Any]:
    from scipy.stats import spearmanr

    x = [float(row[x_field]) for row in rows]
    y = [float(row["policy_reference_kl"]) for row in rows]
    if len(x) < 2 or len(set(x)) < 2 or len(set(y)) < 2:
        return {"rho": None, "p_value": None, "num_chunks": len(x), "reason": "constant"}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = spearmanr(x, y)
    rho = float(result.statistic)
    p_value = float(result.pvalue)
    if not math.isfinite(rho) or not math.isfinite(p_value):
        return {"rho": None, "p_value": None, "num_chunks": len(x), "reason": "non_finite"}
    return {"rho": rho, "p_value": p_value, "num_chunks": len(x)}


def build_credit_targets(
    chosen_chunk_logratios: torch.Tensor,
    rejected_chunk_logratios: torch.Tensor,
    allocation_mode: str,
    delta0: float,
    temperature: float,
    kappa: float,
    chosen_dataset_proxy: Optional[Sequence[float]] = None,
    rejected_dataset_proxy: Optional[Sequence[float]] = None,
    fallback_to_uniform: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if allocation_mode == "logratio":
        chosen_advantages = chosen_chunk_logratios.detach()
        rejected_advantages = rejected_chunk_logratios.detach()
        use_advantage_shape = True
    elif allocation_mode == "dataset_score":
        chosen_advantages = chosen_dataset_proxy
        rejected_advantages = rejected_dataset_proxy
        use_advantage_shape = True
    elif allocation_mode == "uniform":
        chosen_advantages = None
        rejected_advantages = None
        use_advantage_shape = False
    else:
        raise ValueError(f"Unknown allocation_mode: {allocation_mode}")

    return construct_arc_bpo_one_sided_targets(
        chosen_chunk_logratios.numel(),
        rejected_chunk_logratios.numel(),
        delta0,
        chosen_advantages=chosen_advantages,
        rejected_advantages=rejected_advantages,
        temperature=temperature,
        kappa=kappa,
        use_advantage_shape=use_advantage_shape,
        fallback_to_uniform=fallback_to_uniform,
        device=chosen_chunk_logratios.device,
        dtype=chosen_chunk_logratios.dtype,
        verify_calibration=True,
    )


def _normalized_model_id(value: str) -> str:
    path = Path(value)
    if path.exists():
        return os.path.normcase(str(path.resolve()))
    return value.rstrip("/").lower()


def _local_adapter_path(policy_model: str) -> Optional[str]:
    path = Path(policy_model)
    if not path.exists():
        return None
    if (path / "adapter_config.json").is_file():
        return str(path)
    if (path / "adapter" / "adapter_config.json").is_file():
        return str(path / "adapter")
    return None


def _model_load_kwargs(args: argparse.Namespace) -> Dict[str, Any]:
    dtype: Any = "auto" if args.dtype == "auto" else getattr(torch, args.dtype)
    if args.device_map in {"auto", "balanced", "balanced_low_0", "sequential"}:
        device_map: Any = args.device_map
    else:
        device_map = {"": args.device_map}
    kwargs: Dict[str, Any] = {
        "torch_dtype": dtype,
        "device_map": device_map,
        "low_cpu_mem_usage": True,
        "trust_remote_code": args.trust_remote_code,
    }
    if args.attn_implementation != "auto":
        kwargs["attn_implementation"] = args.attn_implementation
    return kwargs


def _read_adapter_base(adapter_path: str, trust_remote_code: bool) -> str:
    local_config = Path(adapter_path) / "adapter_config.json"
    if local_config.is_file():
        with local_config.open("r", encoding="utf-8") as handle:
            config = json.load(handle)
        base = config.get("base_model_name_or_path")
    else:
        from peft import PeftConfig

        base = PeftConfig.from_pretrained(adapter_path).base_model_name_or_path
    if not base:
        raise ValueError(
            f"Adapter {adapter_path!r} does not record base_model_name_or_path; "
            "pass --policy_base_model explicitly."
        )
    return str(base)


def _input_device(model: torch.nn.Module) -> torch.device:
    embedding = model.get_input_embeddings()
    device = embedding.weight.device
    if device.type != "meta":
        return device
    for parameter in model.parameters():
        if parameter.device.type != "meta":
            return parameter.device
    raise RuntimeError("Could not determine a real input device for the model.")


@dataclass
class ModelBundle:
    policy: torch.nn.Module
    reference: Optional[torch.nn.Module]
    tokenizer: Any
    shared_adapter_reference: bool
    adapter_path: Optional[str]
    policy_base_model: Optional[str]

    def _selected_logits(
        self,
        model: torch.nn.Module,
        input_ids: Sequence[int],
        attention_mask: Sequence[int],
        prediction_positions: torch.Tensor,
        disable_adapter: bool = False,
    ) -> torch.Tensor:
        device = _input_device(model)
        ids = torch.tensor([input_ids], dtype=torch.long, device=device)
        mask = torch.tensor([attention_mask], dtype=torch.long, device=device)
        context = model.disable_adapter() if disable_adapter else contextlib.nullcontext()
        with torch.inference_mode(), context:
            output = model(ids, attention_mask=mask, use_cache=False)
            positions = prediction_positions.to(output.logits.device)
            selected = output.logits[0].index_select(0, positions).detach().cpu()
        del output, ids, mask
        return selected

    def selected_policy_logits(
        self,
        input_ids: Sequence[int],
        attention_mask: Sequence[int],
        prediction_positions: torch.Tensor,
    ) -> torch.Tensor:
        return self._selected_logits(
            self.policy, input_ids, attention_mask, prediction_positions, disable_adapter=False
        )

    def selected_reference_logits(
        self,
        input_ids: Sequence[int],
        attention_mask: Sequence[int],
        prediction_positions: torch.Tensor,
    ) -> torch.Tensor:
        if self.shared_adapter_reference:
            return self._selected_logits(
                self.policy,
                input_ids,
                attention_mask,
                prediction_positions,
                disable_adapter=True,
            )
        if self.reference is None:
            raise RuntimeError("A separate reference model was not loaded.")
        return self._selected_logits(
            self.reference,
            input_ids,
            attention_mask,
            prediction_positions,
            disable_adapter=False,
        )


def load_model_bundle(args: argparse.Namespace) -> ModelBundle:
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    adapter_path = args.policy_adapter or _local_adapter_path(args.policy_model)
    model_kwargs = _model_load_kwargs(args)
    policy_base_model: Optional[str] = None

    if adapter_path:
        if args.policy_adapter:
            policy_base_model = args.policy_base_model or args.policy_model
        else:
            policy_base_model = args.policy_base_model or _read_adapter_base(
                adapter_path, args.trust_remote_code
            )
        print(f"Loading policy base model: {policy_base_model}")
        base_policy = AutoModelForCausalLM.from_pretrained(policy_base_model, **model_kwargs)
        print(f"Loading ARC-BPO adapter: {adapter_path}")
        policy = PeftModel.from_pretrained(base_policy, adapter_path, is_trainable=False)
    else:
        print(f"Loading merged ARC-BPO policy: {args.policy_model}")
        policy = AutoModelForCausalLM.from_pretrained(args.policy_model, **model_kwargs)

    shared_adapter_reference = bool(
        adapter_path
        and args.reuse_base_for_reference
        and policy_base_model
        and _normalized_model_id(policy_base_model)
        == _normalized_model_id(args.reference_model)
    )
    reference = None
    if shared_adapter_reference:
        print("Reference equals the PEFT base; reference logits will use disable_adapter().")
    else:
        print(f"Loading frozen reference model: {args.reference_model}")
        reference = AutoModelForCausalLM.from_pretrained(args.reference_model, **model_kwargs)

    tokenizer_source = args.tokenizer or policy_base_model or args.policy_model
    print(f"Loading shared tokenizer: {tokenizer_source}")
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source,
        trust_remote_code=args.trust_remote_code,
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer has neither pad_token_id nor eos_token_id.")
        tokenizer.pad_token_id = tokenizer.eos_token_id

    policy.eval()
    policy.requires_grad_(False)
    if reference is not None:
        reference.eval()
        reference.requires_grad_(False)

    policy_vocab = int(policy.get_output_embeddings().weight.shape[0])
    reference_vocab = (
        policy_vocab
        if shared_adapter_reference
        else int(reference.get_output_embeddings().weight.shape[0])
    )
    if policy_vocab != reference_vocab:
        raise ValueError(
            f"Policy/reference vocabulary sizes differ: {policy_vocab} vs {reference_vocab}."
        )
    if len(tokenizer) > policy_vocab:
        raise ValueError(
            f"Tokenizer contains {len(tokenizer)} entries but model vocabulary has {policy_vocab}."
        )

    return ModelBundle(
        policy=policy,
        reference=reference,
        tokenizer=tokenizer,
        shared_adapter_reference=shared_adapter_reference,
        adapter_path=adapter_path,
        policy_base_model=policy_base_model,
    )


def _resolve_kl_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def observed_logps_and_forward_kl(
    policy_logits: torch.Tensor,
    reference_logits: torch.Tensor,
    observed_token_ids: torch.Tensor,
    kl_device: torch.device,
    token_batch_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute observed-token log-probs and full-vocab KL(policy || reference)."""
    if policy_logits.shape != reference_logits.shape:
        raise ValueError(
            f"Policy/reference selected logits differ: {policy_logits.shape} vs "
            f"{reference_logits.shape}."
        )
    if policy_logits.dim() != 2:
        raise ValueError("Selected logits must have shape [response_tokens, vocabulary].")
    if policy_logits.shape[0] != observed_token_ids.numel():
        raise ValueError("Observed-token count does not match selected prediction positions.")
    if token_batch_size <= 0:
        raise ValueError("kl_token_batch_size must be positive.")

    policy_observed: List[torch.Tensor] = []
    reference_observed: List[torch.Tensor] = []
    token_kls: List[torch.Tensor] = []
    for start in range(0, policy_logits.shape[0], token_batch_size):
        end = min(start + token_batch_size, policy_logits.shape[0])
        policy_block = policy_logits[start:end].to(device=kl_device, dtype=torch.float32)
        reference_block = reference_logits[start:end].to(device=kl_device, dtype=torch.float32)
        token_block = observed_token_ids[start:end].to(kl_device)

        policy_log_probs = torch.log_softmax(policy_block, dim=-1)
        reference_log_probs = torch.log_softmax(reference_block, dim=-1)
        policy_probs = policy_log_probs.exp()
        kl = (policy_probs * (policy_log_probs - reference_log_probs)).sum(dim=-1)
        indices = token_block.unsqueeze(1)

        policy_observed.append(policy_log_probs.gather(1, indices).squeeze(1).cpu())
        reference_observed.append(reference_log_probs.gather(1, indices).squeeze(1).cpu())
        token_kls.append(kl.cpu())

        del policy_block, reference_block, policy_log_probs, reference_log_probs, policy_probs

    return (
        torch.cat(policy_observed),
        torch.cat(reference_observed),
        torch.cat(token_kls),
    )


def _validate_spans(spans: Sequence[Tuple[int, int]], num_tokens: int):
    cursor = 0
    for start, end in spans:
        if int(start) != cursor or int(end) <= int(start):
            raise ValueError(f"Non-contiguous or empty chunk span: {(start, end)} at {cursor}.")
        cursor = int(end)
    if cursor != num_tokens:
        raise ValueError(f"Chunk spans cover {cursor} tokens, expected {num_tokens}.")


def analyze_response_side(
    bundle: ModelBundle,
    tokenized: Dict[str, Any],
    side_key: str,
    chunk_spans: Sequence[Tuple[int, int]],
    beta: float,
    kl_device: torch.device,
    kl_token_batch_size: int,
    kl_negative_tolerance: float,
) -> Dict[str, Any]:
    labels = torch.tensor(tokenized[f"{side_key}_labels"], dtype=torch.long)
    response_mask = torch.tensor(tokenized[f"{side_key}_response_mask"], dtype=torch.bool)
    shifted_valid = response_mask[1:] & (labels[1:] != -100)
    prediction_positions = shifted_valid.nonzero(as_tuple=False).flatten()
    observed_token_ids = labels[1:][shifted_valid]
    if observed_token_ids.numel() == 0:
        raise ValueError(f"{side_key} response contains no valid next-token positions.")

    spans = [(int(start), int(end)) for start, end in chunk_spans]
    _validate_spans(spans, observed_token_ids.numel())

    input_ids = tokenized[f"{side_key}_input_ids"]
    attention_mask = tokenized[f"{side_key}_attention_mask"]
    policy_logits = bundle.selected_policy_logits(
        input_ids, attention_mask, prediction_positions
    )
    reference_logits = bundle.selected_reference_logits(
        input_ids, attention_mask, prediction_positions
    )
    policy_logps, reference_logps, token_kl = observed_logps_and_forward_kl(
        policy_logits,
        reference_logits,
        observed_token_ids,
        kl_device=kl_device,
        token_batch_size=kl_token_batch_size,
    )
    del policy_logits, reference_logits

    min_kl = float(token_kl.min().item())
    if min_kl < -abs(kl_negative_tolerance):
        raise ValueError(
            f"Full-vocabulary KL(policy || reference) is materially negative: {min_kl:.8g}."
        )
    token_kl = token_kl.clamp_min(0.0)

    chunk_logratios, sequence_logratio = compute_exact_chunk_log_ratios(
        policy_logps,
        reference_logps,
        spans,
        beta=beta,
        verify_telescope=True,
    )
    chunk_kls = torch.stack([token_kl[start:end].mean() for start, end in spans])
    chunk_texts = [
        bundle.tokenizer.decode(
            observed_token_ids[start:end].tolist(),
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        for start, end in spans
    ]
    return {
        "spans": spans,
        "chunk_logratios": chunk_logratios,
        "sequence_logratio": sequence_logratio,
        "chunk_kls": chunk_kls,
        "chunk_texts": chunk_texts,
        "min_raw_token_kl": min_kl,
    }


def _load_heldout_dataset(args: argparse.Namespace):
    from datasets import load_dataset

    if "train" in args.split.lower() and not args.allow_train_split:
        raise ValueError(
            f"Refusing primary reviewer analysis on split {args.split!r}. "
            "Use a held-out split, or pass --allow_train_split explicitly."
        )
    dataset_path = Path(args.dataset)
    if dataset_path.is_file():
        if dataset_path.suffix.lower() not in {".json", ".jsonl"}:
            raise ValueError("Local --dataset currently supports JSON or JSONL files only.")
        dataset = load_dataset("json", data_files={"heldout": str(dataset_path)}, split="heldout")
    else:
        dataset = load_dataset(args.dataset, split=args.split, revision=args.dataset_revision)
    if args.shuffle:
        dataset = dataset.shuffle(seed=args.seed)
    return dataset


def _example_messages(example: Dict[str, Any]) -> Tuple[List[dict], List[dict], List[dict]]:
    chosen = example.get("chosen")
    rejected = example.get("rejected")
    if not isinstance(chosen, list) or not isinstance(rejected, list):
        raise ValueError("Each row must contain list-valued 'chosen' and 'rejected' conversations.")
    if len(chosen) != 2 or len(rejected) != 2:
        raise ValueError("ARC-BPO currently expects exactly one user and one assistant turn.")
    if chosen[:-1] != rejected[:-1]:
        raise ValueError("Chosen and rejected conversations do not share the same prompt.")
    return chosen[:-1], [chosen[-1]], [rejected[-1]]


def collect_chunk_rows(
    args: argparse.Namespace,
    bundle: ModelBundle,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    # Import lazily so --help and the statistics-only unit tests do not require
    # the Hugging Face dataset stack to be installed.
    from preference_datasets import tokenize_batch_element

    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = lambda iterable, **_: iterable

    dataset = _load_heldout_dataset(args)
    kl_device = _resolve_kl_device(args.kl_device)
    rows: List[Dict[str, Any]] = []
    calibration_errors: List[float] = []
    min_raw_kls: List[float] = []
    filtered_examples = 0
    processed_examples = 0

    iterator = tqdm(enumerate(dataset), total=len(dataset), desc="Credit-drift analysis")
    for dataset_index, example in iterator:
        prompt, chosen, rejected = _example_messages(example)
        tokenized = tokenize_batch_element(
            prompt,
            chosen,
            rejected,
            bundle.tokenizer,
            max_length=args.max_length,
            max_tokens_per_chunk=args.max_tokens_per_chunk,
            min_tokens_per_chunk=args.min_tokens_per_chunk,
            chosen_score=example.get("score_chosen"),
            rejected_score=example.get("score_rejected"),
        )
        if tokenized is None:
            filtered_examples += 1
            continue

        winner = analyze_response_side(
            bundle,
            tokenized,
            "chosen",
            tokenized["chosen_chunk_spans"],
            args.beta,
            kl_device,
            args.kl_token_batch_size,
            args.kl_negative_tolerance,
        )
        loser = analyze_response_side(
            bundle,
            tokenized,
            "rejected",
            tokenized["rejected_chunk_spans"],
            args.beta,
            kl_device,
            args.kl_token_batch_size,
            args.kl_negative_tolerance,
        )

        tau_w, tau_l, pi_w, rho_l = build_credit_targets(
            winner["chunk_logratios"],
            loser["chunk_logratios"],
            allocation_mode=args.allocation_mode,
            delta0=args.delta0,
            temperature=args.temperature,
            kappa=args.kappa,
            chosen_dataset_proxy=tokenized.get("chosen_adv_proxy"),
            rejected_dataset_proxy=tokenized.get("rejected_adv_proxy"),
            fallback_to_uniform=args.fallback_to_uniform_shape,
        )
        calibration_error = abs(float((tau_w.sum() - tau_l.sum()).item()) - args.delta0)
        if calibration_error > 1e-5:
            raise ValueError(
                f"ARC-BPO calibration failed at dataset row {dataset_index}: "
                f"error={calibration_error:.8g}."
            )
        if not torch.allclose(pi_w.sum(), torch.tensor(1.0, dtype=pi_w.dtype), atol=1e-5):
            raise ValueError("Winner allocation does not sum to one.")
        if not torch.allclose(rho_l.sum(), torch.tensor(1.0, dtype=rho_l.dtype), atol=1e-5):
            raise ValueError("Loser allocation does not sum to one.")

        example_id = str(example.get("id", example.get("prompt_id", dataset_index)))
        example_index = processed_examples
        for side_name, analysis, taus, allocations in (
            ("winner", winner, tau_w, pi_w),
            ("loser", loser, tau_l, rho_l),
        ):
            num_chunks = len(analysis["spans"])
            for chunk_id, ((start, end), text, logratio, allocation, tau, kl) in enumerate(
                zip(
                    analysis["spans"],
                    analysis["chunk_texts"],
                    analysis["chunk_logratios"],
                    allocations,
                    taus,
                    analysis["chunk_kls"],
                )
            ):
                rows.append(
                    {
                        "example_id": example_id,
                        "example_index": example_index,
                        "dataset_index": dataset_index,
                        "side": side_name,
                        "chunk_id": chunk_id,
                        "start_token": start,
                        "end_token": end,
                        "num_tokens": end - start,
                        "chunk_text": text,
                        "chunk_logratio": float(logratio.item()),
                        "allocation_weight": float(allocation.item()),
                        "tau": float(tau.item()),
                        "credit_magnitude": abs(float(tau.item())),
                        "policy_reference_kl": float(kl.item()),
                        "response_num_chunks": num_chunks,
                        "credit_source": args.allocation_mode,
                    }
                )

        calibration_errors.append(calibration_error)
        min_raw_kls.extend([winner["min_raw_token_kl"], loser["min_raw_token_kl"]])
        processed_examples += 1
        if args.max_examples > 0 and processed_examples >= args.max_examples:
            break

    if not rows:
        raise RuntimeError("No valid held-out chunks were produced.")

    diagnostics = {
        "dataset_rows_seen": dataset_index + 1 if len(dataset) else 0,
        "processed_examples": processed_examples,
        "filtered_examples": filtered_examples,
        "num_chunks": len(rows),
        "mean_calibration_error": statistics.fmean(calibration_errors),
        "max_calibration_error": max(calibration_errors),
        "min_raw_token_kl": min(min_raw_kls),
    }
    return rows, diagnostics


def correlation_payload(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    analyses: Dict[str, Any] = {}
    for label, selected in (
        ("all", list(rows)),
        ("winner", [row for row in rows if row["side"] == "winner"]),
        ("loser", [row for row in rows if row["side"] == "loser"]),
    ):
        analyses[label] = {
            "credit_magnitude_vs_kl": spearman_result(selected, "credit_magnitude"),
            "allocation_weight_vs_kl": spearman_result(selected, "allocation_weight"),
            "chunk_length_vs_kl": spearman_result(selected, "num_tokens"),
        }
    primary = analyses["all"]["credit_magnitude_vs_kl"]
    return {
        "spearman_rho": primary["rho"],
        "p_value": primary["p_value"],
        "num_chunks": primary["num_chunks"],
        "analyses": analyses,
    }


def _plot_outputs(
    output_dir: Path,
    rows: Sequence[Dict[str, Any]],
    grouped: Sequence[Dict[str, Any]],
    correlations: Dict[str, Any],
    seed: int,
    scatter_max_points: int,
):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "Plot generation requires matplotlib. Install "
            "diversity_metrics/requirements_credit_drift.txt."
        ) from exc

    labels = [row["credit_group"] for row in grouped]
    means = [row["mean_policy_reference_kl"] for row in grouped]
    valid_indices = [idx for idx, mean in enumerate(means) if mean is not None]
    fig, axis = plt.subplots(figsize=(5.4, 3.8))
    for idx in valid_indices:
        mean = float(means[idx])
        lower = grouped[idx]["kl_ci_lower"]
        upper = grouped[idx]["kl_ci_upper"]
        yerr = None
        if lower is not None and upper is not None:
            yerr = [[max(0.0, mean - float(lower))], [max(0.0, float(upper) - mean)]]
        axis.bar(idx, mean, color="#4472C4", alpha=0.88)
        if yerr is not None:
            axis.errorbar(idx, mean, yerr=yerr, fmt="none", color="black", capsize=4)
    axis.set_xticks(range(len(labels)), labels)
    axis.set_xlabel("ARC-BPO credit group")
    axis.set_ylabel("Mean KL(policy || reference)")
    axis.set_title("Credit-conditioned policy drift")
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "credit_group_kl.pdf", bbox_inches="tight")
    plt.close(fig)

    points = list(rows)
    if scatter_max_points > 0 and len(points) > scatter_max_points:
        points = random.Random(seed).sample(points, scatter_max_points)
    x = [float(row["credit_magnitude"]) for row in points]
    y = [float(row["policy_reference_kl"]) for row in points]

    ordered = sorted(
        (float(row["credit_magnitude"]), float(row["policy_reference_kl"])) for row in rows
    )
    deciles: List[Tuple[float, float]] = []
    for bin_index in range(10):
        start = len(ordered) * bin_index // 10
        end = len(ordered) * (bin_index + 1) // 10
        current = ordered[start:end]
        if current:
            deciles.append(
                (
                    statistics.fmean(value[0] for value in current),
                    statistics.fmean(value[1] for value in current),
                )
            )

    fig, axis = plt.subplots(figsize=(5.4, 3.8))
    axis.scatter(x, y, s=8, alpha=0.12, color="#4472C4", edgecolors="none")
    if deciles:
        axis.plot(
            [point[0] for point in deciles],
            [point[1] for point in deciles],
            marker="o",
            linewidth=2,
            color="#C00000",
            label="Decile-binned mean",
        )
        axis.legend(frameon=False)
    rho = correlations.get("spearman_rho")
    rho_text = "undefined" if rho is None else f"{rho:.3f}"
    axis.set_title(f"Continuous credit vs policy drift (Spearman rho={rho_text})")
    axis.set_xlabel("Credit magnitude |tau|")
    axis.set_ylabel("Mean chunk KL(policy || reference)")
    axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "credit_vs_kl.pdf", bbox_inches="tight")
    plt.close(fig)


def _format_number(value: Any, precision: int = 6) -> str:
    if value is None:
        return "NA"
    return f"{float(value):.{precision}g}"


def _write_summary(
    path: Path,
    args: argparse.Namespace,
    grouped: Sequence[Dict[str, Any]],
    correlations: Dict[str, Any],
    diagnostics: Dict[str, Any],
    thresholds: Dict[str, Dict[str, float]],
):
    means = [row["mean_policy_reference_kl"] for row in grouped]
    monotonic = bool(
        all(value is not None for value in means)
        and float(means[0]) < float(means[1]) < float(means[2])
    )
    if monotonic:
        interpretation = (
            "The observed group means follow KL_low < KL_medium < KL_high, which is "
            "consistent with the proposed localized-update mechanism. This is associative "
            "evidence, not a causal or theoretical guarantee of diversity."
        )
    else:
        interpretation = (
            "The observed group means do not show a strict KL_low < KL_medium < KL_high "
            "ordering. The current run therefore does not support claiming the proposed "
            "credit-localized drift mechanism."
        )

    source_notes = {
        "logratio": (
            "Credit is a post-hoc allocation computed from detached checkpoint chunk "
            "log-ratios, as requested in the reviewer plan. It is not the historical "
            "training target unless the checkpoint used this exact allocation rule."
        ),
        "uniform": (
            "Credit reconstructs the public ARC-BPO launchers' uniform target. Variation "
            "in |tau| is then driven by the number of chunks in each response."
        ),
        "dataset_score": (
            "Credit reconstructs the repository's score-proxy allocation using held-out "
            "score_chosen/score_rejected fields, with the configured fallback behavior."
        ),
    }

    lines = [
        "# ARC-BPO Credit-Conditioned Policy Drift",
        "",
        f"- Policy: `{args.policy_model}`",
        f"- Reference: `{args.reference_model}`",
        f"- Dataset/split: `{args.dataset}` / `{args.split}`",
        f"- Credit source: `{args.allocation_mode}`",
        f"- Valid preference pairs: {diagnostics['processed_examples']}",
        f"- Chunks: {diagnostics['num_chunks']}",
        "",
        source_notes[args.allocation_mode],
        "",
        "## Primary grouped result",
        "",
        "| Credit group | # chunks | Mean abs(tau) | Mean KL(policy vs reference) | 95% cluster-bootstrap CI |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in grouped:
        ci = f"[{_format_number(row['kl_ci_lower'])}, {_format_number(row['kl_ci_upper'])}]"
        lines.append(
            f"| {row['credit_group']} | {row['num_chunks']} | "
            f"{_format_number(row['mean_credit_magnitude'])} | "
            f"{_format_number(row['mean_policy_reference_kl'])} | {ci} |"
        )
    lines.extend(
        [
            "",
            "## Continuous association",
            "",
            f"- Spearman rho: {_format_number(correlations.get('spearman_rho'))}",
            f"- p-value: {_format_number(correlations.get('p_value'))}",
            f"- N chunks: {correlations.get('num_chunks')}",
            "",
            "## Sanity checks",
            "",
            f"- Maximum calibration error: {_format_number(diagnostics['max_calibration_error'])}",
            f"- Mean calibration error: {_format_number(diagnostics['mean_calibration_error'])}",
            f"- Minimum raw token KL: {_format_number(diagnostics['min_raw_token_kl'])}",
            f"- Credit tertiles: {json.dumps(thresholds['credit_magnitude'])}",
            "",
            "## Interpretation",
            "",
            interpretation,
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    args = parse_args()
    if args.max_length <= 0:
        raise ValueError("max_length must be positive.")
    if args.max_examples < 0:
        raise ValueError("max_examples cannot be negative.")
    if args.beta <= 0 or args.delta0 <= 0 or args.temperature <= 0:
        raise ValueError("beta, delta0, and temperature must be positive.")
    if args.kappa < 0:
        raise ValueError("kappa cannot be negative.")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    bundle = load_model_bundle(args)
    rows, diagnostics = collect_chunk_rows(args, bundle)

    thresholds = {
        "credit_magnitude": assign_tertile_groups(
            rows, "credit_magnitude", "credit_group"
        ),
        "allocation_weight": assign_tertile_groups(
            rows, "allocation_weight", "allocation_group"
        ),
    }
    grouped = grouped_credit_summary(
        rows,
        bootstrap_iterations=args.bootstrap_iterations,
        confidence_level=args.confidence_level,
        seed=args.seed,
    )
    winner_rows = [row for row in rows if row["side"] == "winner"]
    loser_rows = [row for row in rows if row["side"] == "loser"]
    grouped_winner = grouped_credit_summary(
        winner_rows,
        bootstrap_iterations=args.bootstrap_iterations,
        confidence_level=args.confidence_level,
        seed=args.seed + 1,
    )
    grouped_loser = grouped_credit_summary(
        loser_rows,
        bootstrap_iterations=args.bootstrap_iterations,
        confidence_level=args.confidence_level,
        seed=args.seed + 2,
    )
    correlations = correlation_payload(rows)

    _write_csv(output_dir / "chunk_level_metrics.csv", rows, CHUNK_FIELDS)
    _write_csv(output_dir / "grouped_credit_drift.csv", grouped, GROUP_FIELDS)
    _write_csv(
        output_dir / "grouped_credit_drift_winner.csv", grouped_winner, GROUP_FIELDS
    )
    _write_csv(output_dir / "grouped_credit_drift_loser.csv", grouped_loser, GROUP_FIELDS)
    _write_json(output_dir / "correlation_results.json", correlations)

    config_payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "arguments": vars(args),
        "resolved_output_dir": str(output_dir),
        "resolved_adapter_path": bundle.adapter_path,
        "resolved_policy_base_model": bundle.policy_base_model,
        "shared_adapter_reference": bundle.shared_adapter_reference,
        "tertile_thresholds": thresholds,
        "diagnostics": diagnostics,
        "versions": {"torch": torch.__version__},
    }
    _write_json(output_dir / "config.json", config_payload)
    _plot_outputs(
        output_dir,
        rows,
        grouped,
        correlations,
        seed=args.seed,
        scatter_max_points=args.scatter_max_points,
    )
    _write_summary(
        output_dir / "summary.md",
        args,
        grouped,
        correlations,
        diagnostics,
        thresholds,
    )

    print(f"Analyzed {diagnostics['processed_examples']} preference pairs.")
    print(f"Wrote {len(rows)} chunk rows to {output_dir / 'chunk_level_metrics.csv'}")
    print(f"Reviewer summary: {output_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
