from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple


TASKS = ("hellaswag", "arc", "mmlu", "truthfulqa", "winogrande", "gsm8k")
LM_EVAL_TASKS = {
    "hellaswag": {"task": "hellaswag", "fewshot": 10, "metrics": ("acc_norm,none", "acc,none")},
    "arc": {"task": "arc_challenge", "fewshot": 25, "metrics": ("acc_norm,none", "acc,none")},
    "mmlu": {"task": "mmlu", "fewshot": 5, "metrics": ("acc,none",)},
    "truthfulqa": {"task": "truthfulqa_mc2", "fewshot": 0, "metrics": ("acc,none",)},
    "winogrande": {"task": "winogrande", "fewshot": 5, "metrics": ("acc,none",)},
    "gsm8k": {
        "task": "gsm8k",
        "fewshot": 5,
        "metrics": ("exact_match,strict-match", "exact_match,flexible-extract", "exact_match,none"),
    },
}

IDENTITY_PATHS = {
    "exp_name",
    "local_run_dir",
    "output_dir",
    "fsdp_port",
    "sensitivity",
}
NOISE_PATHS = {"label_noise_rate", "label_noise_seed", "label_noise_indices_path"}


@dataclass(frozen=True)
class RunSpec:
    sweep: str
    parameter: str
    value_label: str
    value: Optional[float]
    seed: int
    noise_rate: float
    order: int

    @property
    def noise_label(self) -> str:
        return "clean" if self.noise_rate == 0 else f"noise{int(round(100 * self.noise_rate))}"

    @property
    def run_name(self) -> str:
        value = re.sub(r"[^A-Za-z0-9._-]+", "_", self.value_label)
        return f"sens_{self.sweep}_{value}_{self.noise_label}_seed{self.seed}"


def load_resolved_config(path: str):
    from omegaconf import OmegaConf

    config = OmegaConf.load(path)
    OmegaConf.resolve(config)
    missing = OmegaConf.missing_keys(config)
    if missing:
        raise ValueError(f"Base config has unresolved mandatory keys: {sorted(missing)}")
    if "model" not in config or "loss" not in config:
        raise ValueError(
            "--base_config must be a resolved config.yaml saved by an actual ARC-BPO run, "
            "not the uncomposed config/config.yaml defaults file."
        )
    return config


def build_llama3_10k_bs64_base(
    repository_root: Path,
    output_root: Path,
    seed: int = 0,
    gradient_accumulation_steps: int = 4,
):
    """Build the exact advantage-enabled Llama-3 sensitivity baseline.

    This preset is intentionally explicit so a GPU launcher does not need
    to train a throwaway default run merely to obtain a resolved Hydra config.
    The generated config is still archived and audited like a config captured
    from a main run.
    """
    from omegaconf import OmegaConf

    repository_root = Path(repository_root).resolve()
    output_root = Path(output_root).resolve()
    gradient_accumulation_steps = int(gradient_accumulation_steps)
    if gradient_accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be positive.")
    config = OmegaConf.load(repository_root / "config" / "config.yaml")
    config.pop("defaults", None)
    config.model = OmegaConf.load(repository_root / "config" / "model" / "llama_8b.yaml")
    config.loss = OmegaConf.load(repository_root / "config" / "loss" / "arc_bpo.yaml")

    config.seed = int(seed)
    config.exp_name = (
        "arc-bpo-sensitivity-default-llama3-10k-bs64-"
        f"ga{gradient_accumulation_steps}"
    )
    config.output_dir = str(output_root)
    config.local_run_dir = str(output_root / "base-config-only")
    config.fsdp_port = None

    config.datasets = "princeton-nlp/llama3-ultrafeedback-armorm"
    config.dataset_train_split = "train"
    config.dataset_test_split = "test"
    config.batch_size = 64
    config.gradient_accumulation_steps = gradient_accumulation_steps
    config.n_examples = 10000
    config.n_epochs = None
    config.skip_examples = 0
    config.n_eval_examples = 0
    config.do_first_eval = False
    config.label_noise_rate = 0.0
    config.label_noise_seed = 0
    config.label_noise_indices_path = None

    config.trainer = "FSDPTrainer"
    config.lr = 5e-7
    config.weight_decay = 0.0
    config.optimizer = "RMSprop"
    config.scheduler = "cosine"
    config.warmup_ratio = 0.05
    config.max_length = 2048
    config.activation_checkpointing = True
    config.save_checkpoint = False
    config.save_every_examples = 10000
    config.wandb.enabled = False

    config.model.name_or_path = "RLHFlow/LLaMA3-SFT-v2"
    config.model.use_lora = True
    config.model.adapter_path = None

    config.loss.name = "arc_bpo"
    config.loss.T = 2.0
    config.loss.delta_star = 2.0
    config.loss.kappa = 2.0
    config.loss.sba_lambda = 1.0
    config.loss.use_advantage_shape = True
    config.loss.fallback_to_uniform_shape = False
    config.loss.winsorize_advantages = True
    config.loss.beta = 0.1
    config.loss.min_tokens_per_chunk = 4
    config.loss.max_tokens_per_chunk = 64

    OmegaConf.resolve(config)
    return config


def config_to_plain(config) -> Dict[str, Any]:
    from omegaconf import OmegaConf

    return OmegaConf.to_container(config, resolve=True, enum_to_str=True)


def canonical_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def flatten_config(payload: Mapping[str, Any], prefix: str = "") -> Dict[str, Any]:
    flattened: Dict[str, Any] = {}
    for key, value in payload.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            flattened.update(flatten_config(value, path))
        else:
            flattened[path] = value
    return flattened


def config_diff(base: Mapping[str, Any], candidate: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    before = flatten_config(base)
    after = flatten_config(candidate)
    output: Dict[str, Dict[str, Any]] = {}
    for key in sorted(set(before) | set(after)):
        if before.get(key) != after.get(key):
            output[key] = {"base": before.get(key), "run": after.get(key)}
    return output


def set_config_path(config, path: str, value: Any):
    node = config
    parts = path.split(".")
    for part in parts[:-1]:
        node = node[part]
    node[parts[-1]] = value


def get_config_path(config, path: str, default: Any = None) -> Any:
    node = config
    for part in path.split("."):
        try:
            node = node[part]
        except (KeyError, TypeError):
            return default
    return node


def normalize_base_config(config):
    """Add explicit fields whose historical behavior was implicit in code."""
    from omegaconf import OmegaConf

    normalized = OmegaConf.create(config_to_plain(config))
    if "winsorize_advantages" not in normalized.loss:
        # Before the explicit flag existed, advantage shaping always winsorized.
        normalized.loss.winsorize_advantages = True
    if "label_noise_indices_path" not in normalized:
        normalized.label_noise_indices_path = None
    return normalized


def validate_sensitivity_base(config):
    if str(config.loss.name) != "arc_bpo":
        raise ValueError(f"Sensitivity base loss must be arc_bpo, got {config.loss.name!r}.")
    if not bool(getattr(config.loss, "use_advantage_shape", False)):
        raise ValueError(
            "The supplied main-run config uses uniform allocation "
            "(loss.use_advantage_shape=false). T and kappa are inactive, so this would not "
            "be a valid ARC-BPO sensitivity study. Supply the resolved advantage-enabled "
            "main config used for the paper."
        )
    if not bool(getattr(config.loss, "winsorize_advantages", True)):
        raise ValueError(
            "The fixed sensitivity default requires loss.winsorize_advantages=true."
        )
    if float(getattr(config, "label_noise_rate", 0.0)) != 0.0:
        raise ValueError(
            "The final main-run base must be clean (label_noise_rate=0); noise is added "
            "only to the controlled kappa/lambda conditions."
        )
    required = (
        "loss.T",
        "loss.kappa",
        "loss.delta_star",
        "loss.sba_lambda",
        "loss.beta",
        "loss.min_tokens_per_chunk",
        "loss.max_tokens_per_chunk",
        "model.name_or_path",
        "datasets",
        "dataset_train_split",
        "dataset_test_split",
        "lr",
        "batch_size",
        "gradient_accumulation_steps",
    )
    missing = [path for path in required if get_config_path(config, path) is None]
    if missing:
        raise ValueError(f"Resolved main config is missing sensitivity controls: {missing}")
    if float(config.loss.T) <= 0 or float(config.loss.kappa) < 0:
        raise ValueError("Main allocation temperature must be positive and kappa non-negative.")
    if float(config.loss.sba_lambda) < 0:
        raise ValueError("Main SBA lambda must be non-negative.")

    required_defaults = {
        "loss.T": 2.0,
        "loss.delta_star": 2.0,
        "loss.kappa": 2.0,
        "loss.sba_lambda": 1.0,
    }
    mismatched_defaults = {
        path: {"expected": expected, "actual": float(get_config_path(config, path))}
        for path, expected in required_defaults.items()
        if not math.isclose(
            float(get_config_path(config, path)), expected, rel_tol=0, abs_tol=1e-12
        )
    }
    if mismatched_defaults:
        raise ValueError(
            "Sensitivity base does not match the fixed defaults in the current spec: "
            f"{mismatched_defaults}"
        )


def _append_default(grid: Sequence[float], default: float) -> List[float]:
    values = list(grid)
    if not any(math.isclose(float(value), float(default), rel_tol=0, abs_tol=1e-12) for value in values):
        values.append(float(default))
    return values


def build_run_specs(
    base_config,
    sweeps: Sequence[str],
    seeds: Sequence[int],
    noise_rate: float,
    include_default_points: bool = True,
) -> List[RunSpec]:
    supported = {"T", "kappa", "delta0", "lambda"}
    unknown = set(sweeps) - supported
    if unknown:
        raise ValueError(f"Unknown sensitivity sweeps: {sorted(unknown)}")
    if not seeds:
        raise ValueError("At least one matched seed is required.")

    grids: Dict[str, List[Tuple[str, Optional[float]]]] = {
        "T": [(format(value, "g"), value) for value in (4.0, 2.0, 1.0, 0.5)],
        "kappa": [(format(value, "g"), value) for value in (3.0, 2.0, 1.5, 1.0)],
        "delta0": [(format(value, "g"), value) for value in (0.5, 1.0, 2.0, 4.0)],
        "lambda": [(format(value, "g"), value) for value in (0.5, 1.0, 2.0)],
    }
    defaults = {
        "T": float(base_config.loss.T),
        "kappa": float(base_config.loss.kappa),
        "delta0": float(base_config.loss.delta_star),
        "lambda": float(base_config.loss.sba_lambda),
    }

    specs: List[RunSpec] = []
    order = 0
    for sweep in (name for name in ("T", "kappa", "delta0", "lambda") if name in sweeps):
        noise_conditions = (0.0, noise_rate) if sweep == "kappa" else (0.0,)
        for current_noise in noise_conditions:
            for value_label, value in grids[sweep]:
                if (
                    not include_default_points
                    and value is not None
                    and math.isclose(value, defaults[sweep], rel_tol=0, abs_tol=1e-12)
                ):
                    continue
                for seed in seeds:
                    specs.append(
                        RunSpec(
                            sweep=sweep,
                            parameter=sweep,
                            value_label=value_label,
                            value=value,
                            seed=int(seed),
                            noise_rate=float(current_noise),
                            order=order,
                        )
                    )
                    order += 1
    return specs


def expected_diff_paths(spec: RunSpec) -> set[str]:
    expected = set(IDENTITY_PATHS) | set(NOISE_PATHS) | {"seed"}
    if spec.sweep == "T":
        expected.add("loss.T")
    elif spec.sweep == "kappa":
        expected.add("loss.kappa")
    elif spec.sweep == "delta0":
        expected.add("loss.delta_star")
    elif spec.sweep == "lambda":
        expected.add("loss.sba_lambda")
    return expected


def patch_run_config(
    base_config,
    spec: RunSpec,
    output_root: Path,
    noise_seed: int,
    noise_manifest: Path,
):
    from omegaconf import OmegaConf

    config = OmegaConf.create(config_to_plain(base_config))
    run_dir = output_root / spec.sweep / spec.value_label / spec.noise_label / f"seed{spec.seed}"
    config.seed = spec.seed
    config.exp_name = spec.run_name
    config.output_dir = str(output_root)
    config.local_run_dir = str(run_dir)
    config.fsdp_port = None
    config.label_noise_rate = spec.noise_rate
    config.label_noise_seed = (
        int(noise_seed)
        if spec.noise_rate > 0
        else int(getattr(base_config, "label_noise_seed", 0))
    )
    config.label_noise_indices_path = str(noise_manifest) if spec.noise_rate > 0 else None

    if spec.sweep == "T":
        config.loss.use_advantage_shape = True
        config.loss.T = float(spec.value)
    elif spec.sweep == "kappa":
        config.loss.use_advantage_shape = True
        config.loss.winsorize_advantages = True
        config.loss.kappa = float(spec.value)
    elif spec.sweep == "delta0":
        config.loss.delta_star = float(spec.value)
    elif spec.sweep == "lambda":
        config.loss.sba_lambda = float(spec.value)
    else:
        raise ValueError(spec.sweep)

    config.sensitivity = {
        "sweep": spec.sweep,
        "parameter": spec.parameter,
        "value": spec.value_label,
        "numeric_value": spec.value,
        "noise_rate": spec.noise_rate,
        "seed": spec.seed,
        "order": spec.order,
    }
    return config


def audit_run_config(base_config, run_config, spec: RunSpec) -> Dict[str, Any]:
    base_plain = config_to_plain(base_config)
    run_plain = config_to_plain(run_config)
    differences = config_diff(base_plain, run_plain)
    expected = expected_diff_paths(spec)
    unexpected = sorted(path for path in differences if path not in expected and not path.startswith("sensitivity."))
    audit = {
        "run_name": spec.run_name,
        "sweep": spec.sweep,
        "value": spec.value_label,
        "seed": spec.seed,
        "noise_rate": spec.noise_rate,
        "expected_paths": sorted(expected),
        "differences": differences,
        "unexpected_paths": unexpected,
        "passed": not unexpected,
    }
    if unexpected:
        raise ValueError(
            f"Config audit failed for {spec.run_name}; unexpected changes: {unexpected}"
        )
    return audit


def scientific_payload(config) -> Dict[str, Any]:
    payload = config_to_plain(config)
    for path in IDENTITY_PATHS:
        payload.pop(path, None)
    return payload


def write_json(path: Path, payload: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_manifest(path: Path) -> List[Dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Sensitivity manifest not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def parse_seeds(value: str) -> List[int]:
    seeds = [int(item.strip()) for item in value.split(",") if item.strip()]
    if len(set(seeds)) != len(seeds):
        raise ValueError("Seed list contains duplicates.")
    return seeds


def parse_sweeps(value: str) -> List[str]:
    aliases = {"t": "T", "T": "T", "kappa": "kappa", "delta0": "delta0", "lambda": "lambda"}
    sweeps = []
    for item in value.split(","):
        item = item.strip()
        if item:
            if item not in aliases:
                raise ValueError(f"Unknown sweep {item!r}.")
            sweeps.append(aliases[item])
    return sweeps
