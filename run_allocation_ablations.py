#!/usr/bin/env python3
"""Audit and launch the one-seed Llama-3-8B allocation ablations.

The launcher is intentionally strict about the reviewer plan.  The current
repository implements uniform targets and advantage-shaped targets with the
SBA generator, but it does not implement a distinct pre-SBA ``advantage``
loss.  That requested row is therefore reported as blocked instead of being
silently aliased to another method.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parent
TRAIN_SCRIPT = REPO_ROOT / "script" / "train" / "arc_bpo_llama.sh"
DEFAULT_OUTPUT_ROOT = "outputs/ablations"
DEFAULT_UNIFORM_CHECKPOINT = (
    "ducthang1703/llama3-arc-bpo-uniform-lora-10k-bs64"
)
ALL_VARIANTS = (
    "uniform",
    "advantage",
    "advantage_sba_no_winsor",
)

ADVANTAGE_BLOCKER = (
    "The repository has no distinct pre-SBA/base-Bregman ARC-BPO loss for the "
    "reviewer row 'Advantage allocation'. arc_bpo_pair_loss always applies "
    "bregman_sba, while A_tbpo/BPO_SBA change the objective and chunk semantics. "
    "Per the plan, this row must not be invented or aliased."
)


@dataclass(frozen=True)
class Variant:
    name: str
    supported: bool
    env_patch: Mapping[str, str]
    reason: str = ""


VARIANTS: Mapping[str, Variant] = {
    "uniform": Variant(
        name="uniform",
        supported=True,
        env_patch={
            "USE_ADVANTAGE_SHAPE": "false",
            "FALLBACK_TO_UNIFORM_SHAPE": "false",
            # Winsorization is inapplicable when the shape is uniform. Pinning
            # it off makes the resolved intent explicit.
            "WINSORIZE_ADVANTAGES": "false",
        },
    ),
    "advantage": Variant(
        name="advantage",
        supported=False,
        env_patch={},
        reason=ADVANTAGE_BLOCKER,
    ),
    "advantage_sba_no_winsor": Variant(
        name="advantage_sba_no_winsor",
        supported=True,
        env_patch={
            "USE_ADVANTAGE_SHAPE": "true",
            # Missing/mismatched ArmoRM scores must fail instead of silently
            # turning this run into the uniform ablation.
            "FALLBACK_TO_UNIFORM_SHAPE": "false",
            "WINSORIZE_ADVANTAGES": "false",
        },
    ),
}

MANIFEST_FIELDS = (
    "variant",
    "seed",
    "status",
    "checkpoint",
    "run_dir",
    "config_path",
    "config_hash",
    "reason",
)


def parse_variants(raw: str) -> List[str]:
    variants = [item.strip() for item in raw.split(",") if item.strip()]
    if not variants:
        raise ValueError("At least one allocation variant is required.")
    unknown = [item for item in variants if item not in VARIANTS]
    if unknown:
        raise ValueError(f"Unknown allocation variants: {unknown}")
    if len(set(variants)) != len(variants):
        raise ValueError("Allocation variants must not be repeated.")
    return variants


def visible_gpu_count(gpu_ids: str) -> int:
    ids = [item.strip() for item in gpu_ids.split(",") if item.strip()]
    if not ids:
        raise ValueError("--gpu_ids must contain at least one GPU id.")
    return len(ids)


def canonical_hash(payload: Mapping[str, str]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def base_environment(
    seed: int,
    gpu_ids: str,
    grad_accum: int,
    output_root: Path,
    n_examples: int = 10000,
    global_batch_size: int = 64,
    base_revision: str = "",
    dataset_revision: str = "",
) -> Dict[str, str]:
    gpu_count = visible_gpu_count(gpu_ids)
    if grad_accum <= 0:
        raise ValueError("--grad_accum must be positive.")
    if n_examples <= 0 or global_batch_size <= 0:
        raise ValueError("--n_examples and --global_batch_size must be positive.")
    if global_batch_size % (grad_accum * gpu_count) != 0:
        raise ValueError(
            f"Global batch {global_batch_size} must be divisible by "
            "grad_accum * GPU count; "
            f"got {grad_accum} * {gpu_count}."
        )
    environment = {
        # Exact public Llama setting associated with the supplied uniform
        # checkpoint. Every new ablation starts from this SFT base, never from
        # the already preference-tuned uniform adapter.
        "MODEL_CONFIG": "llama_8b",
        "DATASETS_RAW": "princeton-nlp/llama3-ultrafeedback-armorm",
        "TRAIN_SPLIT": "train",
        "TEST_SPLIT": "test",
        "SEED": str(seed),
        "GPU_IDS": gpu_ids,
        "BATCH_SIZE": str(global_batch_size),
        "GRAD_ACCUM": str(grad_accum),
        "N_EXAMPLES": str(n_examples),
        "USE_LORA": "true",
        "LR": "5e-7",
        "WEIGHT_DECAY": "0.0",
        "MAX_GRAD_NORM": "10.0",
        "OPTIMIZER": "RMSprop",
        "SCHEDULER": "cosine",
        "WARMUP_RATIO": "0.05",
        "MAX_LENGTH": "2048",
        "TRAINER": "FSDPTrainer",
        "ACTIVATION_CHECKPOINTING": "true",
        "N_EVAL_EXAMPLES": "0",
        "DO_FIRST_EVAL": "false",
        "SAVE_CHECKPOINT": "true",
        "SAVE_EVERY_EXAMPLES": "5000",
        "BETA": "0.1",
        "DELTA_STAR": "2.5",
        "ARC_T": "2.0",
        "KAPPA": "2.0",
        "SBA_LAMBDA": "1.0",
        "SBA_SCALE": "4.0",
        "EXP_CLIP": "30.0",
        "MIN_TOKENS_PER_CHUNK": "4",
        "MAX_TOKENS_PER_CHUNK": "64",
        "OUTPUT_DIR": str((output_root / "checkpoints").resolve()),
        # Prevent an inherited shell variable from uploading the wrong run.
        "HF_REPO_ID": "",
    }
    if base_revision:
        environment["MODEL_REVISION"] = base_revision
    if dataset_revision:
        environment["DATASET_REVISION"] = dataset_revision
    return environment


def run_environment(
    base: Mapping[str, str],
    variant: Variant,
    seed: int,
    output_root: Path,
) -> Dict[str, str]:
    if not variant.supported:
        raise ValueError(variant.reason)
    env = dict(base)
    env.update(variant.env_patch)
    n_examples = int(env["N_EXAMPLES"])
    example_label = "10k" if n_examples == 10000 else f"{n_examples}examples"
    run_name = (
        f"llama3_lora_{example_label}_bs{env['BATCH_SIZE']}_"
        f"{variant.name}_seed{seed}"
    )
    run_dir = (output_root / "runs" / run_name).resolve()
    env.update(
        {
            "EXP_NAME": run_name,
            "RUN_DIR": str(run_dir),
            "LOG_DIR": str((output_root / "logs").resolve()),
            "TRAIN_LOG": str((output_root / "logs" / f"{run_name}.log").resolve()),
        }
    )
    return env


def scientific_environment(env: Mapping[str, str]) -> Dict[str, str]:
    operational = {"OUTPUT_DIR", "RUN_DIR", "LOG_DIR", "TRAIN_LOG", "EXP_NAME"}
    return {key: value for key, value in env.items() if key not in operational}


def config_diff(left: Mapping[str, str], right: Mapping[str, str]) -> List[str]:
    lines = []
    for key in sorted(set(left) | set(right)):
        if left.get(key) != right.get(key):
            lines.append(f"{key}: {left.get(key)!r} -> {right.get(key)!r}")
    return lines


def adapter_checkpoint_complete(path: Path) -> bool:
    if not (path / "adapter_config.json").is_file():
        return False
    weights = list(path.glob("adapter_model*.safetensors")) + list(
        path.glob("adapter_model*.bin")
    )
    return bool(weights) and all(weight.stat().st_size > 1024 for weight in weights)


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_manifest(path: Path, rows: Iterable[Mapping[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plan/audit the Llama-3-8B LoRA 10k, global-bs64 allocation "
            "ablations with exactly one matched seed."
        )
    )
    parser.add_argument(
        "--variants",
        default=",".join(ALL_VARIANTS),
        help="Comma-separated variants; the full reviewer set is the default.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="The one matched seed for every requested variant (default: 0).",
    )
    parser.add_argument("--gpu_ids", default="0,1,2,3")
    parser.add_argument(
        "--grad_accum",
        type=int,
        default=4,
        help="Default 4 matches the public bs64 four-GPU launch setting.",
    )
    parser.add_argument(
        "--n_examples",
        type=int,
        default=10000,
        help="Default 10000; a smaller value is intended only for smoke tests.",
    )
    parser.add_argument(
        "--global_batch_size",
        type=int,
        default=64,
        help="Global training batch size (default: 64).",
    )
    parser.add_argument("--output_root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--base_revision", default="")
    parser.add_argument("--dataset_revision", default="")
    parser.add_argument(
        "--uniform_checkpoint",
        default=DEFAULT_UNIFORM_CHECKPOINT,
        help="Existing uniform LoRA adapter used only as a result reference.",
    )
    parser.add_argument(
        "--reuse_uniform_checkpoint",
        action="store_true",
        help=(
            "Do not retrain uniform; record --uniform_checkpoint in the manifest. "
            "Its original resolved config still must be audited before publication."
        ),
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    variants = parse_variants(args.variants)
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    base = base_environment(
        args.seed,
        args.gpu_ids,
        args.grad_accum,
        output_root,
        n_examples=args.n_examples,
        global_batch_size=args.global_batch_size,
        base_revision=args.base_revision,
        dataset_revision=args.dataset_revision,
    )
    requested = [VARIANTS[name] for name in variants]
    blocked = [variant for variant in requested if not variant.supported]
    supported_envs: Dict[str, Dict[str, str]] = {
        variant.name: run_environment(base, variant, args.seed, output_root)
        for variant in requested
        if variant.supported
    }

    rows: List[Dict[str, str]] = []
    for variant in requested:
        if not variant.supported:
            rows.append(
                {
                    "variant": variant.name,
                    "seed": str(args.seed),
                    "status": "blocked_missing_loss_definition",
                    "checkpoint": "",
                    "run_dir": "",
                    "config_path": "",
                    "config_hash": "",
                    "reason": variant.reason,
                }
            )
            continue

        env = supported_envs[variant.name]
        run_dir = Path(env["RUN_DIR"])
        checkpoint = run_dir / "LATEST" / "adapter"
        status = "planned"
        checkpoint_value = str(checkpoint)
        reason = ""
        if variant.name == "uniform" and args.reuse_uniform_checkpoint:
            status = "external_checkpoint_unverified"
            checkpoint_value = args.uniform_checkpoint
            reason = (
                "Adapter metadata identifies the SFT base, but the HF adapter-only "
                "upload does not prove seed/data/batch/training/eval settings. Locate "
                "the original resolved config before treating it as comparable."
            )
        rows.append(
            {
                "variant": variant.name,
                "seed": str(args.seed),
                "status": status,
                "checkpoint": checkpoint_value,
                "run_dir": str(run_dir),
                "config_path": str(run_dir / "config.yaml"),
                "config_hash": (
                    ""
                    if status == "external_checkpoint_unverified"
                    else canonical_hash(scientific_environment(env))
                ),
                "reason": reason,
            }
        )

    manifest_path = output_root / "run_manifest.csv"
    write_manifest(manifest_path, rows)

    diffs = {}
    supported_names = list(supported_envs)
    for left_index, left_name in enumerate(supported_names):
        for right_name in supported_names[left_index + 1 :]:
            changes = config_diff(
                scientific_environment(supported_envs[left_name]),
                scientific_environment(supported_envs[right_name]),
            )
            diff_name = f"config_diff_{left_name}_vs_{right_name}.txt"
            diff_path = output_root / diff_name
            diff_path.write_text("\n".join(changes) + ("\n" if changes else ""), encoding="utf-8")
            diffs[f"{left_name}_vs_{right_name}"] = changes

    audit = {
        "one_seed": [args.seed],
        "model": "RLHFlow/LLaMA3-SFT-v2",
        "model_revision": args.base_revision or None,
        "initialization": "fresh LoRA on common SFT base; never the uniform adapter",
        "dataset": "princeton-nlp/llama3-ultrafeedback-armorm",
        "dataset_revision": args.dataset_revision or None,
        "n_examples": args.n_examples,
        "global_batch_size": args.global_batch_size,
        "gradient_accumulation_steps": args.grad_accum,
        "gpu_ids": args.gpu_ids,
        "uniform_checkpoint_reference": args.uniform_checkpoint,
        "requested_variants": variants,
        "blocked_variants": {variant.name: variant.reason for variant in blocked},
        "config_diffs": diffs,
        "config_diffs_scope": "expected launcher settings, not unverified external metadata",
        "requested_set_runnable": not blocked,
        "full_reviewer_study_runnable": all(
            VARIANTS[name].supported for name in ALL_VARIANTS
        ),
    }
    write_json(output_root / "config_audit.json", audit)

    print(f"Manifest: {manifest_path}")
    print(f"Config audit: {output_root / 'config_audit.json'}")
    if blocked:
        for variant in blocked:
            print(f"BLOCKED [{variant.name}]: {variant.reason}", file=sys.stderr)
        if args.execute:
            print("No training was started because the requested set is not scientifically defined.")
            return 2

    if not args.execute:
        print("Dry run only. Add --execute after inspecting the audit.")
        return 0

    for row in rows:
        if row["status"] == "external_checkpoint_unverified":
            print(f"Skipping uniform training; recorded {row['checkpoint']} as unverified reference.")
            continue
        if row["status"] != "planned":
            continue
        checkpoint = Path(row["checkpoint"])
        if checkpoint.is_dir() and not args.force:
            if not adapter_checkpoint_complete(checkpoint):
                row["status"] = "incomplete_checkpoint"
                write_manifest(manifest_path, rows)
                raise RuntimeError(
                    f"Existing adapter is incomplete: {checkpoint}. Inspect it and "
                    "pass --force only when explicit retraining is intended."
                )
            row["status"] = "checkpoint_exists"
            write_manifest(manifest_path, rows)
            print(f"Skipping existing checkpoint: {checkpoint}")
            continue

        variant = row["variant"]
        run_dir = Path(row["run_dir"])
        run_dir.mkdir(parents=True, exist_ok=True)
        environment = os.environ.copy()
        environment.update(supported_envs[variant])
        # CUDA_VISIBLE_DEVICES takes precedence inside the public launcher, so
        # pin it here instead of inheriting a conflicting parent value.
        environment["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
        row["status"] = "running"
        write_manifest(manifest_path, rows)
        print(f"Training {variant} with seed={args.seed}")
        try:
            subprocess.run(
                ["bash", str(TRAIN_SCRIPT)],
                cwd=REPO_ROOT,
                env=environment,
                check=True,
            )
        except subprocess.CalledProcessError:
            row["status"] = "failed"
            write_manifest(manifest_path, rows)
            raise
        if not checkpoint.is_dir():
            row["status"] = "missing_checkpoint"
            write_manifest(manifest_path, rows)
            raise RuntimeError(f"Training completed without LoRA adapter: {checkpoint}")
        row["status"] = "trained"
        write_manifest(manifest_path, rows)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
