#!/usr/bin/env python3
"""Generate, audit, and optionally execute ARC-BPO sensitivity runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List

from sensitivity.common import (
    audit_run_config,
    build_llama3_10k_bs64_base,
    build_run_specs,
    canonical_hash,
    config_to_plain,
    load_resolved_config,
    normalize_base_config,
    parse_seeds,
    parse_sweeps,
    patch_run_config,
    scientific_payload,
    validate_sensitivity_base,
    write_csv,
    write_json,
)


MANIFEST_FIELDS = (
    "run_name",
    "sweep",
    "parameter",
    "value",
    "numeric_value",
    "seed",
    "noise_rate",
    "config_path",
    "run_dir",
    "checkpoint",
    "config_hash",
    "scientific_hash",
    "base_config_hash",
    "status",
    "hf_repo_id",
    "hf_path",
    "hf_url",
    "hf_status",
)


def _environment_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"{name} must be true or false, got {value!r}.")


def hf_checkpoint_path(run_name: str) -> str:
    """Return a stable, human-readable path for one run inside an HF repo."""
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "-", run_name).strip("-.")
    if not safe_name:
        raise ValueError("run_name must contain at least one safe character.")
    return f"checkpoints/{safe_name}"


def select_run_range(specs, *, start_run: int, max_runs: int):
    """Select a 1-based contiguous suffix/range while retaining total grid size."""
    total_runs = len(specs)
    if start_run < 1:
        raise ValueError("start_run must be a positive 1-based index.")
    if start_run > total_runs:
        raise ValueError(
            f"start_run={start_run} is out of range; the grid contains "
            f"{total_runs} runs."
        )
    selected = list(specs[start_run - 1 :])
    if max_runs:
        selected = selected[:max_runs]
    return selected, total_runs


def execution_preflight(
    config,
    *,
    visible_gpus: int,
    gpu_names: List[str],
    expected_gpus: int = 0,
    expected_gpu_name: str = "",
) -> Dict[str, Any]:
    """Validate the FSDP/GPU/batch contract before starting an expensive sweep."""
    if visible_gpus < 1:
        raise RuntimeError("No visible CUDA GPUs were found.")
    if len(gpu_names) != visible_gpus:
        raise ValueError(
            f"Received {len(gpu_names)} GPU names for {visible_gpus} visible GPUs."
        )
    if expected_gpus and visible_gpus != expected_gpus:
        raise RuntimeError(
            f"Expected {expected_gpus} visible CUDA GPUs, found {visible_gpus}."
        )
    expected_name = expected_gpu_name.strip().lower()
    if expected_name:
        mismatched = [name for name in gpu_names if expected_name not in name.lower()]
        if mismatched:
            raise RuntimeError(
                f"Expected every GPU name to contain {expected_gpu_name!r}; "
                f"mismatched devices: {mismatched}."
            )

    batch_size = int(config.batch_size)
    grad_accum = int(config.gradient_accumulation_steps)
    if batch_size <= 0 or grad_accum <= 0:
        raise ValueError("batch_size and gradient_accumulation_steps must be positive.")
    divisor = grad_accum * visible_gpus
    if batch_size % divisor:
        raise ValueError(
            f"Global batch_size={batch_size} must be divisible by "
            f"gradient_accumulation_steps*visible_gpus={divisor}."
        )

    configured_examples = getattr(config, "n_examples", None)
    optimizer_steps = None
    full_batch_examples = None
    if configured_examples is not None:
        configured_examples = int(configured_examples)
        if configured_examples <= 0:
            raise ValueError("n_examples must be positive when it is configured.")
        optimizer_steps = math.ceil(configured_examples / batch_size)
        full_batch_examples = optimizer_steps * batch_size

    return {
        "visible_gpus": visible_gpus,
        "gpu_names": list(gpu_names),
        "global_batch_size": batch_size,
        "gradient_accumulation_steps": grad_accum,
        "per_gpu_microbatch": batch_size // divisor,
        "configured_examples": configured_examples,
        "optimizer_steps": optimizer_steps,
        "full_batch_examples": full_batch_examples,
    }


def checkpoint_complete(checkpoint: Path, use_lora: bool) -> bool:
    checkpoint = Path(checkpoint)
    if not checkpoint.is_dir():
        return False
    if use_lora:
        payload = checkpoint / "adapter"
        config_path = payload / "adapter_config.json"
        weights = list(payload.glob("adapter_model*.safetensors")) + list(
            payload.glob("adapter_model*.bin")
        )
    else:
        payload = checkpoint
        config_path = payload / "config.json"
        weights = list(payload.glob("model*.safetensors")) + list(
            payload.glob("pytorch_model*.bin")
        )
    return config_path.is_file() and bool(weights) and all(
        path.stat().st_size > 1024 for path in weights
    )


def upload_checkpoint_to_hf(
    *,
    api,
    repo_id: str,
    row: Dict[str, Any],
    checkpoint: Path,
    adapter_only: bool,
) -> tuple[str, str]:
    """Upload one checkpoint plus its audited configuration to Hugging Face."""
    checkpoint = Path(checkpoint)
    config_path = Path(str(row["config_path"]))
    config_diff_path = config_path.parent / "config_diff.json"
    destination = hf_checkpoint_path(str(row["run_name"]))

    if not checkpoint.is_dir():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {checkpoint}")
    if adapter_only:
        adapter = checkpoint / "adapter"
        weights = list(adapter.glob("adapter_model*.safetensors")) + list(
            adapter.glob("adapter_model*.bin")
        )
        if (
            not (adapter / "adapter_config.json").is_file()
            or not weights
            or any(path.stat().st_size <= 1024 for path in weights)
        ):
            raise RuntimeError(f"Complete LoRA adapter not found at {adapter}")

    with tempfile.TemporaryDirectory(prefix="arc-bpo-hf-upload-") as temporary:
        staging = Path(temporary)
        if adapter_only:
            shutil.copytree(checkpoint / "adapter", staging / "adapter")
        else:
            shutil.copytree(checkpoint, staging / "checkpoint")
        shutil.copy2(config_path, staging / "resolved_config.yaml")
        if config_diff_path.is_file():
            shutil.copy2(config_diff_path, staging / "config_diff.json")
        metadata = {
            "run_name": row["run_name"],
            "sweep": row["sweep"],
            "parameter": row["parameter"],
            "value": row["value"],
            "seed": row["seed"],
            "noise_rate": row["noise_rate"],
            "checkpoint_layout": "adapter" if adapter_only else "checkpoint",
            "scientific_hash": row["scientific_hash"],
        }
        (staging / "checkpoint_metadata.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        api.upload_folder(
            repo_id=repo_id,
            repo_type="model",
            folder_path=str(staging),
            path_in_repo=destination,
            commit_message=f"Upload {row['run_name']}",
        )

    url = f"https://huggingface.co/{repo_id}/tree/main/{destination}"
    return destination, url


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create one-factor-at-a-time ARC-BPO sensitivity configs from an exact "
            "resolved main-run config. The default is audit-only; pass --execute to train."
        )
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--base_config",
        help="Resolved config.yaml saved by the advantage-enabled ARC-BPO main run.",
    )
    source.add_argument(
        "--preset",
        choices=("llama3-10k-bs64",),
        help="Build the exact advantage-enabled Llama-3/10k/global-bs64 baseline.",
    )
    parser.add_argument("--output_root", default="outputs/sensitivity")
    parser.add_argument("--sweeps", default="T,kappa,delta0,lambda")
    parser.add_argument("--seeds", default="42", help="Comma-separated matched seeds.")
    parser.add_argument("--noise_rate", type=float, default=0.20)
    parser.add_argument("--noise_seed", type=int, default=2026)
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=0,
        help=(
            "Override gradient accumulation for a built-in preset; 0 keeps the "
            "preset default (4). This is rejected with --base_config."
        ),
    )
    parser.add_argument(
        "--max_runs",
        type=int,
        default=0,
        help="Limit generated runs for a launcher smoke test; 0 generates the full grid.",
    )
    parser.add_argument(
        "--start_run",
        type=int,
        default=1,
        help="Start at this 1-based position in the generated grid (default: 1).",
    )
    parser.add_argument(
        "--exclude_default_points",
        action="store_true",
        help=(
            "Generate only the 14 missing runs from the current spec. The clean and "
            "20%%-noise default rows must then be supplied from the fixed published anchors."
        ),
    )
    parser.add_argument("--execute", action="store_true", help="Run training sequentially.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Retrain even when the run's LATEST checkpoint already exists.",
    )
    parser.add_argument(
        "--expected_gpus",
        type=int,
        default=0,
        help="When executing, require exactly this many visible CUDA GPUs; 0 disables the check.",
    )
    parser.add_argument(
        "--expected_gpu_name",
        default="",
        help="When executing, require every CUDA device name to contain this text.",
    )
    parser.add_argument(
        "--hf_repo_id",
        default=os.environ.get("HF_REPO_ID", ""),
        help="Optional model repo receiving every checkpoint (or set HF_REPO_ID).",
    )
    parser.add_argument(
        "--hf_private",
        action=argparse.BooleanOptionalAction,
        default=_environment_bool("HF_PRIVATE", True),
        help="Create/use a private HF repo (or set HF_PRIVATE).",
    )
    parser.add_argument(
        "--hf_upload_adapter_only",
        action=argparse.BooleanOptionalAction,
        default=_environment_bool("HF_UPLOAD_ADAPTER_ONLY", True),
        help="Upload only LoRA weights plus the audited config.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if not 0.0 < args.noise_rate < 1.0:
        raise ValueError("noise_rate must be strictly between zero and one.")
    if args.max_runs < 0:
        raise ValueError("max_runs cannot be negative.")
    if args.start_run < 1:
        raise ValueError("start_run must be a positive 1-based index.")
    if args.expected_gpus < 0:
        raise ValueError("expected_gpus cannot be negative.")
    if args.gradient_accumulation_steps < 0:
        raise ValueError("gradient_accumulation_steps cannot be negative.")
    if args.base_config and args.gradient_accumulation_steps:
        raise ValueError(
            "--gradient_accumulation_steps can only be used with --preset; "
            "a supplied --base_config must remain unchanged."
        )

    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    if args.preset == "llama3-10k-bs64":
        preset_grad_accum = args.gradient_accumulation_steps or 4
        base = build_llama3_10k_bs64_base(
            Path(__file__).resolve().parent,
            output_root,
            seed=parse_seeds(args.seeds)[0],
            gradient_accumulation_steps=preset_grad_accum,
        )
        base_source = f"preset:llama3-10k-bs64:grad_accum={preset_grad_accum}"
    else:
        base = load_resolved_config(args.base_config)
        base_source = str(Path(args.base_config).resolve())
    base = normalize_base_config(base)
    validate_sensitivity_base(base)
    base_plain = config_to_plain(base)
    base_config_hash = canonical_hash(scientific_payload(base))

    write_json(output_root / "default_config.json", base_plain)
    from omegaconf import OmegaConf

    OmegaConf.save(base, output_root / "default_config.yaml")

    sweeps = parse_sweeps(args.sweeps)
    seeds = parse_seeds(args.seeds)
    specs = build_run_specs(
        base,
        sweeps,
        seeds,
        args.noise_rate,
        include_default_points=not args.exclude_default_points,
    )
    first_run_index = args.start_run
    specs, total_runs = select_run_range(
        specs,
        start_run=first_run_index,
        max_runs=args.max_runs,
    )

    if args.execute:
        import torch

        visible_gpus = torch.cuda.device_count()
        gpu_names = [torch.cuda.get_device_name(index) for index in range(visible_gpus)]
        preflight = execution_preflight(
            base,
            visible_gpus=visible_gpus,
            gpu_names=gpu_names,
            expected_gpus=args.expected_gpus,
            expected_gpu_name=args.expected_gpu_name,
        )
        print(f"[preflight] GPUs: {preflight['gpu_names']}")
        print(
            "[preflight] global_batch={global_batch_size} grad_accum="
            "{gradient_accumulation_steps} per_gpu_microbatch={per_gpu_microbatch}".format(
                **preflight
            )
        )
        if preflight["configured_examples"] is not None:
            print(
                "[preflight] configured_examples={configured_examples} optimizer_steps="
                "{optimizer_steps} full_batch_examples={full_batch_examples}".format(
                    **preflight
                )
            )

    hf_api = None
    if args.execute and args.hf_repo_id:
        if args.hf_upload_adapter_only and not bool(base.model.use_lora):
            raise ValueError("Adapter-only HF upload requires model.use_lora=true.")
        from huggingface_hub import HfApi

        hf_api = HfApi(token=os.environ.get("HF_TOKEN"))
        hf_api.create_repo(
            repo_id=args.hf_repo_id,
            repo_type="model",
            private=args.hf_private,
            exist_ok=True,
        )

    noise_manifest = output_root / f"noise{int(round(100 * args.noise_rate))}_indices.json"
    manifest_rows: List[Dict[str, Any]] = []
    audits: List[Dict[str, Any]] = []
    manifest_path = output_root / "run_manifest.csv"

    for index, spec in enumerate(specs, start=first_run_index):
        run_config = patch_run_config(
            base,
            spec,
            output_root=output_root,
            noise_seed=args.noise_seed,
            noise_manifest=noise_manifest,
        )
        audit = audit_run_config(base, run_config, spec)
        run_dir = Path(str(run_config.local_run_dir))
        run_dir.mkdir(parents=True, exist_ok=True)
        config_path = run_dir / "resolved_config.yaml"
        OmegaConf.save(run_config, config_path)
        write_json(run_dir / "config_diff.json", audit)
        audits.append(audit)

        checkpoint = run_dir / "LATEST"
        status = "planned"
        row = {
            "run_name": spec.run_name,
            "sweep": spec.sweep,
            "parameter": spec.parameter,
            "value": spec.value_label,
            "numeric_value": "" if spec.value is None else spec.value,
            "seed": spec.seed,
            "noise_rate": spec.noise_rate,
            "config_path": str(config_path),
            "run_dir": str(run_dir),
            "checkpoint": str(checkpoint),
            "config_hash": canonical_hash(config_to_plain(run_config)),
            "scientific_hash": canonical_hash(scientific_payload(run_config)),
            "base_config_hash": base_config_hash,
            "status": status,
            "hf_repo_id": args.hf_repo_id,
            "hf_path": hf_checkpoint_path(spec.run_name) if args.hf_repo_id else "",
            "hf_url": "",
            "hf_status": "pending" if args.hf_repo_id else "disabled",
        }
        manifest_rows.append(row)
        write_csv(manifest_path, manifest_rows, MANIFEST_FIELDS)

        if not args.execute:
            print(f"[{index}/{total_runs}] planned {spec.run_name}: {config_path}")
            continue
        if checkpoint_complete(checkpoint, bool(run_config.model.use_lora)) and not args.force:
            row["status"] = "checkpoint_exists"
            write_csv(manifest_path, manifest_rows, MANIFEST_FIELDS)
            print(f"[{index}/{total_runs}] skipping existing checkpoint: {checkpoint}")
        else:
            command = [sys.executable, "train_resolved_config.py", "--config", str(config_path)]
            print(f"[{index}/{total_runs}] training {spec.run_name}")
            row["status"] = "running"
            write_csv(manifest_path, manifest_rows, MANIFEST_FIELDS)
            try:
                subprocess.run(command, check=True, cwd=Path(__file__).resolve().parent)
            except subprocess.CalledProcessError:
                row["status"] = "failed"
                write_csv(manifest_path, manifest_rows, MANIFEST_FIELDS)
                raise
            if not checkpoint_complete(checkpoint, bool(run_config.model.use_lora)):
                row["status"] = "missing_checkpoint"
                write_csv(manifest_path, manifest_rows, MANIFEST_FIELDS)
                raise RuntimeError(
                    f"Training finished but a complete LATEST checkpoint was not created: "
                    f"{checkpoint}"
                )
            row["status"] = "trained"
            write_csv(manifest_path, manifest_rows, MANIFEST_FIELDS)

        if hf_api is not None:
            print(
                f"[{index}/{total_runs}] uploading {spec.run_name} to "
                f"{args.hf_repo_id}/{row['hf_path']}"
            )
            row["hf_status"] = "uploading"
            write_csv(manifest_path, manifest_rows, MANIFEST_FIELDS)
            try:
                destination, url = upload_checkpoint_to_hf(
                    api=hf_api,
                    repo_id=args.hf_repo_id,
                    row=row,
                    checkpoint=checkpoint,
                    adapter_only=args.hf_upload_adapter_only,
                )
            except Exception:
                row["hf_status"] = "failed"
                write_csv(manifest_path, manifest_rows, MANIFEST_FIELDS)
                raise
            row["hf_path"] = destination
            row["hf_url"] = url
            row["hf_status"] = "uploaded"
            write_csv(manifest_path, manifest_rows, MANIFEST_FIELDS)

    write_json(
        output_root / "config_audit.json",
        {
            "base_config": base_source,
            "base_config_hash": base_config_hash,
            "noise_manifest": str(noise_manifest),
            "num_runs": len(specs),
            "all_passed": all(audit["passed"] for audit in audits),
            "runs": audits,
        },
    )
    print(f"Wrote {len(specs)} audited run configs to {output_root}")
    print(f"Manifest: {manifest_path}")
    if not args.execute:
        print("Dry run only. Re-run with --execute after inspecting config_audit.json.")


if __name__ == "__main__":
    main()
