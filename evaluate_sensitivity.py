#!/usr/bin/env python3
"""Evaluate final ARC-BPO sensitivity checkpoints with one frozen protocol."""

from __future__ import annotations

import argparse
import importlib.metadata
import math
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from sensitivity.common import (
    LM_EVAL_TASKS,
    TASKS,
    canonical_hash,
    config_to_plain,
    load_resolved_config,
    read_json,
    read_manifest,
    write_csv,
    write_json,
)


EVALUATION_MANIFEST_FIELDS = (
    "run_name",
    "sweep",
    "value",
    "seed",
    "noise_rate",
    "checkpoint",
    "result_path",
    "protocol_hash",
    "status",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate every final LATEST checkpoint in a sensitivity manifest. "
            "LoRA adapters are merged with their recorded base model before evaluation."
        )
    )
    parser.add_argument("--manifest", default="outputs/sensitivity/run_manifest.csv")
    parser.add_argument(
        "--lm_eval_command",
        default="lm_eval",
        help="Harness command prefix matching the repo's pinned 0.4.9.2 environment.",
    )
    parser.add_argument("--model_backend", choices=("vllm", "hf"), default="vllm")
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    parser.add_argument("--max_model_len", type=int, default=4096)
    parser.add_argument("--batch_size", default="auto")
    parser.add_argument("--device", default="cuda:0", help="Used only by the hf backend.")
    parser.add_argument("--score_scale", type=float, default=100.0)
    parser.add_argument("--evaluation_seed", default="0,1234,1234,1234")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--log_samples", action="store_true")
    parser.add_argument("--merge_device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--merge_dtype",
        choices=("auto", "float16", "bfloat16", "float32"),
        default="bfloat16",
    )
    parser.add_argument(
        "--merge_root",
        help="Keep merged LoRA models under this directory. By default they are temporary.",
    )
    parser.add_argument("--only_status", default="", help="Comma-separated manifest statuses.")
    parser.add_argument(
        "--run_names",
        default="",
        help="Optional comma-separated exact run names to evaluate.",
    )
    parser.add_argument("--max_runs", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print commands without requiring checkpoints or running the harness.",
    )
    return parser.parse_args()


def _installed_version(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def evaluation_protocol(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "version": 1,
        "harness": "lm-evaluation-harness",
        "harness_version": _installed_version("lm_eval"),
        "command_prefix": shlex.split(args.lm_eval_command, posix=True),
        "model_backend": args.model_backend,
        "tasks": LM_EVAL_TASKS,
        "batch_size": args.batch_size,
        "dtype": args.dtype,
        "tensor_parallel_size": args.tensor_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_model_len": args.max_model_len,
        "device": args.device if args.model_backend == "hf" else None,
        "score_scale": args.score_scale,
        "evaluation_seed": args.evaluation_seed,
        "trust_remote_code": args.trust_remote_code,
        "apply_chat_template": False,
    }


def _adapter_path(checkpoint: Path) -> Optional[Path]:
    nested = checkpoint / "adapter"
    if (nested / "adapter_config.json").is_file():
        return nested
    if (checkpoint / "adapter_config.json").is_file():
        return checkpoint
    return None


def _merge_adapter(
    base_model: str,
    base_revision: str | None,
    adapter: Path,
    output: Path,
    args: argparse.Namespace,
):
    command = [
        sys.executable,
        str(Path(__file__).resolve().parent / "merge.py"),
        "--base_model",
        base_model,
        "--adapter",
        str(adapter),
        "--output",
        str(output),
        "--device",
        args.merge_device,
        "--dtype",
        args.merge_dtype,
    ]
    if base_revision:
        command.extend(("--base_revision", base_revision))
    print(" ".join(shlex.quote(part) for part in command))
    if not args.dry_run:
        subprocess.run(command, check=True, cwd=Path(__file__).resolve().parent)


def build_lm_eval_command(
    model_path: Path,
    task_name: str,
    output_path: Path,
    args: argparse.Namespace,
) -> List[str]:
    task = LM_EVAL_TASKS[task_name]
    if args.model_backend == "vllm":
        model_args = ",".join(
            (
                f"pretrained={model_path}",
                f"tensor_parallel_size={args.tensor_parallel_size}",
                f"dtype={args.dtype}",
                f"gpu_memory_utilization={args.gpu_memory_utilization}",
                f"max_model_len={args.max_model_len}",
            )
        )
    else:
        model_args = f"pretrained={model_path},dtype={args.dtype}"

    command = shlex.split(args.lm_eval_command, posix=True)
    command.extend(
        [
            "--model",
            args.model_backend,
            "--model_args",
            model_args,
            "--tasks",
            str(task["task"]),
            "--num_fewshot",
            str(task["fewshot"]),
            "--batch_size",
            args.batch_size,
            "--seed",
            args.evaluation_seed,
            "--output_path",
            str(output_path),
        ]
    )
    if args.model_backend == "hf":
        command.extend(("--device", args.device))
    if args.trust_remote_code:
        command.append("--trust_remote_code")
    if args.log_samples:
        command.append("--log_samples")
    return command


def _result_candidates(output_path: Path) -> List[Path]:
    return sorted(
        (
            path
            for path in output_path.rglob("*.json")
            if path.name == "results.json" or path.name.startswith("results_")
        ),
        key=lambda path: path.stat().st_mtime,
    )


def _metric_block(payload: Mapping[str, Any], harness_task: str) -> Mapping[str, Any]:
    for section in ("results", "groups"):
        values = payload.get(section, {})
        if harness_task in values:
            return values[harness_task]
    raise KeyError(f"Harness result has no task/group entry for {harness_task!r}.")


def extract_task_score(payload: Mapping[str, Any], task_name: str) -> float:
    task = LM_EVAL_TASKS[task_name]
    metrics = _metric_block(payload, str(task["task"]))
    for metric in task["metrics"]:
        if metric in metrics:
            value = float(metrics[metric])
            if not math.isfinite(value):
                raise ValueError(f"Non-finite {task_name} metric {metric}: {value}")
            return value
    raise KeyError(
        f"No approved metric for {task_name}; expected one of {task['metrics']}, "
        f"found {sorted(metrics)}."
    )


def _evaluate_model(
    model_path: Path,
    evaluation_dir: Path,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    task_scores: Dict[str, float] = {}
    raw_results: Dict[str, str] = {}
    for task_name in TASKS:
        task_dir = evaluation_dir / "raw" / task_name
        command = build_lm_eval_command(model_path, task_name, task_dir, args)
        print(" ".join(shlex.quote(part) for part in command))
        if args.dry_run:
            continue
        task_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(command, check=True, cwd=Path(__file__).resolve().parent)
        candidates = _result_candidates(task_dir)
        if not candidates:
            raise FileNotFoundError(f"No lm-eval results JSON found below {task_dir}.")
        result_path = candidates[-1]
        raw_results[task_name] = str(result_path)
        task_scores[task_name] = extract_task_score(read_json(result_path), task_name)
    return {"task_scores": task_scores, "raw_results": raw_results}


def _filtered_rows(rows: Sequence[Dict[str, str]], args: argparse.Namespace):
    statuses = {item.strip() for item in args.only_status.split(",") if item.strip()}
    run_names = {item.strip() for item in args.run_names.split(",") if item.strip()}
    selected = [
        row
        for row in rows
        if (not statuses or row.get("status") in statuses)
        and (not run_names or row.get("run_name") in run_names)
    ]
    if run_names:
        found = {row.get("run_name") for row in selected}
        missing = sorted(run_names - found)
        if missing:
            raise ValueError(f"Requested run names were not found after filtering: {missing}")
    return selected[: args.max_runs] if args.max_runs else selected


def main():
    args = parse_args()
    if args.tensor_parallel_size < 1 or args.max_model_len < 1:
        raise ValueError("tensor_parallel_size and max_model_len must be positive.")
    if not 0.0 < args.gpu_memory_utilization <= 1.0:
        raise ValueError("gpu_memory_utilization must be in (0, 1].")
    if args.score_scale <= 0 or args.max_runs < 0:
        raise ValueError("score_scale must be positive and max_runs cannot be negative.")

    manifest_path = Path(args.manifest).resolve()
    rows = _filtered_rows(read_manifest(manifest_path), args)
    protocol = evaluation_protocol(args)
    protocol_hash = canonical_hash(protocol)
    write_json(manifest_path.parent / "evaluation_protocol.json", {**protocol, "hash": protocol_hash})

    evaluation_manifest = manifest_path.parent / "evaluation_manifest.csv"
    evaluation_rows: List[Dict[str, Any]] = (
        list(read_manifest(evaluation_manifest)) if evaluation_manifest.is_file() else []
    )
    for index, row in enumerate(rows, start=1):
        checkpoint = Path(row["checkpoint"]).resolve()
        run_dir = Path(row["run_dir"]).resolve()
        evaluation_dir = run_dir / "evaluation"
        result_path = run_dir / "evaluation_metrics.json"
        output_row = {
            "run_name": row["run_name"],
            "sweep": row["sweep"],
            "value": row["value"],
            "seed": row["seed"],
            "noise_rate": row["noise_rate"],
            "checkpoint": str(checkpoint),
            "result_path": str(result_path),
            "protocol_hash": protocol_hash,
            "status": "planned" if args.dry_run else "running",
        }
        evaluation_rows = [
            existing
            for existing in evaluation_rows
            if existing.get("run_name") != output_row["run_name"]
        ]
        evaluation_rows.append(output_row)
        write_csv(evaluation_manifest, evaluation_rows, EVALUATION_MANIFEST_FIELDS)

        if result_path.is_file() and not args.force:
            existing = read_json(result_path)
            if existing.get("evaluation_protocol_hash") != protocol_hash:
                raise ValueError(
                    f"Existing result uses a different protocol: {result_path}. "
                    "Use a new output tree or inspect and re-run with --force."
                )
            if existing.get("config_hash") != row["config_hash"]:
                raise ValueError(f"Existing result/config mismatch: {result_path}")
            output_row["status"] = "result_exists"
            write_csv(evaluation_manifest, evaluation_rows, EVALUATION_MANIFEST_FIELDS)
            print(f"[{index}/{len(rows)}] result exists: {result_path}")
            continue
        if not checkpoint.is_dir() and not args.dry_run:
            output_row["status"] = "missing_checkpoint"
            write_csv(evaluation_manifest, evaluation_rows, EVALUATION_MANIFEST_FIELDS)
            raise FileNotFoundError(f"Final checkpoint not found: {checkpoint}")

        config = load_resolved_config(row["config_path"])
        adapter = _adapter_path(checkpoint)
        if args.dry_run and adapter is None and bool(getattr(config.model, "use_lora", False)):
            adapter = checkpoint / "adapter"
        temporary: Optional[tempfile.TemporaryDirectory] = None
        try:
            if adapter is None:
                model_path = checkpoint
                if not args.dry_run and not (model_path / "config.json").is_file():
                    raise FileNotFoundError(
                        f"Checkpoint is neither a full model nor a LoRA adapter: {checkpoint}"
                    )
            elif args.merge_root:
                model_path = Path(args.merge_root).resolve() / row["run_name"]
                if not (model_path / "config.json").is_file():
                    _merge_adapter(
                        str(config.model.name_or_path),
                        str(getattr(config.model, "revision", "") or "") or None,
                        adapter,
                        model_path,
                        args,
                    )
            else:
                temporary = tempfile.TemporaryDirectory(prefix="arc-bpo-sensitivity-")
                model_path = Path(temporary.name) / "merged"
                _merge_adapter(
                    str(config.model.name_or_path),
                    str(getattr(config.model, "revision", "") or "") or None,
                    adapter,
                    model_path,
                    args,
                )

            print(f"[{index}/{len(rows)}] evaluating {row['run_name']}")
            evaluated = _evaluate_model(model_path, evaluation_dir, args)
            if args.dry_run:
                output_row["status"] = "dry_run"
                continue
            scaled = {
                task: score * args.score_scale for task, score in evaluated["task_scores"].items()
            }
            if set(scaled) != set(TASKS):
                raise RuntimeError(f"Incomplete task results for {row['run_name']}: {sorted(scaled)}")
            average = sum(scaled.values()) / len(TASKS)
            write_json(
                result_path,
                {
                    "version": 1,
                    "run_name": row["run_name"],
                    "sweep": row["sweep"],
                    "parameter": row["parameter"],
                    "value": row["value"],
                    "numeric_value": row["numeric_value"],
                    "seed": int(row["seed"]),
                    "noise_rate": float(row["noise_rate"]),
                    "checkpoint": str(checkpoint),
                    "config_path": row["config_path"],
                    "config_hash": row["config_hash"],
                    "scientific_hash": row["scientific_hash"],
                    "base_config_hash": row["base_config_hash"],
                    "evaluation_protocol_hash": protocol_hash,
                    "score_scale": args.score_scale,
                    "tasks": scaled,
                    "average": average,
                    "raw_results": evaluated["raw_results"],
                },
            )
            output_row["status"] = "evaluated"
        except Exception:
            output_row["status"] = "failed"
            raise
        finally:
            if temporary is not None:
                temporary.cleanup()
            write_csv(evaluation_manifest, evaluation_rows, EVALUATION_MANIFEST_FIELDS)

    print(f"Evaluation manifest: {evaluation_manifest}")


if __name__ == "__main__":
    main()
