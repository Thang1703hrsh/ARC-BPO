#!/usr/bin/env python3
"""Run the current Llama-3-8B ARC-BPO sensitivity spec on Modal."""

from __future__ import annotations

import json
from pathlib import Path

import modal


APP_NAME = "arc-bpo-hyperparameter-sensitivity"
GPU_TYPE = "A100-80GB:4"
GPU_COUNT = 4
BASE_MODEL = "RLHFlow/LLaMA3-SFT-v2"
DATASET_REPO = "princeton-nlp/llama3-ultrafeedback-armorm"
REMOTE_REPO = "/root/arc-bpo"
CACHE_ROOT = "/cache"
RESULTS_ROOT = "/results"
_HERE = Path(__file__).resolve().parent

cache_volume = modal.Volume.from_name("arc-bpo-hf-cache", create_if_missing=True)
results_volume = modal.Volume.from_name(
    "arc-bpo-hyperparameter-sensitivity-results",
    create_if_missing=True,
)

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-devel-ubuntu22.04",
        add_python="3.11",
    )
    .apt_install("git")
    .uv_pip_install(
        "torch==2.6.0",
        "vllm==0.8.5",
        "transformers==4.52.4",
        "datasets==3.5.0",
        "accelerate==1.7.0",
        "peft==0.15.2",
        "hydra-core==1.3.2",
        "omegaconf==2.3.0",
        "wandb==0.20.1",
        "lm-eval[sentencepiece,vllm]==0.4.9.2",
        "matplotlib>=3.8.0",
        "scipy>=1.12.0",
        "sentencepiece>=0.2.0",
        "protobuf>=4.25.0",
        "tqdm>=4.67.0",
        "huggingface_hub[hf_xet]>=0.31.0,<1",
        "safetensors>=0.5.3",
    )
    .env(
        {
            "HF_HOME": f"{CACHE_ROOT}/huggingface",
            "HF_XET_HIGH_PERFORMANCE": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "WANDB_MODE": "disabled",
            "MPLBACKEND": "Agg",
            "PYTHONPATH": REMOTE_REPO,
        }
    )
    .add_local_file(
        str(_HERE / "run_sensitivity.py"),
        remote_path=f"{REMOTE_REPO}/run_sensitivity.py",
    )
    .add_local_file(
        str(_HERE / "train_resolved_config.py"),
        remote_path=f"{REMOTE_REPO}/train_resolved_config.py",
    )
    .add_local_file(
        str(_HERE / "evaluate_sensitivity.py"),
        remote_path=f"{REMOTE_REPO}/evaluate_sensitivity.py",
    )
    .add_local_file(
        str(_HERE / "summarize_sensitivity.py"),
        remote_path=f"{REMOTE_REPO}/summarize_sensitivity.py",
    )
    .add_local_file(
        str(_HERE / "ARC_BPO_Hyperparameter_Sensitivity_Codex_Spec.md"),
        remote_path=f"{REMOTE_REPO}/ARC_BPO_Hyperparameter_Sensitivity_Codex_Spec.md",
    )
    .add_local_file(str(_HERE / "train.py"), remote_path=f"{REMOTE_REPO}/train.py")
    .add_local_file(
        str(_HERE / "trainers.py"), remote_path=f"{REMOTE_REPO}/trainers.py"
    )
    .add_local_file(
        str(_HERE / "preference_datasets.py"),
        remote_path=f"{REMOTE_REPO}/preference_datasets.py",
    )
    .add_local_file(str(_HERE / "utils.py"), remote_path=f"{REMOTE_REPO}/utils.py")
    .add_local_file(str(_HERE / "merge.py"), remote_path=f"{REMOTE_REPO}/merge.py")
    .add_local_file(
        str(_HERE / "baseline_head.py"),
        remote_path=f"{REMOTE_REPO}/baseline_head.py",
    )
    .add_local_file(
        str(_HERE / "arc_bpo_chunking.py"),
        remote_path=f"{REMOTE_REPO}/arc_bpo_chunking.py",
    )
    .add_local_file(
        str(_HERE / "arc_bpo_scores.py"),
        remote_path=f"{REMOTE_REPO}/arc_bpo_scores.py",
    )
    .add_local_dir(str(_HERE / "config"), remote_path=f"{REMOTE_REPO}/config")
    .add_local_dir(str(_HERE / "loss"), remote_path=f"{REMOTE_REPO}/loss")
    .add_local_dir(
        str(_HERE / "sensitivity"), remote_path=f"{REMOTE_REPO}/sensitivity"
    )
)

app = modal.App(APP_NAME)
hf_secret = modal.Secret.from_name("huggingface-secret")


def _slug(value: str) -> str:
    slug = "".join(
        character if character.isalnum() or character in "-_." else "-"
        for character in value
    ).strip("-.")
    if not slug:
        raise ValueError("Output name must contain at least one safe character.")
    return slug


def _stream_command(command, log_path: Path) -> None:
    import os
    import subprocess

    normalized = [str(part) for part in command]
    printable = " ".join(normalized)
    print(f"[command] {printable}", flush=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONPATH": REMOTE_REPO,
            "CUDA_VISIBLE_DEVICES": "0,1,2,3",
            "WANDB_MODE": "disabled",
        }
    )
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"[command] {printable}\n")
        process = subprocess.Popen(
            normalized,
            cwd=REMOTE_REPO,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
        return_code = process.wait()
    if return_code:
        raise RuntimeError(
            f"Command failed with exit code {return_code}: {printable}. "
            f"See {log_path}."
        )


def _read_manifest(path: Path) -> list[dict]:
    import csv

    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_manifest(path: Path, rows: list[dict]) -> None:
    import csv

    if not rows:
        raise ValueError("Cannot write an empty sensitivity manifest.")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _checkpoint_complete(checkpoint: Path) -> bool:
    adapter = checkpoint / "adapter"
    if not (adapter / "adapter_config.json").is_file():
        return False
    weights = list(adapter.glob("adapter_model*.safetensors")) + list(
        adapter.glob("adapter_model*.bin")
    )
    return bool(weights) and all(path.stat().st_size > 1024 for path in weights)


def _same_run(existing: dict, current: dict) -> bool:
    keys = (
        "mode",
        "base_model",
        "base_revision",
        "dataset",
        "dataset_revision",
        "seed",
        "noise_seed",
        "n_examples",
        "global_batch_size",
        "gradient_accumulation_steps",
        "gpu_type",
        "spec_sha256",
        "hf_repo_id",
        "hf_private",
    )
    return all(existing.get(key) == current.get(key) for key in keys)


def _write_base_config(
    destination: Path,
    output_root: Path,
    *,
    mode: str,
    base_revision: str,
    dataset_revision: str,
    seed: int,
) -> None:
    from omegaconf import OmegaConf

    config = OmegaConf.load(f"{REMOTE_REPO}/config/config.yaml")
    config.pop("defaults", None)
    config.model = OmegaConf.load(f"{REMOTE_REPO}/config/model/llama_8b.yaml")
    config.loss = OmegaConf.load(f"{REMOTE_REPO}/config/loss/arc_bpo.yaml")

    config.seed = seed
    config.exp_name = "arc-bpo-sensitivity-default-llama3-10k-bs64"
    config.output_dir = str(output_root)
    config.local_run_dir = str(output_root / "base-config-only")
    config.fsdp_port = None
    config.datasets = DATASET_REPO
    config.dataset_revision = dataset_revision
    config.dataset_train_split = "train"
    config.dataset_test_split = "test"
    config.batch_size = 64
    config.gradient_accumulation_steps = 4
    config.n_examples = 64 if mode == "smoke" else 10000
    config.n_epochs = None
    config.skip_examples = 0
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
    config.save_checkpoint = True
    config.save_every_examples = 5000
    config.wandb.enabled = False

    config.model.name_or_path = BASE_MODEL
    config.model.revision = base_revision
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
    destination.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config, destination)


@app.function(
    image=image,
    gpu=GPU_TYPE,
    cpu=16.0,
    memory=262144,
    timeout=24 * 60 * 60,
    max_containers=1,
    secrets=[hf_secret],
    volumes={CACHE_ROOT: cache_volume, RESULTS_ROOT: results_volume},
)
def run_full_sensitivity(config: dict) -> dict:
    import hashlib
    import os
    import sys
    from datetime import datetime, timezone

    import torch
    from huggingface_hub import HfApi

    if torch.cuda.device_count() != GPU_COUNT:
        raise RuntimeError(f"Expected {GPU_COUNT} GPUs, found {torch.cuda.device_count()}.")
    gpu_names = [torch.cuda.get_device_name(index) for index in range(GPU_COUNT)]
    gpu_memory = [
        torch.cuda.get_device_properties(index).total_memory / 1024**3
        for index in range(GPU_COUNT)
    ]
    if any("A100" not in name.upper() or memory < 70 for name, memory in zip(gpu_names, gpu_memory)):
        raise RuntimeError(
            f"This setting requires four A100-80GB GPUs; got {list(zip(gpu_names, gpu_memory))}."
        )

    token = os.environ.get("HF_TOKEN")
    api = HfApi(token=token)
    base_revision = str(
        api.model_info(BASE_MODEL, revision=config["base_revision"] or None).sha
    )
    dataset_revision = str(
        api.dataset_info(DATASET_REPO, revision=config["dataset_revision"] or None).sha
    )
    spec_path = Path(f"{REMOTE_REPO}/ARC_BPO_Hyperparameter_Sensitivity_Codex_Spec.md")
    spec_sha256 = hashlib.sha256(spec_path.read_bytes()).hexdigest()

    run_slug = _slug(config["output_name"])
    output_root = (Path(RESULTS_ROOT) / "sensitivity" / run_slug / config["mode"]).resolve()
    if Path(RESULTS_ROOT).resolve() not in output_root.parents:
        raise ValueError(f"Unsafe output path: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    run_config = {
        **config,
        "base_model": BASE_MODEL,
        "base_revision": base_revision,
        "dataset": DATASET_REPO,
        "dataset_revision": dataset_revision,
        "spec_sha256": spec_sha256,
        "seed": 0,
        "noise_seed": 2026,
        "n_examples": 64 if config["mode"] == "smoke" else 10000,
        "global_batch_size": 64,
        "gradient_accumulation_steps": 4,
        "gpu_type": GPU_TYPE,
        "gpu_names": gpu_names,
        "gpu_memory_gib": gpu_memory,
        "num_new_training_runs": 14,
        "default_rows": "reused from immutable scores in the current Spec",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "status": "planning",
    }
    run_config_path = output_root / "modal_run_config.json"
    if run_config_path.is_file():
        existing = json.loads(run_config_path.read_text(encoding="utf-8"))
        if not _same_run(existing, run_config):
            raise RuntimeError(
                f"Output {output_root} belongs to a different configuration; "
                "choose a different --output-name."
            )
    run_config_path.write_text(json.dumps(run_config, indent=2) + "\n", encoding="utf-8")
    results_volume.commit()

    base_config = output_root / "default_config_source.yaml"
    _write_base_config(
        base_config,
        output_root,
        mode=config["mode"],
        base_revision=base_revision,
        dataset_revision=dataset_revision,
        seed=0,
    )
    plan_command = [
        sys.executable,
        f"{REMOTE_REPO}/run_sensitivity.py",
        "--base_config",
        base_config,
        "--output_root",
        output_root,
        "--seeds",
        "0",
        "--noise_rate",
        "0.20",
        "--noise_seed",
        "2026",
        "--exclude_default_points",
    ]
    if config["hf_repo_id"]:
        plan_command.extend(["--hf_repo_id", config["hf_repo_id"]])
    _stream_command(plan_command, output_root / "logs" / "plan.log")
    manifest_path = output_root / "run_manifest.csv"
    rows = _read_manifest(manifest_path)
    if len(rows) != 14:
        raise RuntimeError(f"Current spec must produce exactly 14 new runs, found {len(rows)}.")
    if config["hf_repo_id"]:
        api.create_repo(
            repo_id=config["hf_repo_id"],
            repo_type="model",
            private=config["hf_private"],
            exist_ok=True,
        )
    results_volume.commit()

    run_config["status"] = "training"
    run_config_path.write_text(json.dumps(run_config, indent=2) + "\n", encoding="utf-8")
    results_volume.commit()
    for index, row in enumerate(rows, start=1):
        checkpoint = Path(row["checkpoint"])
        if _checkpoint_complete(checkpoint) and not config["force"]:
            row["status"] = "checkpoint_exists"
            _write_manifest(manifest_path, rows)
            print(f"[train {index}/14] resume: {row['run_name']}", flush=True)
        else:
            row["status"] = "running"
            _write_manifest(manifest_path, rows)
            results_volume.commit()
            try:
                _stream_command(
                    [
                        sys.executable,
                        f"{REMOTE_REPO}/train_resolved_config.py",
                        "--config",
                        row["config_path"],
                    ],
                    output_root / "logs" / f"train_{index:02d}_{row['run_name']}.log",
                )
            except Exception:
                row["status"] = "failed"
                _write_manifest(manifest_path, rows)
                results_volume.commit()
                raise
            if not _checkpoint_complete(checkpoint):
                row["status"] = "incomplete_checkpoint"
                _write_manifest(manifest_path, rows)
                results_volume.commit()
                raise RuntimeError(f"Incomplete LoRA checkpoint after training: {checkpoint}")
            row["status"] = "trained"
            _write_manifest(manifest_path, rows)
            results_volume.commit()

        if config["hf_repo_id"]:
            from run_sensitivity import upload_checkpoint_to_hf

            row["hf_status"] = "uploading"
            _write_manifest(manifest_path, rows)
            results_volume.commit()
            try:
                hf_path, hf_url = upload_checkpoint_to_hf(
                    api=api,
                    repo_id=config["hf_repo_id"],
                    row=row,
                    checkpoint=checkpoint,
                    adapter_only=True,
                )
            except Exception:
                row["hf_status"] = "failed"
                _write_manifest(manifest_path, rows)
                results_volume.commit()
                raise
            row["hf_path"] = hf_path
            row["hf_url"] = hf_url
            row["hf_status"] = "uploaded"
            _write_manifest(manifest_path, rows)
            results_volume.commit()

    run_config["status"] = "evaluating"
    run_config_path.write_text(json.dumps(run_config, indent=2) + "\n", encoding="utf-8")
    results_volume.commit()
    for index, row in enumerate(rows, start=1):
        _stream_command(
            [
                sys.executable,
                f"{REMOTE_REPO}/evaluate_sensitivity.py",
                "--manifest",
                manifest_path,
                "--run_names",
                row["run_name"],
                "--only_status",
                "trained,checkpoint_exists",
                "--tensor_parallel_size",
                "4",
                "--dtype",
                "bfloat16",
                "--gpu_memory_utilization",
                "0.90",
                "--max_model_len",
                "4096",
                "--batch_size",
                "auto:4",
                "--merge_device",
                "auto",
                "--merge_dtype",
                "bfloat16",
            ],
            output_root / "logs" / f"evaluate_{index:02d}_{row['run_name']}.log",
        )
        results_volume.commit()

    _stream_command(
        [
            sys.executable,
            f"{REMOTE_REPO}/summarize_sensitivity.py",
            "--manifest",
            manifest_path,
            "--published_anchors",
            f"{REMOTE_REPO}/sensitivity/published_anchors.json",
            "--main_result",
            f"{REMOTE_REPO}/sensitivity/published_main_result.json",
        ],
        output_root / "logs" / "summarize.log",
    )

    run_config.update(
        {
            "status": "complete",
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "manifest": str(manifest_path),
            "sensitivity_results_csv": str(output_root / "sensitivity_results.csv"),
            "sensitivity_results_json": str(output_root / "sensitivity_results.json"),
            "summary": str(output_root / "summary.md"),
        }
    )
    run_config_path.write_text(json.dumps(run_config, indent=2) + "\n", encoding="utf-8")
    result = {
        "status": "complete",
        "output_root": str(output_root),
        "new_training_runs": 14,
        "seed": 0,
        "n_examples": run_config["n_examples"],
        "global_batch_size": 64,
        "gpu": gpu_names,
        "manifest": str(manifest_path),
        "summary": str(output_root / "summary.md"),
    }
    (output_root / "modal_result.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    results_volume.commit()
    return result


@app.local_entrypoint()
def main(
    mode: str = "smoke",
    output_name: str = "llama3-10k-bs64-spec-v2",
    base_revision: str = "",
    dataset_revision: str = "",
    hf_repo_id: str = "",
    hf_private: bool = False,
    force: bool = False,
):
    if mode not in {"smoke", "full"}:
        raise ValueError("--mode must be smoke or full.")
    config = {
        "mode": mode,
        "output_name": output_name,
        "base_revision": base_revision,
        "dataset_revision": dataset_revision,
        "hf_repo_id": hf_repo_id,
        "hf_private": hf_private,
        "force": force,
    }
    function_call = run_full_sensitivity.spawn(config)
    result = function_call.get()
    print(json.dumps(result, indent=2, ensure_ascii=False))
