#!/usr/bin/env python3
"""Run Mistral-7B-v0.1 ARC-BPO allocation ablations on Modal."""

from __future__ import annotations

import json
from pathlib import Path

import modal


APP_NAME = "arc-bpo-mistral-allocation-ablation"
GPU_TYPE = "H100!"
GPU_COUNT = 1
CPU_COUNT = 4.0
MEMORY_MIB = 64 * 1024
BASE_MODEL = "HuggingFaceH4/mistral-7b-sft-alpha"
DATASET_REPO = "HuggingFaceH4/ultrafeedback_binarized"
UNIFORM_CHECKPOINT = "ducthang1703/mistral7b-arc-bpo-uniform-lora-16k-bs64-seed0"

REMOTE_REPO = "/root/arc-bpo"
CACHE_ROOT = "/cache"
RESULTS_ROOT = "/results"
_HERE = Path(__file__).resolve().parent

cache_volume = modal.Volume.from_name("arc-bpo-hf-cache", create_if_missing=True)
results_volume = modal.Volume.from_name(
    "arc-bpo-mistral-allocation-ablation-results",
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
        "transformers==4.52.4",
        "datasets==3.5.0",
        "accelerate==1.7.0",
        "peft==0.14.0",
        "hydra-core==1.3.2",
        "omegaconf==2.3.0",
        "wandb==0.20.1",
        "sentencepiece==0.2.0",
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
            "PYTHONPATH": REMOTE_REPO,
        }
    )
    .add_local_file(
        str(_HERE / "run_allocation_ablations.py"),
        remote_path=f"{REMOTE_REPO}/run_allocation_ablations.py",
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
    .add_local_file(
        str(_HERE / "huggingface_utils.py"),
        remote_path=f"{REMOTE_REPO}/huggingface_utils.py",
    )
    .add_local_file(
        str(_HERE / "script" / "train" / "arc_bpo_mistral.sh"),
        remote_path=f"{REMOTE_REPO}/script/train/arc_bpo_mistral.sh",
    )
    .add_local_dir(str(_HERE / "config"), remote_path=f"{REMOTE_REPO}/config")
    .add_local_dir(str(_HERE / "loss"), remote_path=f"{REMOTE_REPO}/loss")
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


def _same_run(existing: dict, current: dict) -> bool:
    keys = (
        "mode",
        "variant",
        "base_model",
        "base_revision",
        "dataset",
        "dataset_revision",
        "uniform_checkpoint",
        "uniform_revision",
        "seed",
        "n_examples",
        "global_batch_size",
        "gradient_accumulation_steps",
        "gpu_type",
        "cpu_count",
        "memory_mib",
    )
    return all(existing.get(key) == current.get(key) for key in keys)


def _stream_command(command: list[str], log_path: Path) -> None:
    import os
    import subprocess

    printable = " ".join(str(part) for part in command)
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
            command,
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
        raise subprocess.CalledProcessError(return_code, command)


def _resolve_hf_metadata(config: dict) -> dict:
    import os

    from huggingface_hub import HfApi

    token = os.environ.get("HF_TOKEN")
    api = HfApi(token=token)
    base_info = api.model_info(BASE_MODEL, revision=config["base_revision"] or None)
    dataset_info = api.dataset_info(
        DATASET_REPO,
        revision=config["dataset_revision"] or None,
    )
    cache_volume.commit()
    return {
        "base_revision": str(base_info.sha),
        "dataset_revision": str(dataset_info.sha),
        "uniform_revision": str(config.get("uniform_revision") or ""),
        "uniform_adapter_config": None,
        "uniform_recorded_base": None,
    }


def _find_variant_row(manifest_path: Path, variant: str) -> dict:
    import csv

    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    matches = [row for row in rows if row.get("variant") == variant]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one {variant!r} row in {manifest_path}, found {len(matches)}."
        )
    return matches[0]


def _adapter_complete(adapter: Path) -> bool:
    if not (adapter / "adapter_config.json").is_file():
        return False
    weights = list(adapter.glob("adapter_model*.safetensors")) + list(
        adapter.glob("adapter_model*.bin")
    )
    return bool(weights) and all(path.stat().st_size > 1024 for path in weights)


def _upload_adapter(adapter: Path, run_dir: Path, output_root: Path, config: dict) -> dict:
    import os
    import shutil
    import tempfile

    from huggingface_hub import HfApi, upload_folder

    repo_id = str(config["hf_repo_id"]).strip()
    if not repo_id:
        return {"status": "disabled"}
    token = os.environ.get("HF_TOKEN")
    api = HfApi(token=token)
    api.create_repo(
        repo_id=repo_id,
        private=bool(config["hf_private"]),
        exist_ok=True,
    )
    with tempfile.TemporaryDirectory(prefix="arc-bpo-allocation-export-") as temporary:
        export = Path(temporary)
        for source in adapter.iterdir():
            destination = export / source.name
            if source.is_dir():
                shutil.copytree(source, destination)
            else:
                shutil.copy2(source, destination)
        shutil.copy2(run_dir / "config.yaml", export / "training_config.yaml")
        shutil.copy2(output_root / "config_audit.json", export / "allocation_config_audit.json")
        shutil.copy2(output_root / "modal_run_config.json", export / "modal_run_config.json")
        upload_folder(
            repo_id=repo_id,
            folder_path=export,
            commit_message=f"Upload ARC-BPO {config['variant']} LoRA checkpoint",
            token=token,
        )
    return {
        "status": "uploaded",
        "repo_id": repo_id,
        "url": f"https://huggingface.co/{repo_id}",
        "private": bool(config["hf_private"]),
    }


@app.function(
    image=image,
    gpu=GPU_TYPE,
    cpu=CPU_COUNT,
    memory=MEMORY_MIB,
    timeout=24 * 60 * 60,
    max_containers=1,
    secrets=[hf_secret],
    volumes={CACHE_ROOT: cache_volume, RESULTS_ROOT: results_volume},
)
def train_allocation_ablation(config: dict) -> dict:
    import sys
    from datetime import datetime, timezone

    import torch

    if torch.cuda.device_count() != GPU_COUNT:
        raise RuntimeError(
            f"Expected exactly {GPU_COUNT} Modal GPUs, found {torch.cuda.device_count()}."
        )
    gpu_names = [torch.cuda.get_device_name(index) for index in range(GPU_COUNT)]
    gpu_memory_gib = [
        torch.cuda.get_device_properties(index).total_memory / 1024**3
        for index in range(GPU_COUNT)
    ]
    invalid = [
        (name, memory)
        for name, memory in zip(gpu_names, gpu_memory_gib)
        if "H100" not in name.upper() or memory < 70
    ]
    if invalid:
        raise RuntimeError(f"Expected one H100-80GB GPU; incompatible devices: {invalid}")
    print(
        f"[gpu] names={gpu_names}; memory_gib={gpu_memory_gib}; count={GPU_COUNT}",
        flush=True,
    )

    variant = str(config["variant"])
    if variant not in {"uniform", "advantage", "advantage_sba_no_winsor"}:
        raise ValueError(f"Unsupported Modal allocation variant: {variant!r}.")
    if int(config["global_batch_size"]) != 64:
        raise ValueError("This Modal setting is pinned to global batch size 64.")
    if int(config["seed"]) != 0:
        raise ValueError("This reviewer setting is pinned to the single matched seed 0.")

    revisions = _resolve_hf_metadata(config)
    run_slug = _slug(config["output_name"])
    output_root = (
        Path(RESULTS_ROOT) / "allocation_ablations" / run_slug / str(config["mode"])
    ).resolve()
    if Path(RESULTS_ROOT).resolve() not in output_root.parents:
        raise ValueError(f"Unsafe Modal output path: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    run_config = {
        **config,
        **revisions,
        "base_model": BASE_MODEL,
        "dataset": DATASET_REPO,
        "gpu_type": GPU_TYPE,
        "gpu_name": gpu_names,
        "gpu_memory_gib": gpu_memory_gib,
        "gpu_count": GPU_COUNT,
        "cpu_count": CPU_COUNT,
        "memory_mib": MEMORY_MIB,
        "microbatch_per_gpu": (
            int(config["global_batch_size"])
            // (int(config["gradient_accumulation_steps"]) * GPU_COUNT)
        ),
        "initialization": "fresh LoRA on HuggingFaceH4/mistral-7b-sft-alpha",
        "checkpoint_policy": "final 16k checkpoint only; no intermediate snapshots",
        "uniform_checkpoint_role": "existing result reference only; never initialization",
        "output_root": str(output_root),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "status": "starting",
    }
    config_path = output_root / "modal_run_config.json"
    if config_path.is_file():
        existing = json.loads(config_path.read_text(encoding="utf-8"))
        if not _same_run(existing, run_config):
            raise RuntimeError(
                f"Output {output_root} belongs to a different configuration. "
                "Choose a new --output-name; --force cannot mix configurations."
            )
    config_path.write_text(json.dumps(run_config, indent=2) + "\n", encoding="utf-8")
    results_volume.commit()

    variants_arg = variant

    command = [
        sys.executable,
        f"{REMOTE_REPO}/run_allocation_ablations.py",
        "--variants",
        variants_arg,
        "--seed",
        str(config["seed"]),
        "--gpu_ids",
        "0",
        "--grad_accum",
        str(config["gradient_accumulation_steps"]),
        "--n_examples",
        str(config["n_examples"]),
        "--global_batch_size",
        str(config["global_batch_size"]),
        "--base_revision",
        revisions["base_revision"],
        "--dataset_revision",
        revisions["dataset_revision"],
        "--output_root",
        str(output_root),
        "--uniform_checkpoint",
        str(config["uniform_checkpoint"]),
        "--execute",
    ]
    if config["force"]:
        command.append("--force")

    run_config["status"] = "running"
    config_path.write_text(json.dumps(run_config, indent=2) + "\n", encoding="utf-8")
    results_volume.commit()
    try:
        _stream_command(command, output_root / "logs" / "modal_launcher.log")
    except Exception as error:
        run_config["status"] = "failed"
        run_config["error"] = f"{type(error).__name__}: {error}"
        run_config["finished_at"] = datetime.now(timezone.utc).isoformat()
        config_path.write_text(json.dumps(run_config, indent=2) + "\n", encoding="utf-8")
        results_volume.commit()
        raise

    manifest_path = output_root / "run_manifest.csv"
    row = _find_variant_row(manifest_path, variant)
    if row["status"] not in {"trained", "checkpoint_exists"}:
        raise RuntimeError(f"Unexpected final manifest status for {variant}: {row['status']}")
    adapter = Path(row["checkpoint"])
    if not _adapter_complete(adapter):
        raise RuntimeError(f"Incomplete LoRA adapter checkpoint: {adapter}")
    run_dir = Path(row["run_dir"])
    if not (run_dir / "config.yaml").is_file():
        raise FileNotFoundError(f"Resolved training config missing: {run_dir / 'config.yaml'}")

    upload = _upload_adapter(adapter, run_dir, output_root, config)
    run_config.update(
        {
            "status": "complete",
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "adapter": str(adapter),
            "resolved_training_config": str(run_dir / "config.yaml"),
            "manifest_status": row["status"],
            "upload": upload,
        }
    )
    config_path.write_text(json.dumps(run_config, indent=2) + "\n", encoding="utf-8")
    result = {
        "status": "complete",
        "variant": variant,
        "seed": int(config["seed"]),
        "n_examples": int(config["n_examples"]),
        "global_batch_size": int(config["global_batch_size"]),
        "gradient_accumulation_steps": int(config["gradient_accumulation_steps"]),
        "gpu": gpu_names,
        "output_root": str(output_root),
        "adapter": str(adapter),
        "manifest": str(manifest_path),
        "config_audit": str(output_root / "config_audit.json"),
        "upload": upload,
    }
    (output_root / "modal_result.json").write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    results_volume.commit()
    return result


@app.local_entrypoint()
def main(
    mode: str = "smoke",
    variant: str = "advantage_sba_no_winsor",
    output_name: str = "",
    uniform_checkpoint: str = UNIFORM_CHECKPOINT,
    uniform_revision: str = "",
    base_revision: str = "",
    dataset_revision: str = "",
    seed: int = 0,
    n_examples: int = -1,
    global_batch_size: int = 64,
    gradient_accumulation_steps: int = 64,
    hf_repo_id: str = "",
    hf_private: bool = True,
    force: bool = False,
    background: bool = False,
):
    if mode == "smoke":
        effective_examples = 64 if n_examples < 0 else n_examples
    elif mode == "full":
        effective_examples = 16000 if n_examples < 0 else n_examples
    else:
        raise ValueError("--mode must be smoke or full.")
    if effective_examples <= 0:
        raise ValueError("--n-examples must be positive.")
    if gradient_accumulation_steps <= 0:
        raise ValueError("--gradient-accumulation-steps must be positive.")
    if global_batch_size % (gradient_accumulation_steps * GPU_COUNT) != 0:
        raise ValueError(
            "Global batch size must be divisible by gradient accumulation steps * GPU count."
        )

    effective_output_name = output_name.strip()
    if not effective_output_name:
        dataset_label = "16k" if effective_examples == 16000 else f"{effective_examples}examples"
        effective_output_name = (
            f"mistral7b-{variant}-lora-{dataset_label}-bs{global_batch_size}-seed{seed}"
        )

    config = {
        "mode": mode,
        "variant": variant,
        "output_name": effective_output_name,
        "uniform_checkpoint": uniform_checkpoint,
        "uniform_revision": uniform_revision,
        "base_revision": base_revision,
        "dataset_revision": dataset_revision,
        "seed": seed,
        "n_examples": effective_examples,
        "global_batch_size": global_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "hf_repo_id": hf_repo_id,
        "hf_private": hf_private,
        "force": force,
    }
    if background:
        call = train_allocation_ablation.spawn(config)
        print(json.dumps({"status": "submitted", "call_id": call.object_id}, indent=2))
    else:
        result = train_allocation_ablation.remote(config)
        print(json.dumps(result, indent=2, ensure_ascii=False))
