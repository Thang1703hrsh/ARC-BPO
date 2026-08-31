#!/usr/bin/env python3
"""Run the ARC-BPO generation-diversity reviewer analysis on one Modal A100."""

from __future__ import annotations

import json
from pathlib import Path

import modal


APP_NAME = "arc-bpo-generation-diversity"
CHECKPOINT_REPO = "ducthang1703/llama3-arc-bpo-uniform-lora-10k-bs64"
REFERENCE_MODEL = "RLHFlow/LLaMA3-SFT-v2"
DATASET_REPO = "princeton-nlp/llama3-ultrafeedback-armorm"
# These paths belong to the Linux Modal container. Keep them as POSIX strings so
# importing this file on Windows cannot turn them into WindowsPath instances.
REMOTE_REPO = "/root/arc-bpo"
CACHE_ROOT = "/cache"
RESULTS_ROOT = "/results"

_HERE = Path(__file__).resolve().parent

cache_volume = modal.Volume.from_name("arc-bpo-hf-cache", create_if_missing=True)
results_volume = modal.Volume.from_name(
    "arc-bpo-generation-diversity-results", create_if_missing=True
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
        "huggingface_hub[hf_xet]>=0.31.0,<1",
        "safetensors>=0.5.3",
        "sentencepiece>=0.2.0",
        "protobuf>=4.25.0",
        "scipy>=1.12.0",
        "matplotlib>=3.8.0",
        "sacrebleu>=2.5.0",
        "tqdm>=4.67.0",
    )
    .env(
        {
            "HF_HOME": f"{CACHE_ROOT}/huggingface",
            "HF_XET_HIGH_PERFORMANCE": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "MPLBACKEND": "Agg",
            "PYTHONPATH": REMOTE_REPO,
        }
    )
    .add_local_file(
        str(_HERE / "analyze_credit_drift.py"),
        remote_path=f"{REMOTE_REPO}/analyze_credit_drift.py",
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
        str(_HERE / "preference_datasets.py"),
        remote_path=f"{REMOTE_REPO}/preference_datasets.py",
    )
    .add_local_file(str(_HERE / "utils.py"), remote_path=f"{REMOTE_REPO}/utils.py")
    .add_local_file(str(_HERE / "merge.py"), remote_path=f"{REMOTE_REPO}/merge.py")
    .add_local_dir(str(_HERE / "loss"), remote_path=f"{REMOTE_REPO}/loss")
    .add_local_dir(
        str(_HERE / "diversity_metrics"),
        remote_path=f"{REMOTE_REPO}/diversity_metrics",
    )
)

app = modal.App(APP_NAME)
hf_secret = modal.Secret.from_name("huggingface-secret")


def _slug(value: str) -> str:
    return "".join(character if character.isalnum() or character in "-_." else "-" for character in value)


def _resolved_snapshot(repo_id: str, revision: str, repo_type: str = "model"):
    import os

    from huggingface_hub import HfApi, snapshot_download

    token = os.environ.get("HF_TOKEN")
    api = HfApi(token=token)
    if repo_type == "dataset":
        info = api.dataset_info(repo_id, revision=revision or None)
    else:
        info = api.model_info(repo_id, revision=revision or None)
    commit = str(info.sha)
    destination = Path(CACHE_ROOT) / "snapshots" / repo_type / _slug(repo_id) / commit
    marker = destination / ".complete"
    if not marker.is_file():
        destination.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=repo_id,
            repo_type=repo_type,
            revision=commit,
            local_dir=destination,
            token=token,
        )
        marker.write_text(commit + "\n", encoding="utf-8")
    return destination, commit


def _resolve_dataset_revision(repo_id: str, revision: str) -> str:
    import os

    from huggingface_hub import HfApi

    return str(HfApi(token=os.environ.get("HF_TOKEN")).dataset_info(
        repo_id,
        revision=revision or None,
    ).sha)


def _find_adapter(snapshot: Path) -> Path:
    candidates = sorted(snapshot.rglob("adapter_config.json"))
    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected exactly one adapter_config.json in {snapshot}, found {len(candidates)}."
        )
    adapter = candidates[0].parent
    weight_files = list(adapter.glob("adapter_model*.safetensors")) + list(
        adapter.glob("adapter_model*.bin")
    )
    if not weight_files:
        raise FileNotFoundError(f"No PEFT adapter weights found in {adapter}.")
    return adapter


def _stream_command(command, log_path: Path):
    import os
    import subprocess

    normalized_command = [str(part) for part in command]
    printable = " ".join(normalized_command)
    print(f"[command] {printable}", flush=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = REMOTE_REPO
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"[command] {printable}\n")
        process = subprocess.Popen(
            normalized_command,
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
            f"See the full command output in {log_path}."
        )


def _credit_outputs_complete(output: Path) -> bool:
    required = (
        "chunk_level_metrics.csv",
        "grouped_credit_drift.csv",
        "grouped_credit_drift_winner.csv",
        "grouped_credit_drift_loser.csv",
        "correlation_results.json",
        "config.json",
        "credit_group_kl.pdf",
        "credit_vs_kl.pdf",
        "summary.md",
    )
    return all((output / name).is_file() for name in required)


def _merged_model_complete(output: Path) -> bool:
    config = output / "config.json"
    unsharded = output / "model.safetensors"
    index_path = output / "model.safetensors.index.json"
    if not config.is_file():
        return False
    if unsharded.is_file() and unsharded.stat().st_size > 1024:
        return True
    if not index_path.is_file():
        return False
    index = json.loads(index_path.read_text(encoding="utf-8"))
    shards = {output / name for name in index.get("weight_map", {}).values()}
    return bool(shards) and all(path.is_file() and path.stat().st_size > 1024 for path in shards)


def _same_run(existing: dict, current: dict) -> bool:
    keys = (
        "checkpoint_repo",
        "adapter_revision",
        "reference_model",
        "reference_revision",
        "dataset",
        "dataset_revision",
        "split",
        "mode",
        "max_examples",
        "bootstrap_iterations",
        "generation_prompts",
        "beta",
        "delta0",
        "temperature",
        "kappa",
        "min_tokens_per_chunk",
        "max_tokens_per_chunk",
        "max_length",
        "kl_token_batch_size",
        "generation_batch_size",
        "generation_max_model_len",
        "gpu_memory_utilization",
        "seed",
        "generation_seed",
        "run_uniform_control",
        "run_generation",
    )
    return all(existing.get(key) == current.get(key) for key in keys)


@app.function(
    image=image,
    gpu="A100",
    cpu=8.0,
    memory=65536,
    timeout=24 * 60 * 60,
    secrets=[hf_secret],
    volumes={CACHE_ROOT: cache_volume, RESULTS_ROOT: results_volume},
)
def run_reviewer_analysis(config: dict):
    import csv
    import os
    import shutil
    import subprocess
    import sys
    import tarfile
    from datetime import datetime, timezone

    import torch

    remote_repo = Path(REMOTE_REPO)
    cache_root = Path(CACHE_ROOT)
    results_root = Path(RESULTS_ROOT)

    if torch.cuda.device_count() != 1:
        raise RuntimeError(f"Expected exactly one Modal GPU, found {torch.cuda.device_count()}.")
    gpu_name = torch.cuda.get_device_name(0)
    if "A100" not in gpu_name.upper():
        raise RuntimeError(f"Expected an A100, got {gpu_name!r}.")
    print(f"[gpu] {gpu_name}; count=1", flush=True)

    mode = str(config["mode"])
    if mode not in {"smoke", "full"}:
        raise ValueError("mode must be 'smoke' or 'full'.")
    max_examples = int(config["max_examples"])
    bootstrap_iterations = int(config["bootstrap_iterations"])
    generation_prompts = int(config["generation_prompts"])
    if min(max_examples, bootstrap_iterations, generation_prompts) < 0:
        raise ValueError("Effective example, bootstrap, and prompt counts cannot be negative.")

    adapter_snapshot, adapter_revision = _resolved_snapshot(
        config["checkpoint_repo"], config["checkpoint_revision"]
    )
    adapter_path = _find_adapter(adapter_snapshot)
    adapter_config = json.loads((adapter_path / "adapter_config.json").read_text(encoding="utf-8"))
    recorded_base = str(adapter_config.get("base_model_name_or_path", ""))
    if not recorded_base:
        raise ValueError("The adapter does not record base_model_name_or_path.")
    if recorded_base.rstrip("/").lower() != config["reference_model"].rstrip("/").lower():
        raise ValueError(
            "Reference mismatch: adapter records "
            f"{recorded_base!r}, requested {config['reference_model']!r}."
        )
    base_snapshot, reference_revision = _resolved_snapshot(
        config["reference_model"], config["reference_revision"]
    )
    dataset_revision = _resolve_dataset_revision(config["dataset"], config["dataset_revision"])
    cache_volume.commit()

    run_slug = _slug(config["output_name"] or config["checkpoint_repo"].split("/")[-1])
    output_root = (results_root / run_slug / mode).resolve()
    if results_root.resolve() not in output_root.parents:
        raise ValueError(f"Unsafe output path: {output_root}")
    if config["force"] and output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    run_config = {
        **config,
        "adapter_revision": adapter_revision,
        "reference_revision": reference_revision,
        "dataset_revision": dataset_revision,
        "adapter_path": str(adapter_path),
        "base_snapshot": str(base_snapshot),
        "gpu_name": gpu_name,
        "gpu_count": 1,
        "resolved_output_dir": str(output_root),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "primary_credit_source": "post-hoc checkpoint chunk log-ratio",
        "uniform_control_meaning": "exact target shape used by this uniform checkpoint",
        "diversity_entropy_estimator": "renormalized top-20 log-probabilities",
    }
    config_path = output_root / "modal_run_config.json"
    if config_path.is_file() and not config["force"]:
        existing = json.loads(config_path.read_text(encoding="utf-8"))
        if not _same_run(existing, run_config):
            raise RuntimeError(
                f"Output {output_root} belongs to a different run. Choose --output-name "
                "or pass --force explicitly."
            )
    config_path.write_text(json.dumps(run_config, indent=2) + "\n", encoding="utf-8")
    results_volume.commit()

    common_credit = [
        sys.executable,
        remote_repo / "analyze_credit_drift.py",
        "--policy_model",
        base_snapshot,
        "--policy_adapter",
        adapter_path,
        "--policy_base_model",
        base_snapshot,
        "--reference_model",
        base_snapshot,
        "--tokenizer",
        base_snapshot,
        "--dataset",
        config["dataset"],
        "--dataset_revision",
        dataset_revision,
        "--split",
        config["split"],
        "--beta",
        config["beta"],
        "--delta0",
        config["delta0"],
        "--temperature",
        config["temperature"],
        "--kappa",
        config["kappa"],
        "--min_tokens_per_chunk",
        config["min_tokens_per_chunk"],
        "--max_tokens_per_chunk",
        config["max_tokens_per_chunk"],
        "--max_length",
        config["max_length"],
        "--bootstrap_iterations",
        bootstrap_iterations,
        "--confidence_level",
        0.95,
        "--dtype",
        "bfloat16",
        "--device_map",
        "cuda:0",
        "--attn_implementation",
        "sdpa",
        "--kl_device",
        "cuda:0",
        "--kl_token_batch_size",
        config["kl_token_batch_size"],
        "--seed",
        config["seed"],
    ]
    if max_examples > 0:
        common_credit.extend(("--max_examples", max_examples))

    credit_modes = ["logratio"]
    if config["run_uniform_control"]:
        credit_modes.append("uniform")
    for credit_mode in credit_modes:
        credit_output = output_root / f"credit_drift_{credit_mode}"
        if _credit_outputs_complete(credit_output) and not config["force"]:
            print(f"[resume] credit analysis already complete: {credit_output}")
            continue
        command = common_credit + [
            "--allocation_mode",
            credit_mode,
            "--output_dir",
            credit_output,
        ]
        _stream_command(command, output_root / "logs" / f"credit_drift_{credit_mode}.log")
        if not _credit_outputs_complete(credit_output):
            raise RuntimeError(f"Credit analysis did not produce all required files: {credit_output}")
        results_volume.commit()

    diversity_metrics = None
    if config["run_generation"]:
        prompts_path = output_root / "diversity" / "heldout_prompts.jsonl"
        indices_path = output_root / "diversity" / "credit_dataset_indices.json"
        generation_path = output_root / "diversity" / "diversity_generations.jsonl"
        metrics_path = output_root / "diversity" / "diversity_metrics.json"
        with (output_root / "credit_drift_logratio" / "chunk_level_metrics.csv").open(
            "r", encoding="utf-8", newline=""
        ) as handle:
            credit_indices = list(
                dict.fromkeys(int(row["dataset_index"]) for row in csv.DictReader(handle))
            )
        indices_path.parent.mkdir(parents=True, exist_ok=True)
        indices_path.write_text(json.dumps(credit_indices, indent=2) + "\n", encoding="utf-8")
        if not prompts_path.is_file() or config["force"]:
            command = [
                sys.executable,
                remote_repo / "diversity_metrics" / "prepare_prompts.py",
                "--dataset",
                config["dataset"],
                "--dataset_revision",
                dataset_revision,
                "--split",
                config["split"],
                "--out",
                prompts_path,
                "--max_prompts",
                generation_prompts,
                "--indices_file",
                indices_path,
                "--seed",
                config["seed"],
            ]
            _stream_command(command, output_root / "logs" / "prepare_prompts.log")
            results_volume.commit()

        merged_path = cache_root / "merged" / _slug(config["checkpoint_repo"]) / adapter_revision
        if not _merged_model_complete(merged_path):
            _stream_command(
                [
                    sys.executable,
                    remote_repo / "merge.py",
                    "--base_model",
                    base_snapshot,
                    "--adapter",
                    adapter_path,
                    "--output",
                    merged_path,
                    "--device",
                    "auto",
                    "--dtype",
                    "bfloat16",
                ],
                output_root / "logs" / "merge_adapter.log",
            )
            cache_volume.commit()

        if not metrics_path.is_file() or config["force"]:
            _stream_command(
                [
                    sys.executable,
                    remote_repo / "diversity_metrics" / "generation_vllm.py",
                    "--model",
                    merged_path,
                    "--prompts",
                    prompts_path,
                    "--out",
                    generation_path,
                    "--k",
                    5,
                    "--tensor_parallel_size",
                    1,
                    "--batch_size",
                    config["generation_batch_size"],
                    "--max_new_tokens",
                    128,
                    "--temperature",
                    1.0,
                    "--top_p",
                    0.95,
                    "--seed",
                    config["generation_seed"],
                    "--logprobs_k",
                    20,
                    "--dtype",
                    "bfloat16",
                    "--gpu_memory_utilization",
                    config["gpu_memory_utilization"],
                    "--max_model_len",
                    config["generation_max_model_len"],
                ],
                output_root / "logs" / "generation.log",
            )
            _stream_command(
                [
                    sys.executable,
                    remote_repo / "diversity_metrics" / "compute_diversity.py",
                    "--infile",
                    generation_path,
                    "--out",
                    metrics_path,
                ],
                output_root / "logs" / "compute_diversity.log",
            )
            results_volume.commit()
        diversity_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    archive_path = results_root / f"{run_slug}-{mode}.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        archive.add(output_root, arcname=f"{run_slug}/{mode}")
    results_volume.commit()

    primary_summary = (output_root / "credit_drift_logratio" / "summary.md").read_text(
        encoding="utf-8"
    )
    return {
        "output_dir": str(output_root),
        "archive": str(archive_path),
        "adapter_revision": adapter_revision,
        "reference_revision": reference_revision,
        "dataset_revision": dataset_revision,
        "gpu": gpu_name,
        "primary_summary": primary_summary,
        "diversity_metrics": diversity_metrics,
    }


@app.local_entrypoint()
def main(
    mode: str = "smoke",
    checkpoint_repo: str = CHECKPOINT_REPO,
    checkpoint_revision: str = "",
    reference_model: str = REFERENCE_MODEL,
    reference_revision: str = "",
    dataset: str = DATASET_REPO,
    dataset_revision: str = "",
    split: str = "test",
    output_name: str = "llama3-arc-bpo-uniform-lora-10k-bs64",
    max_examples: int = -1,
    bootstrap_iterations: int = -1,
    generation_prompts: int = -1,
    run_uniform_control: bool = True,
    run_generation: bool = True,
    beta: float = 0.1,
    delta0: float = 2.5,
    temperature: float = 2.0,
    kappa: float = 2.0,
    min_tokens_per_chunk: int = 4,
    max_tokens_per_chunk: int = 64,
    max_length: int = 2048,
    kl_token_batch_size: int = 8,
    generation_batch_size: int = 32,
    generation_max_model_len: int = 4096,
    gpu_memory_utilization: float = 0.90,
    seed: int = 42,
    generation_seed: int = 1234,
    force: bool = False,
):
    if mode == "smoke":
        effective_examples = 8 if max_examples < 0 else max_examples
        effective_bootstrap = 100 if bootstrap_iterations < 0 else bootstrap_iterations
        effective_prompts = 8 if generation_prompts < 0 else generation_prompts
    elif mode == "full":
        effective_examples = 0 if max_examples < 0 else max_examples
        effective_bootstrap = 2000 if bootstrap_iterations < 0 else bootstrap_iterations
        effective_prompts = 0 if generation_prompts < 0 else generation_prompts
    else:
        raise ValueError("--mode must be smoke or full.")

    config = {
        "mode": mode,
        "checkpoint_repo": checkpoint_repo,
        "checkpoint_revision": checkpoint_revision,
        "reference_model": reference_model,
        "reference_revision": reference_revision,
        "dataset": dataset,
        "dataset_revision": dataset_revision,
        "split": split,
        "output_name": output_name,
        "max_examples": effective_examples,
        "bootstrap_iterations": effective_bootstrap,
        "generation_prompts": effective_prompts,
        "run_uniform_control": run_uniform_control,
        "run_generation": run_generation,
        "beta": beta,
        "delta0": delta0,
        "temperature": temperature,
        "kappa": kappa,
        "min_tokens_per_chunk": min_tokens_per_chunk,
        "max_tokens_per_chunk": max_tokens_per_chunk,
        "max_length": max_length,
        "kl_token_batch_size": kl_token_batch_size,
        "generation_batch_size": generation_batch_size,
        "generation_max_model_len": generation_max_model_len,
        "gpu_memory_utilization": gpu_memory_utilization,
        "seed": seed,
        "generation_seed": generation_seed,
        "force": force,
    }
    # Submit this long-running job as a durable asynchronous invocation. A
    # synchronous remote() call is canceled shortly after the local Modal
    # client disconnects, even when the ephemeral App itself is detached.
    function_call = run_reviewer_analysis.spawn(config)
    result = function_call.get()
    print(json.dumps(result, indent=2, ensure_ascii=False))
