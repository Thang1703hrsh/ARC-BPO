"""Evaluate two private ARC-BPO LoRA repositories in parallel on two H100s."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

import modal


APP_NAME = "arc-bpo-eval-hf-allocation-h100"
HF_SECRET_NAME = "huggingface-secret-2"
GPU_TYPE = "H100"
CPU_COUNT = 4.0
MEMORY_MIB = 16 * 1024
EVAL_BATCH_SIZE = "auto"
BASE_MODEL_FALLBACK = "HuggingFaceH4/mistral-7b-sft-alpha"
BASE_REVISION_FALLBACK = "7fd07f275ea812e71d5edc64c307b39511fb6616"
HF_REPOS = (
    "ducthang1703/mistral7b-arc-bpo-advantage-sba-no-winsor-lora-16k-bs64-seed0",
    "ducthang1703/mistral7b-arc-bpo-uniform-quadratic-lora-smoke64-bs64-seed0",
)
REMOTE_REPO = "/root/arc-bpo"
CACHE_ROOT = "/cache"
RESULTS_ROOT = "/results"
_HERE = Path(__file__).resolve().parent

TASKS = ("hellaswag", "arc", "mmlu", "truthfulqa", "winogrande", "gsm8k")
LM_EVAL_TASKS = {
    "hellaswag": {
        "task": "hellaswag",
        "fewshot": 10,
        "metrics": ("acc_norm,none", "acc,none"),
    },
    "arc": {
        "task": "arc_challenge",
        "fewshot": 25,
        "metrics": ("acc_norm,none", "acc,none"),
    },
    "mmlu": {"task": "mmlu", "fewshot": 5, "metrics": ("acc,none",)},
    "truthfulqa": {
        "task": "truthfulqa_mc2",
        "fewshot": 0,
        "metrics": ("acc,none",),
    },
    "winogrande": {
        "task": "winogrande",
        "fewshot": 5,
        "metrics": ("acc,none",),
    },
    "gsm8k": {
        "task": "gsm8k",
        "fewshot": 5,
        "metrics": (
            "exact_match,strict-match",
            "exact_match,flexible-extract",
            "exact_match,none",
        ),
    },
}

cache_volume = modal.Volume.from_name("arc-bpo-hf-cache", create_if_missing=True)
results_volume = modal.Volume.from_name(
    "arc-bpo-mistral-allocation-eval-results",
    create_if_missing=True,
)

preflight_image = modal.Image.debian_slim(python_version="3.11").uv_pip_install(
    "huggingface_hub[hf_xet]>=0.31.0,<1"
)

eval_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-devel-ubuntu22.04",
        add_python="3.11",
    )
    .uv_pip_install(
        "torch==2.6.0",
        "vllm==0.8.5",
        "transformers==4.52.4",
        "datasets==3.5.0",
        "accelerate==1.7.0",
        "peft==0.15.2",
        "lm-eval[sentencepiece,vllm]==0.4.9.2",
        "sentencepiece>=0.2.0",
        "protobuf>=4.25.0",
        "huggingface_hub[hf_xet]>=0.31.0,<1",
        "safetensors>=0.5.3",
    )
    .env(
        {
            "HF_HOME": f"{CACHE_ROOT}/huggingface",
            "HF_XET_HIGH_PERFORMANCE": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "WANDB_MODE": "disabled",
        }
    )
    .add_local_file(str(_HERE / "merge.py"), remote_path=f"{REMOTE_REPO}/merge.py")
)

app = modal.App(APP_NAME)
hf_secret = modal.Secret.from_name(HF_SECRET_NAME)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _get_hf_token() -> str:
    import os

    token_names = (
        "HF_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
        "HUGGINGFACE_TOKEN",
        "HUGGINGFACEHUB_API_TOKEN",
        "hf2",
    )
    for token_name in token_names:
        token = os.environ.get(token_name)
        if token:
            return token
    matching_keys = sorted(
        key for key in os.environ if "HF" in key.upper() or "HUGGING" in key.upper()
    )
    raise RuntimeError(
        "huggingface-secret does not expose a supported token key. "
        f"Expected one of {token_names}; matching environment keys: {matching_keys}."
    )


def _metric_block(payload: Mapping[str, Any], harness_task: str) -> Mapping[str, Any]:
    for section in ("results", "groups"):
        values = payload.get(section, {})
        if harness_task in values:
            return values[harness_task]
    raise KeyError(f"Harness result has no task/group entry for {harness_task!r}.")


def _extract_score(payload: Mapping[str, Any], task_name: str) -> float:
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


def _result_candidates(output_path: Path) -> list[Path]:
    return sorted(
        (
            path
            for path in output_path.rglob("*.json")
            if path.name == "results.json" or path.name.startswith("results_")
        ),
        key=lambda path: path.stat().st_mtime,
    )


@app.function(
    image=preflight_image,
    cpu=0.25,
    memory=512,
    timeout=5 * 60,
    secrets=[hf_secret],
)
def validate_hf_repositories() -> list[dict[str, Any]]:
    from huggingface_hub import HfApi

    token = _get_hf_token()
    api = HfApi(token=token)
    validated: list[dict[str, Any]] = []
    for repo_id in HF_REPOS:
        info = api.model_info(repo_id=repo_id, files_metadata=False)
        files = sorted(sibling.rfilename for sibling in info.siblings)
        if "adapter_config.json" in files:
            adapter_subdir = ""
        elif "adapter/adapter_config.json" in files:
            adapter_subdir = "adapter"
        else:
            raise FileNotFoundError(
                f"{repo_id} contains no adapter_config.json at root or adapter/."
            )
        if not any(
            name.startswith(f"{adapter_subdir}/adapter_model".lstrip("/"))
            and (name.endswith(".safetensors") or name.endswith(".bin"))
            for name in files
        ):
            raise FileNotFoundError(f"{repo_id} contains no LoRA adapter weights.")
        validated.append(
            {
                "repo_id": repo_id,
                "revision": str(info.sha),
                "adapter_subdir": adapter_subdir,
                "files": files,
            }
        )
    return validated


def _run_logged(command: list[str], log_path: Path, cwd: str) -> None:
    import subprocess

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        try:
            subprocess.run(
                command,
                check=True,
                cwd=cwd,
                stdout=log_file,
                stderr=subprocess.STDOUT,
            )
        except subprocess.CalledProcessError:
            results_volume.commit()
            raise


@app.function(
    image=eval_image,
    gpu=GPU_TYPE,
    cpu=CPU_COUNT,
    memory=MEMORY_MIB,
    timeout=24 * 60 * 60,
    max_containers=2,
    secrets=[hf_secret],
    volumes={CACHE_ROOT: cache_volume, RESULTS_ROOT: results_volume},
)
def evaluate_hf_repositories(
    repositories: list[dict[str, Any]], force: bool = False
) -> dict[str, Any]:
    import importlib.metadata
    import os
    import shutil
    import sys

    import torch
    from huggingface_hub import snapshot_download

    print("[stage] container_started", flush=True)
    if torch.cuda.device_count() != 1:
        raise RuntimeError(f"Expected one GPU, got {torch.cuda.device_count()}.")
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory_gib = round(
        torch.cuda.get_device_properties(0).total_memory / (1024**3), 2
    )
    if "H100" not in gpu_name.upper() or gpu_memory_gib < 70:
        raise RuntimeError(
            f"Expected one H100 with at least 70 GiB; got {gpu_name} "
            f"({gpu_memory_gib} GiB)."
        )

    token = _get_hf_token()
    os.environ["HF_TOKEN"] = token
    aggregate: dict[str, Any] = {
        "status": "running",
        "resources": {
            "requested_gpu": GPU_TYPE,
            "actual_gpu": gpu_name,
            "gpu_memory_gib": gpu_memory_gib,
            "cpu": CPU_COUNT,
            "system_memory_mib": MEMORY_MIB,
        },
        "models": {},
    }
    adapter_root = Path("/tmp/arc-bpo-hf-adapters")
    merge_root = Path("/tmp/arc-bpo-merged")
    adapter_root.mkdir(parents=True, exist_ok=True)
    merge_root.mkdir(parents=True, exist_ok=True)

    for model_index, repository in enumerate(repositories, start=1):
        repo_id = str(repository["repo_id"])
        revision = str(repository["revision"])
        repo_name = repo_id.split("/", 1)[-1]
        output_root = Path(RESULTS_ROOT) / "hf_evaluations" / repo_name
        final_path = output_root / "evaluation_metrics.json"
        if final_path.is_file() and not force:
            result = _read_json(final_path)
            aggregate["models"][repo_id] = result
            print(f"[stage] result_exists repo={repo_id}", flush=True)
            continue

        print(
            f"[stage] download_started model={model_index}/{len(repositories)} "
            f"repo={repo_id}",
            flush=True,
        )
        local_snapshot = adapter_root / repo_name
        snapshot_download(
            repo_id=repo_id,
            revision=revision,
            local_dir=local_snapshot,
            token=token,
        )
        adapter_subdir = str(repository.get("adapter_subdir") or "")
        adapter_path = local_snapshot / adapter_subdir if adapter_subdir else local_snapshot
        adapter_config = _read_json(adapter_path / "adapter_config.json")
        run_config_path = local_snapshot / "modal_run_config.json"
        run_config = _read_json(run_config_path) if run_config_path.is_file() else {}
        base_model = str(
            adapter_config.get("base_model_name_or_path") or BASE_MODEL_FALLBACK
        )
        base_revision = str(
            run_config.get("base_revision") or BASE_REVISION_FALLBACK
        )
        protocol = {
            "version": 1,
            "repo_id": repo_id,
            "repo_revision": revision,
            "variant": run_config.get("variant"),
            "n_examples": run_config.get("n_examples"),
            "base_model": base_model,
            "base_revision": base_revision,
            "harness": "lm-evaluation-harness",
            "harness_version": importlib.metadata.version("lm_eval"),
            "model_backend": "vllm",
            "tasks": LM_EVAL_TASKS,
            "batch_size": EVAL_BATCH_SIZE,
            "dtype": "bfloat16",
            "tensor_parallel_size": 1,
            "gpu_memory_utilization": 0.90,
            "max_model_len": 4096,
            "evaluation_seed": "0,1234,1234,1234",
            "apply_chat_template": False,
            "resources": aggregate["resources"],
        }
        _write_json(output_root / "evaluation_protocol.json", protocol)
        results_volume.commit()

        merged_model = merge_root / repo_name
        if merged_model.exists():
            if merged_model.parent != merge_root:
                raise RuntimeError(f"Unsafe temporary merge path: {merged_model}")
            shutil.rmtree(merged_model)
        print(f"[stage] merge_started repo={repo_id}", flush=True)
        _run_logged(
            [
                sys.executable,
                f"{REMOTE_REPO}/merge.py",
                "--base_model",
                base_model,
                "--base_revision",
                base_revision,
                "--adapter",
                str(adapter_path),
                "--output",
                str(merged_model),
                "--device",
                "cuda",
                "--dtype",
                "bfloat16",
                "--max_shard_size",
                "5GB",
            ],
            output_root / "logs" / "merge.log",
            REMOTE_REPO,
        )
        merged_config_path = merged_model / "config.json"
        merged_config = _read_json(merged_config_path)
        if merged_config.get("head_dim") is None:
            hidden_size = int(merged_config["hidden_size"])
            num_heads = int(merged_config["num_attention_heads"])
            if hidden_size % num_heads:
                raise ValueError("Cannot derive an integral attention head_dim.")
            merged_config["head_dim"] = hidden_size // num_heads
            _write_json(merged_config_path, merged_config)
        results_volume.commit()
        print(f"[stage] merge_completed repo={repo_id}", flush=True)

        scores: dict[str, float] = {}
        raw_results: dict[str, str] = {}
        for task_name in TASKS:
            task = LM_EVAL_TASKS[task_name]
            task_root = output_root / "tasks" / task_name
            summary_path = task_root / "task_summary.json"
            if summary_path.is_file() and not force:
                summary = _read_json(summary_path)
                scores[task_name] = float(summary["score"])
                raw_results[task_name] = str(summary["raw_result"])
                continue

            print(f"[stage] eval_started repo={repo_id} task={task_name}", flush=True)
            raw_dir = task_root / "raw"
            raw_dir.mkdir(parents=True, exist_ok=True)
            log_path = output_root / "logs" / f"eval_{task_name}.log"
            _run_logged(
                [
                    "lm_eval",
                    "--model",
                    "vllm",
                    "--model_args",
                    ",".join(
                        (
                            f"pretrained={merged_model}",
                            "tensor_parallel_size=1",
                            "dtype=bfloat16",
                            "gpu_memory_utilization=0.90",
                            "max_model_len=4096",
                        )
                    ),
                    "--tasks",
                    str(task["task"]),
                    "--num_fewshot",
                    str(task["fewshot"]),
                    "--batch_size",
                    EVAL_BATCH_SIZE,
                    "--seed",
                    "0,1234,1234,1234",
                    "--output_path",
                    str(raw_dir),
                ],
                log_path,
                REMOTE_REPO,
            )
            candidates = _result_candidates(raw_dir)
            if not candidates:
                raise FileNotFoundError(f"No lm-eval JSON found below {raw_dir}.")
            raw_path = candidates[-1]
            score = _extract_score(_read_json(raw_path), task_name) * 100.0
            summary = {
                "task": task_name,
                "harness_task": task["task"],
                "fewshot": task["fewshot"],
                "score": score,
                "raw_result": str(raw_path),
                "log": str(log_path),
            }
            _write_json(summary_path, summary)
            results_volume.commit()
            scores[task_name] = score
            raw_results[task_name] = str(raw_path)
            print(
                f"[stage] eval_completed repo={repo_id} task={task_name} "
                f"score={score:.4f}",
                flush=True,
            )

        result = {
            "status": "complete",
            "repo_id": repo_id,
            "repo_revision": revision,
            "variant": run_config.get("variant"),
            "n_examples": run_config.get("n_examples"),
            "scores": scores,
            "average": sum(scores.values()) / len(TASKS),
            "raw_results": raw_results,
            "protocol": str(output_root / "evaluation_protocol.json"),
            "output_root": str(output_root),
            "resources": aggregate["resources"],
        }
        _write_json(final_path, result)
        results_volume.commit()
        aggregate["models"][repo_id] = result
        print(
            f"[stage] model_completed repo={repo_id} average={result['average']:.4f}",
            flush=True,
        )
        if merged_model.parent != merge_root:
            raise RuntimeError(f"Unsafe temporary cleanup path: {merged_model}")
        shutil.rmtree(merged_model)

    aggregate["status"] = "complete"
    repo_names = "__".join(
        str(repository["repo_id"]).split("/", 1)[-1] for repository in repositories
    )
    summary_path = (
        Path(RESULTS_ROOT) / "hf_evaluations" / f"run_summary_{repo_names}.json"
    )
    _write_json(summary_path, aggregate)
    results_volume.commit()
    print("[stage] all_models_completed", flush=True)
    return aggregate


@app.local_entrypoint()
def main(background: bool = True, force: bool = False):
    repositories = validate_hf_repositories.remote()
    if background:
        calls = {
            str(repository["repo_id"]): evaluate_hf_repositories.spawn(
                [repository], force
            ).object_id
            for repository in repositories
        }
        print(
            json.dumps(
                {
                    "status": "submitted",
                    "execution": "parallel",
                    "gpu_per_repository": GPU_TYPE,
                    "calls": calls,
                },
                indent=2,
            )
        )
    else:
        results = list(
            evaluate_hf_repositories.map(
                ([repository] for repository in repositories),
                kwargs={"force": force},
            )
        )
        print(json.dumps(results, indent=2, ensure_ascii=False))
