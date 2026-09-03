"""Evaluate one Hugging Face ARC-BPO sensitivity checkpoint per detached H100 call."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

import modal


APP_NAME = "arc-bpo-eval-hf-sensitivity-h100"
HF_REPO_ID = "ducthang1703/llama3-arc-bpo-sensitivity-10k-bs64-2xa100-ga16"
CHECKPOINTS = (
    "sens_T_0.5_clean_seed0",
    "sens_T_1_clean_seed0",
    "sens_T_4_clean_seed0",
    "sens_kappa_1.5_clean_seed0",
    "sens_kappa_1_clean_seed0",
    "sens_kappa_3_clean_seed0",
)
GPU_TYPE = "H100"
CPU_COUNT = 4.0
MEMORY_MIB = 16 * 1024
EVAL_BATCH_SIZE = "auto"
BASE_MODEL_FALLBACK = "RLHFlow/LLaMA3-SFT-v2"
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
    "arc-bpo-hyperparameter-sensitivity-results",
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
        "pyyaml>=6.0",
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


def _get_hf_token() -> str | None:
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
    return None


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
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
        f"No approved metric for {task_name}; expected {task['metrics']}, "
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


def _run_logged(command: list[str], log_path: Path) -> None:
    import subprocess

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        try:
            subprocess.run(
                command,
                check=True,
                cwd=REMOTE_REPO,
                stdout=log_file,
                stderr=subprocess.STDOUT,
            )
        except subprocess.CalledProcessError:
            results_volume.commit()
            raise


@app.function(
    image=preflight_image,
    cpu=0.25,
    memory=512,
    timeout=5 * 60,
)
def validate_checkpoint(checkpoint_name: str) -> dict[str, Any]:
    from huggingface_hub import HfApi

    if checkpoint_name not in CHECKPOINTS:
        raise ValueError(f"Unsupported checkpoint {checkpoint_name!r}; choose from {CHECKPOINTS}.")
    token = _get_hf_token()
    api = HfApi(token=token)
    info = api.model_info(HF_REPO_ID, files_metadata=False)
    api.model_info(BASE_MODEL_FALLBACK, files_metadata=False)
    files = sorted(sibling.rfilename for sibling in info.siblings)
    root = f"checkpoints/{checkpoint_name}"
    required = (
        f"{root}/adapter/adapter_config.json",
        f"{root}/resolved_config.yaml",
        f"{root}/checkpoint_metadata.json",
    )
    missing = [path for path in required if path not in files]
    if missing:
        raise FileNotFoundError(f"{checkpoint_name} is missing required files: {missing}")
    weight_prefix = f"{root}/adapter/adapter_model"
    weights = [
        path
        for path in files
        if path.startswith(weight_prefix)
        and (path.endswith(".safetensors") or path.endswith(".bin"))
    ]
    if not weights:
        raise FileNotFoundError(f"{checkpoint_name} has no LoRA adapter weights.")
    return {
        "repo_id": HF_REPO_ID,
        "repo_revision": str(info.sha),
        "checkpoint_name": checkpoint_name,
        "checkpoint_root": root,
        "weights": weights,
    }


@app.function(
    image=eval_image,
    gpu=GPU_TYPE,
    cpu=CPU_COUNT,
    memory=MEMORY_MIB,
    timeout=24 * 60 * 60,
    max_containers=3,
    volumes={CACHE_ROOT: cache_volume, RESULTS_ROOT: results_volume},
)
def evaluate_checkpoint(checkpoint: dict[str, Any], force: bool = False) -> dict[str, Any]:
    import importlib.metadata
    import os
    import shutil
    import sys

    import torch
    import yaml
    from huggingface_hub import snapshot_download

    checkpoint_name = str(checkpoint["checkpoint_name"])
    print(f"[stage] container_started checkpoint={checkpoint_name}", flush=True)
    if torch.cuda.device_count() != 1:
        raise RuntimeError(f"Expected one GPU, got {torch.cuda.device_count()}.")
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory_gib = round(
        torch.cuda.get_device_properties(0).total_memory / (1024**3), 2
    )
    if "H100" not in gpu_name.upper() or gpu_memory_gib < 70:
        raise RuntimeError(
            f"Expected one H100 with at least 70 GiB; got {gpu_name} ({gpu_memory_gib} GiB)."
        )

    output_root = (
        Path(RESULTS_ROOT)
        / "hf_sensitivity_evaluations"
        / HF_REPO_ID.split("/", 1)[-1]
        / checkpoint_name
    )
    final_path = output_root / "evaluation_metrics.json"
    if final_path.is_file() and not force:
        print(f"[stage] result_exists checkpoint={checkpoint_name}", flush=True)
        return _read_json(final_path)

    token = _get_hf_token()
    if token:
        os.environ["HF_TOKEN"] = token
    checkpoint_root = str(checkpoint["checkpoint_root"])
    snapshot_root = Path("/tmp/hf-sensitivity") / checkpoint_name
    print(f"[stage] download_started checkpoint={checkpoint_name}", flush=True)
    snapshot_download(
        repo_id=HF_REPO_ID,
        revision=str(checkpoint["repo_revision"]),
        local_dir=snapshot_root,
        allow_patterns=[f"{checkpoint_root}/**"],
        token=token,
    )
    local_checkpoint = snapshot_root / checkpoint_root
    adapter_path = local_checkpoint / "adapter"
    adapter_config = _read_json(adapter_path / "adapter_config.json")
    metadata = _read_json(local_checkpoint / "checkpoint_metadata.json")
    resolved_config = yaml.safe_load(
        (local_checkpoint / "resolved_config.yaml").read_text(encoding="utf-8")
    )
    model_config = resolved_config.get("model", {}) if isinstance(resolved_config, dict) else {}
    base_model = str(
        adapter_config.get("base_model_name_or_path")
        or model_config.get("name_or_path")
        or BASE_MODEL_FALLBACK
    )
    base_revision = adapter_config.get("revision") or model_config.get("revision")
    if base_revision is not None:
        base_revision = str(base_revision)

    resources = {
        "requested_gpu": GPU_TYPE,
        "actual_gpu": gpu_name,
        "gpu_memory_gib": gpu_memory_gib,
        "cpu": CPU_COUNT,
        "system_memory_mib": MEMORY_MIB,
    }
    protocol = {
        "version": 1,
        "repo_id": HF_REPO_ID,
        "repo_revision": str(checkpoint["repo_revision"]),
        "checkpoint_root": checkpoint_root,
        "checkpoint_metadata": metadata,
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
        "resources": resources,
    }
    _write_json(output_root / "evaluation_protocol.json", protocol)
    results_volume.commit()

    merged_model = Path("/tmp/merged-sensitivity") / checkpoint_name
    if merged_model.exists():
        shutil.rmtree(merged_model)
    merge_command = [
        sys.executable,
        f"{REMOTE_REPO}/merge.py",
        "--base_model",
        base_model,
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
    ]
    if base_revision:
        merge_command.extend(("--base_revision", base_revision))
    print(f"[stage] merge_started checkpoint={checkpoint_name}", flush=True)
    _run_logged(merge_command, output_root / "logs" / "merge.log")
    print(f"[stage] merge_completed checkpoint={checkpoint_name}", flush=True)
    results_volume.commit()

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

        print(
            f"[stage] eval_started checkpoint={checkpoint_name} task={task_name}",
            flush=True,
        )
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
            f"[stage] eval_completed checkpoint={checkpoint_name} "
            f"task={task_name} score={score:.4f}",
            flush=True,
        )

    result = {
        "status": "complete",
        "repo_id": HF_REPO_ID,
        "repo_revision": str(checkpoint["repo_revision"]),
        "checkpoint_name": checkpoint_name,
        "checkpoint_metadata": metadata,
        "scores": scores,
        "average": sum(scores.values()) / len(TASKS),
        "raw_results": raw_results,
        "protocol": str(output_root / "evaluation_protocol.json"),
        "output_root": str(output_root),
        "resources": resources,
    }
    _write_json(final_path, result)
    results_volume.commit()
    if merged_model.exists():
        shutil.rmtree(merged_model)
    print(
        f"[stage] checkpoint_completed checkpoint={checkpoint_name} "
        f"average={result['average']:.4f}",
        flush=True,
    )
    return result


@app.local_entrypoint()
def main(checkpoint: str, background: bool = True, force: bool = False):
    validated = validate_checkpoint.remote(checkpoint)
    if background:
        call = evaluate_checkpoint.spawn(validated, force)
        print(
            json.dumps(
                {
                    "status": "submitted",
                    "checkpoint": checkpoint,
                    "gpu": GPU_TYPE,
                    "call_id": call.object_id,
                },
                indent=2,
            )
        )
    else:
        result = evaluate_checkpoint.remote(validated, force)
        print(json.dumps(result, indent=2, ensure_ascii=False))
