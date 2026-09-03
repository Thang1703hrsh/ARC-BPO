"""Evaluate the downloaded Mistral-7B uniform-allocation LoRA checkpoint on Modal.

The adapter is merged once with its immutable base revision.  Each benchmark is
then evaluated separately and committed to a Modal Volume, so a preempted run
can resume at the first unfinished benchmark.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

import modal


APP_NAME = "arc-bpo-eval-mistral-uniform-step157"
GPU_TYPE = "L4"
CPU_COUNT = 2.0
MEMORY_MIB = 16 * 1024
EVAL_BATCH_SIZE = "auto"
BASE_MODEL = "HuggingFaceH4/mistral-7b-sft-alpha"
BASE_REVISION = "7fd07f275ea812e71d5edc64c307b39511fb6616"
CHECKPOINT_LABEL = "mistral7b-uniform-step-157"
REMOTE_REPO = "/root/arc-bpo"
REMOTE_CHECKPOINT = "/checkpoint"
CACHE_ROOT = "/cache"
RESULTS_ROOT = "/results"
_HERE = Path(__file__).resolve().parent
_LOCAL_CHECKPOINT = (
    _HERE
    / "downloaded_checkpoints"
    / "mistral7b-uniform-step-157"
    / "step-157"
)

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


def _validate_local_checkpoint() -> None:
    adapter_config = _LOCAL_CHECKPOINT / "adapter" / "adapter_config.json"
    adapter_weights = _LOCAL_CHECKPOINT / "adapter" / "adapter_model.safetensors"
    if not adapter_config.is_file():
        raise FileNotFoundError(f"Missing LoRA config: {adapter_config}")
    if not adapter_weights.is_file() or adapter_weights.stat().st_size < 1024:
        raise FileNotFoundError(f"Missing or incomplete LoRA weights: {adapter_weights}")


if modal.is_local():
    # The Windows source path is needed only while Modal builds the image.
    # Inside the Linux container the same checkpoint is mounted at
    # REMOTE_CHECKPOINT, so validating _LOCAL_CHECKPOINT there would fail at
    # module-import time before the function can emit logs or write results.
    _validate_local_checkpoint()

cache_volume = modal.Volume.from_name("arc-bpo-hf-cache", create_if_missing=True)
results_volume = modal.Volume.from_name(
    "arc-bpo-mistral-allocation-eval-results",
    create_if_missing=True,
)

image = (
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
    .add_local_dir(str(_LOCAL_CHECKPOINT), remote_path=REMOTE_CHECKPOINT)
)

app = modal.App(APP_NAME)
hf_secret = modal.Secret.from_name("huggingface-secret")


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
    image=image,
    gpu=GPU_TYPE,
    cpu=CPU_COUNT,
    memory=MEMORY_MIB,
    timeout=24 * 60 * 60,
    max_containers=1,
    secrets=[hf_secret],
    volumes={CACHE_ROOT: cache_volume, RESULTS_ROOT: results_volume},
)
def evaluate_step157(force: bool = False) -> dict[str, Any]:
    import importlib.metadata
    import subprocess
    import sys

    import torch

    print("[stage] container_started", flush=True)
    if torch.cuda.device_count() != 1:
        raise RuntimeError(f"Expected exactly one GPU, got {torch.cuda.device_count()}.")
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory_gib = round(
        torch.cuda.get_device_properties(0).total_memory / (1024**3), 2
    )
    if gpu_memory_gib < 20:
        raise RuntimeError(
            f"Evaluation needs a GPU with at least 20 GiB VRAM; got {gpu_name} "
            f"({gpu_memory_gib} GiB)."
        )

    output_root = Path(RESULTS_ROOT) / "allocation_evaluations" / CHECKPOINT_LABEL
    final_path = output_root / "evaluation_metrics.json"
    if final_path.is_file() and not force:
        return _read_json(final_path)

    protocol = {
        "version": 1,
        "checkpoint": CHECKPOINT_LABEL,
        "checkpoint_update": 157,
        "checkpoint_examples": 157 * 64,
        "base_model": BASE_MODEL,
        "base_revision": BASE_REVISION,
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
        "resources": {
            "requested_gpu": GPU_TYPE,
            "actual_gpu": gpu_name,
            "gpu_memory_gib": gpu_memory_gib,
            "cpu": CPU_COUNT,
            "system_memory_mib": MEMORY_MIB,
        },
    }
    _write_json(output_root / "evaluation_protocol.json", protocol)
    results_volume.commit()
    print(f"[stage] protocol_saved output_root={output_root}", flush=True)

    adapter_path = Path(REMOTE_CHECKPOINT) / "adapter"
    if not (adapter_path / "adapter_model.safetensors").is_file():
        raise FileNotFoundError(f"Bundled adapter is incomplete: {adapter_path}")

    merged_model = Path("/tmp") / f"{CHECKPOINT_LABEL}-merged"
    merge_log = output_root / "logs" / "merge.log"
    if not (merged_model / "config.json").is_file():
        print("[stage] merge_started", flush=True)
        merged_model.mkdir(parents=True, exist_ok=True)
        merge_log.parent.mkdir(parents=True, exist_ok=True)
        merge_command = [
            sys.executable,
            f"{REMOTE_REPO}/merge.py",
            "--base_model",
            BASE_MODEL,
            "--base_revision",
            BASE_REVISION,
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
        with merge_log.open("w", encoding="utf-8") as log_file:
            subprocess.run(
                merge_command,
                check=True,
                cwd=REMOTE_REPO,
                stdout=log_file,
                stderr=subprocess.STDOUT,
            )

        # Some historical Mistral configs serialize head_dim as null. Older
        # vLLM releases fail with ``int * NoneType`` instead of deriving it.
        # Persist the mathematically equivalent explicit value in the merged
        # model so the checkpoint is portable across vLLM releases.
        merged_config_path = merged_model / "config.json"
        merged_config = _read_json(merged_config_path)
        if merged_config.get("head_dim") is None:
            hidden_size = int(merged_config["hidden_size"])
            num_heads = int(merged_config["num_attention_heads"])
            if hidden_size % num_heads != 0:
                raise ValueError(
                    "Cannot derive head_dim: hidden_size is not divisible by "
                    "num_attention_heads."
                )
            merged_config["head_dim"] = hidden_size // num_heads
            _write_json(merged_config_path, merged_config)
        results_volume.commit()
        print("[stage] merge_completed", flush=True)

    task_scores: dict[str, float] = {}
    raw_results: dict[str, str] = {}
    for task_name in TASKS:
        task = LM_EVAL_TASKS[task_name]
        task_root = output_root / "tasks" / task_name
        summary_path = task_root / "task_summary.json"
        if summary_path.is_file() and not force:
            summary = _read_json(summary_path)
            task_scores[task_name] = float(summary["score"])
            raw_results[task_name] = str(summary["raw_result"])
            continue

        print(f"[stage] eval_started task={task_name}", flush=True)

        raw_dir = task_root / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        log_path = output_root / "logs" / f"eval_{task_name}.log"
        command = [
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
        ]
        with log_path.open("w", encoding="utf-8") as log_file:
            subprocess.run(
                command,
                check=True,
                cwd=REMOTE_REPO,
                stdout=log_file,
                stderr=subprocess.STDOUT,
            )
        candidates = _result_candidates(raw_dir)
        if not candidates:
            raise FileNotFoundError(f"No lm-eval result JSON found below {raw_dir}.")
        raw_path = candidates[-1]
        score = extract_task_score(_read_json(raw_path), task_name) * 100.0
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
        print(f"[stage] eval_completed task={task_name} score={score:.4f}", flush=True)
        task_scores[task_name] = score
        raw_results[task_name] = str(raw_path)

    result = {
        "status": "complete",
        "checkpoint": CHECKPOINT_LABEL,
        "checkpoint_update": 157,
        "checkpoint_examples": 157 * 64,
        "scores": task_scores,
        "average": sum(task_scores.values()) / len(TASKS),
        "raw_results": raw_results,
        "protocol": str(output_root / "evaluation_protocol.json"),
        "output_root": str(output_root),
        "resources": protocol["resources"],
    }
    _write_json(final_path, result)
    results_volume.commit()
    print(f"[stage] evaluation_completed average={result['average']:.4f}", flush=True)
    return result


@app.local_entrypoint()
def main(background: bool = True, force: bool = False):
    if background:
        call = evaluate_step157.spawn(force)
        print(json.dumps({"status": "submitted", "call_id": call.object_id}, indent=2))
    else:
        result = evaluate_step157.remote(force)
        print(json.dumps(result, indent=2, ensure_ascii=False))
