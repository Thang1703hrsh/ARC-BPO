"""
Open LLM Leaderboard v1 evaluation with the 2xH100 vLLM setting from
script/eval/general/arc.sh.

Run ARC only:
  modal run --detach modal_eval_open_llm_leaderboard_h100x2.py::main --tasks arc_challenge

Print saved results:
  modal run modal_eval_open_llm_leaderboard_h100x2.py::results
"""

import modal


APP_NAME = "tbpo-open-llm-eval-h100x2"
DEFAULT_MODEL = "HuggingFaceH4/mistral-7b-sft-alpha"
DEFAULT_MODEL_LABEL = "mistral-7b-sft-alpha"
RUN_VERSION = "arcsh-vllm-h100x2-vllm-v0-engine-v4"

VOLUME_ROOT = "/vol/output/open_llm_eval"


app = modal.App(APP_NAME)
output_volume = modal.Volume.from_name("tbpo-output", create_if_missing=True)


image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git")
    .pip_install(
        "torch",
        "transformers",
        "accelerate",
        "datasets",
        "huggingface_hub",
        "safetensors",
        "lm_eval[vllm]",
        "vllm",
        "wandb",
    )
    .env(
        {
            "LM_EVAL_LOGLEVEL": "DEBUG",
            "VLLM_LOGLEVEL": "INFO",
            "TOKENIZERS_PARALLELISM": "false",
            "VLLM_USE_V1": "0",
            "WANDB_MODE": "disabled",
        }
    )
)


OPEN_LLM_TASKS = [
    ("arc_challenge", 25),
    ("hellaswag", 10),
    ("truthfulqa_mc2", 0),
    ("mmlu", 5),
    ("winogrande", 5),
    ("gsm8k", 5),
]


_SHARED = dict(
    image=image,
    gpu="H100:2",
    volumes={VOLUME_ROOT: output_volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
)


def _label_from_model(model_path: str) -> str:
    return model_path.strip("/").replace("/", "__")


def _vllm_model_args(
    model_local: str,
    tensor_parallel_size: int,
    dtype: str,
    gpu_memory_utilization: float,
    max_model_len: int,
) -> str:
    return ",".join(
        [
            f"pretrained={model_local}",
            f"tensor_parallel_size={tensor_parallel_size}",
            f"dtype={dtype}",
            f"gpu_memory_utilization={gpu_memory_utilization}",
            f"max_model_len={max_model_len}",
        ]
    )


@app.function(**_SHARED, timeout=60 * 60)
def download_model(model_path: str = DEFAULT_MODEL, model_label: str = DEFAULT_MODEL_LABEL):
    import os
    import subprocess

    model_local = f"{VOLUME_ROOT}/models/{model_label}"
    if os.path.isdir(model_local) and os.listdir(model_local):
        print(f"[SKIP] {model_local} already exists in volume.")
        return model_local

    print(f"[VERSION] {RUN_VERSION}")
    print(f"[DOWNLOAD] {model_path}")
    subprocess.run(
        ["hf", "download", model_path, "--local-dir", model_local],
        check=True,
    )
    output_volume.commit()
    return model_local


@app.function(**_SHARED, timeout=60 * 60 * 8)
def eval_open_llm_leaderboard_h100x2(
    model_path: str = DEFAULT_MODEL,
    model_label: str = DEFAULT_MODEL_LABEL,
    tasks: str = "all",
    tensor_parallel_size: int = 2,
    dtype: str = "auto",
    gpu_memory_utilization: float = 0.9,
    max_model_len: int = 4096,
    batch_size: str = "auto:4",
    log_samples: bool = True,
    wandb_enabled: bool = False,
):
    from collections import deque
    import os
    import subprocess

    model_local = f"{VOLUME_ROOT}/models/{model_label}"
    if not os.path.isdir(model_local) or not os.listdir(model_local):
        raise RuntimeError(
            f"Local model not found at {model_local}. Run download_model first."
        )

    if tasks == "all":
        selected_tasks = OPEN_LLM_TASKS
    else:
        wanted = {task.strip() for task in tasks.split(",") if task.strip()}
        selected_tasks = [(task, shots) for task, shots in OPEN_LLM_TASKS if task in wanted]
        if not selected_tasks:
            raise ValueError(f"No known Open LLM tasks selected from: {tasks}")

    results_dir = f"{VOLUME_ROOT}/results_h100x2/{model_label}"
    log_dir = f"{results_dir}/logs"
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    groups = {}
    for task, fewshot in selected_tasks:
        groups.setdefault(fewshot, []).append(task)

    print(f"[VERSION] {RUN_VERSION}")
    print(f"[MODEL] {model_path}")
    print(f"[LOCAL] {model_local}")
    print(f"[RESULTS] {results_dir}")

    env = os.environ.copy()
    env["LM_EVAL_LOGLEVEL"] = "DEBUG"
    env["VLLM_LOGLEVEL"] = "INFO"
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["VLLM_USE_V1"] = "0"
    if not wandb_enabled:
        env["WANDB_MODE"] = "disabled"
    else:
        env.pop("WANDB_MODE", None)

    preflight_cmd = [
        "python",
        "-c",
        (
            "import torch, transformers, lm_eval, vllm; "
            "from vllm import LLM; "
            "print('[PREFLIGHT] torch=' + torch.__version__); "
            "print('[PREFLIGHT] transformers=' + transformers.__version__); "
            "print('[PREFLIGHT] lm_eval=' + getattr(lm_eval, '__version__', 'unknown')); "
            "print('[PREFLIGHT] vllm=' + vllm.__version__); "
            "print('[PREFLIGHT] vllm.LLM import OK')"
        ),
    ]
    subprocess.run(preflight_cmd, check=True, env=env)

    for fewshot, task_list in sorted(groups.items()):
        tasks_str = ",".join(task_list)
        group_dir = f"{results_dir}/fewshot{fewshot}"
        os.makedirs(group_dir, exist_ok=True)

        model_args = _vllm_model_args(
            model_local=model_local,
            tensor_parallel_size=tensor_parallel_size,
            dtype=dtype,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
        )
        cmd = [
            "python",
            "-m",
            "lm_eval",
            "--model",
            "vllm",
            "--model_args",
            model_args,
            "--tasks",
            tasks_str,
            "--num_fewshot",
            str(fewshot),
            "--batch_size",
            batch_size,
            "--output_path",
            group_dir,
        ]
        if log_samples:
            cmd.append("--log_samples")
        if wandb_enabled:
            cmd.extend(
                [
                    "--wandb_args",
                    (
                        "entity=Token_BPO,project=Token_BPO,"
                        f"name={model_label}_{tasks_str},job_type=eval"
                    ),
                ]
            )

        log_path = f"{log_dir}/fewshot{fewshot}_{tasks_str.replace(',', '_')}.log"
        print(f"\n[EVAL] {tasks_str} ({fewshot}-shot)")
        print(" ".join(cmd))
        print(f"[LOG] {log_path}")
        tail = deque(maxlen=80)
        with open(log_path, "w", encoding="utf-8") as log_file:
            proc = subprocess.Popen(
                cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                print(line, end="")
                log_file.write(line)
                tail.append(line.rstrip("\n"))
            return_code = proc.wait()

        output_volume.commit()
        if return_code != 0:
            print(f"\n[ERROR] lm_eval failed with exit code {return_code}.")
            print(f"[ERROR] Last {len(tail)} log lines from {log_path}:")
            for line in tail:
                print(line)
            raise subprocess.CalledProcessError(return_code, cmd)

    print("\n[DONE] Selected 2xH100 leaderboard benchmarks complete.")
    return f"Results saved to {results_dir}"


@app.function(image=image, volumes={VOLUME_ROOT: output_volume}, timeout=60 * 5)
def _read_results(model_label: str = DEFAULT_MODEL_LABEL):
    import glob
    import json
    import math

    output_volume.reload()
    results_dir = f"{VOLUME_ROOT}/results_h100x2/{model_label}"

    metric_keys = {
        "arc_challenge": ["acc_norm,none", "acc,none"],
        "hellaswag": ["acc_norm,none", "acc,none"],
        "truthfulqa_mc2": ["acc,none"],
        "mmlu": ["acc,none"],
        "winogrande": ["acc,none"],
        "gsm8k": ["exact_match,flexible-extract", "acc,none", "exact_match,none"],
    }

    def fmt(value):
        return f"{value * 100:.2f}" if value is not None and not math.isnan(value) else "N/A"

    all_results = {}
    for result_file in glob.glob(f"{results_dir}/**/results*.json", recursive=True):
        with open(result_file, encoding="utf-8") as f:
            data = json.load(f)
        for task, metrics in data.get("results", {}).items():
            if task in metric_keys and task not in all_results:
                for key in metric_keys[task]:
                    if key in metrics:
                        all_results[task] = metrics[key]
                        break

    print(f"\n{'=' * 45}")
    print(f" Open LLM Leaderboard H100x2 - {model_label}")
    print(f"{'=' * 45}")
    scores = []
    for task, fewshot in OPEN_LLM_TASKS:
        value = all_results.get(task)
        score = value * 100 if value is not None else float("nan")
        if not math.isnan(score):
            scores.append(score)
        print(f"  {task + f' ({fewshot}-shot)':<35} {fmt(value):>6}")
    if scores:
        print(f"  {'-' * 41}")
        print(f"  {'Average':<35} {sum(scores) / len(scores):>6.2f}")
    print(f"{'=' * 45}")


@app.local_entrypoint()
def main(
    model_path: str = DEFAULT_MODEL,
    model_label: str = DEFAULT_MODEL_LABEL,
    tasks: str = "all",
    tensor_parallel_size: int = 2,
    dtype: str = "auto",
    gpu_memory_utilization: float = 0.9,
    max_model_len: int = 4096,
    batch_size: str = "auto:4",
    log_samples: bool = True,
    wandb_enabled: bool = False,
):
    """Download model, then run selected benchmarks on 2xH100 detached."""
    if model_label == DEFAULT_MODEL_LABEL and model_path != DEFAULT_MODEL:
        model_label = _label_from_model(model_path)

    download_model.remote(model_path=model_path, model_label=model_label)
    eval_open_llm_leaderboard_h100x2.spawn(
        model_path=model_path,
        model_label=model_label,
        tasks=tasks,
        tensor_parallel_size=tensor_parallel_size,
        dtype=dtype,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        batch_size=batch_size,
        log_samples=log_samples,
        wandb_enabled=wandb_enabled,
    )
    print("[LAUNCHED] 2xH100 leaderboard eval running on Modal.")
    print(f"  App:     {APP_NAME}")
    print(f"  Version: {RUN_VERSION}")
    print(
        "  Results: modal run modal_eval_open_llm_leaderboard_h100x2.py::results "
        f"--model-label {model_label}"
    )


@app.local_entrypoint()
def results(model_label: str = DEFAULT_MODEL_LABEL):
    """Print saved 2xH100 leaderboard results table from the Modal volume."""
    _read_results.remote(model_label=model_label)
