"""
Open LLM Leaderboard v1 evaluation for the TBPO/ARC-BPO experiments.

Default model:
  HuggingFaceH4/mistral-7b-sft-alpha

Run selected ARC + MMLU eval:
  modal run --detach modal_eval_open_llm_leaderboard.py::main --tasks arc_challenge,mmlu

Print saved results:
  modal run modal_eval_open_llm_leaderboard.py::results
"""

import modal


APP_NAME = "tbpo-open-llm-eval"
DEFAULT_MODEL = "HuggingFaceH4/mistral-7b-sft-alpha"
DEFAULT_MODEL_LABEL = "mistral-7b-sft-alpha"
RUN_VERSION = "baseline-style-vllm-a100-deps-v10"

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
    gpu="A100-80GB",
    volumes={VOLUME_ROOT: output_volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
)


def _label_from_model(model_path: str) -> str:
    return model_path.strip("/").replace("/", "__")


def _vllm_model_args(
    model_local: str,
    dtype: str,
    gpu_memory_utilization: float,
    max_model_len: int,
) -> str:
    return ",".join(
        [
            f"pretrained={model_local}",
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
def eval_open_llm_leaderboard(
    model_path: str = DEFAULT_MODEL,
    model_label: str = DEFAULT_MODEL_LABEL,
    tasks: str = "all",
    dtype: str = "float16",
    gpu_memory_utilization: float = 0.85,
    max_model_len: int = 4096,
    batch_size: str = "auto",
):
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

    results_dir = f"{VOLUME_ROOT}/results/{model_label}"
    os.makedirs(results_dir, exist_ok=True)

    groups = {}
    for task, fewshot in selected_tasks:
        groups.setdefault(fewshot, []).append(task)

    print(f"[VERSION] {RUN_VERSION}")
    print(f"[MODEL] {model_path}")
    print(f"[LOCAL] {model_local}")
    print(f"[RESULTS] {results_dir}")

    for fewshot, task_list in sorted(groups.items()):
        tasks_str = ",".join(task_list)
        group_dir = f"{results_dir}/fewshot{fewshot}"
        os.makedirs(group_dir, exist_ok=True)

        model_args = _vllm_model_args(
            model_local=model_local,
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

        print(f"\n[EVAL] {tasks_str} ({fewshot}-shot)")
        print(" ".join(cmd))
        subprocess.run(cmd, check=True)
        output_volume.commit()

    print("\n[DONE] Selected leaderboard benchmarks complete.")
    return f"Results saved to {results_dir}"


@app.function(image=image, volumes={VOLUME_ROOT: output_volume}, timeout=60 * 5)
def _read_results(model_label: str = DEFAULT_MODEL_LABEL):
    import glob
    import json
    import math

    output_volume.reload()
    results_dir = f"{VOLUME_ROOT}/results/{model_label}"

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
    print(f" Open LLM Leaderboard - {model_label}")
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
    dtype: str = "float16",
    gpu_memory_utilization: float = 0.85,
    max_model_len: int = 4096,
    batch_size: str = "auto",
):
    """Download model, then run selected leaderboard benchmarks detached."""
    if model_label == DEFAULT_MODEL_LABEL and model_path != DEFAULT_MODEL:
        model_label = _label_from_model(model_path)

    download_model.remote(model_path=model_path, model_label=model_label)
    eval_open_llm_leaderboard.spawn(
        model_path=model_path,
        model_label=model_label,
        tasks=tasks,
        dtype=dtype,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        batch_size=batch_size,
    )
    print("[LAUNCHED] Leaderboard eval running on Modal.")
    print(f"  App:     {APP_NAME}")
    print(f"  Version: {RUN_VERSION}")
    print(f"  Results: modal run modal_eval_open_llm_leaderboard.py::results --model-label {model_label}")


@app.local_entrypoint()
def results(model_label: str = DEFAULT_MODEL_LABEL):
    """Print saved leaderboard results table from the Modal volume."""
    _read_results.remote(model_label=model_label)
