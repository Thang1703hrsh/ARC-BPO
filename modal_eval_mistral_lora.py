"""
Quick Open LLM Leaderboard v1 evaluation for a Mistral ARC-BPO LoRA adapter.

Default model:
  ducthang1703/mistral-arc-bpo-uniform-lora-10k

Default tasks:
  arc_challenge      25-shot
  hellaswag          10-shot
  truthfulqa_mc2      0-shot
  mmlu                5-shot
  winogrande          5-shot
  gsm8k               5-shot

Run the default 6-task eval:
  modal run --detach modal_eval_mistral_lora.py::main

Run a subset:
  modal run --detach modal_eval_mistral_lora.py::main --tasks arc,gsm8k

Print saved results:
  modal run modal_eval_mistral_lora.py::results
"""

import modal


APP_NAME = "tbpo-mistral-lora-eval"
DEFAULT_MODEL = "ducthang1703/mistral-arc-bpo-uniform-lora-10k"
DEFAULT_MODEL_LABEL = "mistral-arc-bpo-uniform-lora-10k"
RUN_VERSION = "mistral-arc-bpo-lora-fulltask-v1"

VOLUME_ROOT = "/vol/output/open_llm_eval"


app = modal.App(APP_NAME)
output_volume = modal.Volume.from_name("tbpo-output", create_if_missing=True)


image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git")
    .pip_install(
        # transformers refuses to torch.load .bin checkpoints with torch < 2.6
        # after CVE-2025-32434. The Mistral SFT base may be stored as .bin
        # shards, so keep torch and vLLM on a compatible newer stack.
        "torch>=2.6.0",
        "transformers>=4.48.0,<4.57.0",
        "accelerate",
        "datasets",
        "huggingface_hub",
        "peft",
        "safetensors",
        "lm_eval",
        "vllm>=0.8.5,<0.9.0",
    )
    .env(
        {
            "VLLM_USE_V1": "0",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
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

TASK_ALIASES = {
    "arc": "arc_challenge",
    "arc_challenge": "arc_challenge",
    "truthfulqa": "truthfulqa_mc2",
    "truthfulqa_mc": "truthfulqa_mc2",
    "truthfulqa_mc2": "truthfulqa_mc2",
    "hellaswag": "hellaswag",
    "hs": "hellaswag",
    "mmlu": "mmlu",
    "winogrande": "winogrande",
    "wino": "winogrande",
    "gsm": "gsm8k",
    "gsm8k": "gsm8k",
}


_SHARED = dict(
    image=image,
    gpu="A100-80GB",
    volumes={VOLUME_ROOT: output_volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
)


def _label_from_model(model_path: str) -> str:
    return model_path.strip("/").replace("/", "__")


def _patch_tokenizer_config(model_dir: str):
    import json
    import os

    tokenizer_config = os.path.join(model_dir, "tokenizer_config.json")
    if not os.path.isfile(tokenizer_config):
        return

    with open(tokenizer_config, encoding="utf-8") as f:
        data = json.load(f)

    changed = False

    if data.get("tokenizer_class") == "TokenizersBackend":
        data["tokenizer_class"] = "PreTrainedTokenizerFast"
        changed = True
        print("[TOKENIZER] Patched tokenizer_class TokenizersBackend -> PreTrainedTokenizerFast")

    # Some checkpoints save this field as a list. Newer transformers versions
    # expect a mapping and fail with: AttributeError: 'list' object has no attribute 'keys'.
    if isinstance(data.get("extra_special_tokens"), list):
        data["extra_special_tokens"] = {}
        changed = True
        print("[TOKENIZER] Patched extra_special_tokens list -> dict")

    if changed:
        with open(tokenizer_config, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)


def _patch_model_config(model_dir: str):
    import json
    import os

    config_path = os.path.join(model_dir, "config.json")
    if not os.path.isfile(config_path):
        return

    with open(config_path, encoding="utf-8") as f:
        data = json.load(f)

    changed = False

    # Some transformers versions save Mistral configs with "head_dim": null.
    # vLLM 0.8.x checks the attribute directly and then fails with
    # TypeError: unsupported operand type(s) for *: 'int' and 'NoneType'.
    if data.get("head_dim") is None:
        hidden_size = data.get("hidden_size")
        num_attention_heads = data.get("num_attention_heads")
        if hidden_size and num_attention_heads:
            data["head_dim"] = int(hidden_size) // int(num_attention_heads)
            changed = True
            print(f"[CONFIG] Patched head_dim -> {data['head_dim']}")

    if changed:
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)


def _looks_like_full_model_dir(model_dir: str) -> bool:
    import glob
    import os

    if not os.path.isdir(model_dir):
        return False

    has_config = os.path.isfile(os.path.join(model_dir, "config.json"))
    has_weights = any(
        glob.glob(os.path.join(model_dir, pattern))
        for pattern in (
            "*.safetensors",
            "pytorch_model*.bin",
            "model*.bin",
        )
    )
    return has_config and has_weights


def _find_full_model_dir(download_dir: str):
    import glob
    import os

    candidates = [download_dir]
    candidates.extend(sorted(glob.glob(os.path.join(download_dir, "LATEST"))))
    candidates.extend(sorted(glob.glob(os.path.join(download_dir, "step-*"))))
    candidates.extend(sorted(glob.glob(os.path.join(download_dir, "checkpoint-*"))))
    candidates.extend(
        sorted(
            os.path.dirname(path)
            for path in glob.glob(os.path.join(download_dir, "**", "config.json"), recursive=True)
        )
    )
    candidates.extend(
        sorted(
            path
            for path in glob.glob(os.path.join(download_dir, "*"))
            if os.path.isdir(path)
        )
    )

    seen = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if _looks_like_full_model_dir(candidate):
            return candidate

    return None


def _normalize_task_name(task: str) -> str:
    return TASK_ALIASES.get(task.strip(), task.strip())


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
    import shutil
    import subprocess

    model_local = f"{VOLUME_ROOT}/models/{model_label}"
    if os.path.isdir(model_local) and os.listdir(model_local):
        if _looks_like_full_model_dir(model_local):
            print(f"[SKIP] {model_local} already exists in volume.")
            _patch_tokenizer_config(model_local)
            _patch_model_config(model_local)
            output_volume.commit()
            return model_local
        print(f"[REFRESH] Removing incomplete local model at {model_local}.")
        shutil.rmtree(model_local)

    download_local = f"{VOLUME_ROOT}/adapters/{model_label}"

    def download_repo():
        subprocess.run(
            ["hf", "download", model_path, "--local-dir", download_local],
            check=True,
        )

    print(f"[VERSION] {RUN_VERSION}")
    print(f"[DOWNLOAD MODEL] {model_path}")
    if not os.path.isdir(download_local) or not os.listdir(download_local):
        download_repo()
    else:
        print(f"[SKIP DOWNLOAD] {download_local} already exists in volume.")

    adapter_config = os.path.join(download_local, "adapter_config.json")
    if not os.path.isfile(adapter_config):
        full_model_dir = _find_full_model_dir(download_local)
        if full_model_dir:
            print(f"[FULL MODEL] Found checkpoint at {full_model_dir}")
            print(f"[FULL MODEL] Copying to {model_local}")
            shutil.copytree(full_model_dir, model_local, dirs_exist_ok=True)
            _patch_tokenizer_config(model_local)
            _patch_model_config(model_local)
            output_volume.commit()
            return model_local

        print(f"[INVALID CACHE] {download_local} is not a LoRA adapter or full checkpoint.")
        print("[INVALID CACHE] Removing cached directory and downloading again.")
        shutil.rmtree(download_local, ignore_errors=True)
        download_repo()

    adapter_config = os.path.join(download_local, "adapter_config.json")
    if not os.path.isfile(adapter_config):
        full_model_dir = _find_full_model_dir(download_local)
        if full_model_dir:
            print(f"[FULL MODEL] Found checkpoint at {full_model_dir}")
            print(f"[FULL MODEL] Copying to {model_local}")
            shutil.copytree(full_model_dir, model_local, dirs_exist_ok=True)
            _patch_tokenizer_config(model_local)
            _patch_model_config(model_local)
            output_volume.commit()
            return model_local

        raise RuntimeError(
            f"{model_path} does not look like a PEFT LoRA adapter repo: "
            f"missing {adapter_config}"
        )

    print(f"[MERGE LORA] adapter={download_local}")
    print(f"[MERGE LORA] output={model_local}")

    import torch
    from peft import PeftConfig, PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    peft_config = PeftConfig.from_pretrained(download_local)
    base_model_name = peft_config.base_model_name_or_path
    print(f"[BASE MODEL] {base_model_name}")

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base_model, download_local)
    model = model.merge_and_unload()

    os.makedirs(model_local, exist_ok=True)
    model.save_pretrained(model_local, safe_serialization=True)

    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name,
        use_fast=True,
        trust_remote_code=True,
    )
    tokenizer.save_pretrained(model_local)
    _patch_tokenizer_config(model_local)
    _patch_model_config(model_local)
    output_volume.commit()
    return model_local


@app.function(**_SHARED, timeout=60 * 60 * 8)
def eval_open_llm_leaderboard(
    model_path: str = DEFAULT_MODEL,
    model_label: str = DEFAULT_MODEL_LABEL,
    tasks: str = "all",
    dtype: str = "float16",
    gpu_memory_utilization: float = 0.75,
    max_model_len: int = 2048,
    batch_size: str = "auto",
):
    import os
    import subprocess

    model_local = f"{VOLUME_ROOT}/models/{model_label}"
    if not os.path.isdir(model_local) or not os.listdir(model_local):
        raise RuntimeError(
            f"Local model not found at {model_local}. Run download_model first."
        )
    _patch_tokenizer_config(model_local)
    _patch_model_config(model_local)
    output_volume.commit()

    if tasks == "all":
        selected_tasks = OPEN_LLM_TASKS
    else:
        wanted = {_normalize_task_name(task) for task in tasks.split(",") if task.strip()}
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

    env = os.environ.copy()
    env["VLLM_USE_V1"] = "0"
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

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
        subprocess.run(cmd, check=True, env=env)
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
        "mmlu": ["acc,none", "acc_norm,none"],
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
    print(f" Open LLM Leaderboard 6-task - {model_label}")
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
    gpu_memory_utilization: float = 0.70,
    max_model_len: int = 2048,
    batch_size: str = "4",
):
    """Download and merge the LoRA adapter, then run the selected 6-task eval."""
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
    print("[LAUNCHED] 6-task leaderboard eval running on Modal.")
    print(f"  App:     {APP_NAME}")
    print(f"  Version: {RUN_VERSION}")
    print(f"  Results: modal run modal_eval_mistral_lora.py::results --model-label {model_label}")


@app.local_entrypoint()
def results(model_label: str = DEFAULT_MODEL_LABEL):
    """Print saved 6-task leaderboard results table from the Modal volume."""
    _read_results.remote(model_label=model_label)
