"""
Open LLM Leaderboard v1 evaluation:
  ARC-Challenge (25-shot), HellaSwag (10-shot), TruthfulQA (0-shot),
  MMLU (5-shot), Winogrande (5-shot), GSM8K (5-shot)

Model: mistralai/Mistral-7B-v0.1  (baseline)

Step 1 — launch (detached, no heartbeat timeout):
  modal run --detach modal_eval_leaderboard_baseline.py::main

Step 2 — print results after job finishes (~2-3h):
  modal run modal_eval_leaderboard_baseline.py::results
"""

import modal

app = modal.App("fswift-eval-leaderboard-baseline")

volume = modal.Volume.from_name("fswift-volume", create_if_missing=True)

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
    .add_local_dir(".", remote_path="/root/fSWIFT", ignore=["model_hub", "__pycache__", "*.out", "logs"])
)

# ── Model config ──────────────────────────────────────────────
HF_REPO      = "mistralai/Mistral-7B-v0.1"
MODEL_LABEL  = "Mistral-7B-v0.1"

VOLUME_ROOT  = "/root/fSWIFT/model_hub"
MODEL_LOCAL  = f"{VOLUME_ROOT}/{MODEL_LABEL}"
RESULTS_DIR  = f"{VOLUME_ROOT}/eval_results/leaderboard/{MODEL_LABEL}"

# Open LLM Leaderboard v1 tasks and their few-shot settings
TASKS = [
    ("arc_challenge",   25),
    ("hellaswag",       10),
    ("truthfulqa_mc2",   0),
    ("mmlu",             5),
    ("winogrande",       5),
    ("gsm8k",            5),
]

_SHARED = dict(
    image=image,
    gpu="A100-80GB",
    volumes={VOLUME_ROOT: volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
)


# ── Download model ────────────────────────────────────────────

@app.function(**_SHARED, timeout=60 * 60)
def download_model():
    import os
    import subprocess

    if os.path.isdir(MODEL_LOCAL) and os.listdir(MODEL_LOCAL):
        print(f"[SKIP] {MODEL_LOCAL} already exists in volume.")
        return

    print(f"[DOWNLOAD] {HF_REPO}")
    subprocess.run(
        ["hf", "download", HF_REPO, "--local-dir", MODEL_LOCAL],
        check=True,
    )
    volume.commit()


# ── Run all leaderboard benchmarks in one lm_eval call ───────

@app.function(**_SHARED, timeout=60 * 60 * 6)
def eval_leaderboard():
    import os
    import subprocess

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Build --tasks and --num_fewshot args
    # lm_eval doesn't support per-task few-shot directly, so we run
    # tasks in groups by few-shot count to keep it clean.
    groups = {}
    for task, fewshot in TASKS:
        groups.setdefault(fewshot, []).append(task)

    for fewshot, task_list in sorted(groups.items()):
        tasks_str  = ",".join(task_list)
        group_dir  = f"{RESULTS_DIR}/fewshot{fewshot}"
        os.makedirs(group_dir, exist_ok=True)

        print(f"\n[EVAL] {tasks_str}  ({fewshot}-shot)")
        subprocess.run(
            [
                "python", "-m", "lm_eval",
                "--model", "vllm",
                "--model_args", (
                    f"pretrained={MODEL_LOCAL},"
                    "dtype=float16,"
                    "gpu_memory_utilization=0.85,"
                    "max_model_len=4096"
                ),
                "--tasks",       tasks_str,
                "--num_fewshot", str(fewshot),
                "--batch_size",  "auto",
                "--output_path", group_dir,
            ],
            check=True,
        )
        volume.commit()

    print("\n[DONE] All leaderboard benchmarks complete.")


# ── Print results ─────────────────────────────────────────────

@app.function(image=image, volumes={VOLUME_ROOT: volume}, timeout=60 * 5)
def _read_results():
    import glob
    import json
    import math

    volume.reload()

    def fmt(v):
        return f"{v*100:.2f}" if v is not None and not math.isnan(v) else "N/A"

    # Metric keys per task (in order of preference)
    METRIC_KEYS = {
        "arc_challenge":  ["acc_norm,none", "acc,none"],
        "hellaswag":      ["acc_norm,none", "acc,none"],
        "truthfulqa_mc2": ["acc,none"],
        "mmlu":           ["acc,none"],
        "winogrande":     ["acc,none"],
        "gsm8k":          ["exact_match,flexible-extract", "acc,none", "exact_match,none"],
    }

    all_results = {}
    for result_file in glob.glob(f"{RESULTS_DIR}/**/results*.json", recursive=True):
        data = json.load(open(result_file))
        for task, metrics in data.get("results", {}).items():
            if task in METRIC_KEYS and task not in all_results:
                for key in METRIC_KEYS[task]:
                    if key in metrics:
                        all_results[task] = metrics[key]
                        break

    avg = []
    print(f"\n{'='*45}")
    print(f" Open LLM Leaderboard — {MODEL_LABEL}")
    print(f"{'='*45}")
    for task, fewshot in TASKS:
        val = all_results.get(task)
        score = val * 100 if val is not None else float("nan")
        if not math.isnan(score):
            avg.append(score)
        label = f"{task} ({fewshot}-shot)"
        print(f"  {label:<35} {fmt(val):>6}")
    if avg:
        print(f"  {'─'*41}")
        print(f"  {'Average':<35} {sum(avg)/len(avg):>6.2f}")
    print(f"{'='*45}")


# ── Entrypoints ───────────────────────────────────────────────

@app.local_entrypoint()
def main():
    """Download model then run all leaderboard benchmarks (use --detach)."""
    download_model.remote()
    eval_leaderboard.spawn()
    print(f"[LAUNCHED] Leaderboard eval running on Modal.")
    print(f"  Logs:    modal app logs fswift-eval-leaderboard")
    print(f"  Results: modal run modal_eval_leaderboard.py::results")


@app.local_entrypoint()
def results():
    """Print leaderboard results table from volume."""
    _read_results.remote()
