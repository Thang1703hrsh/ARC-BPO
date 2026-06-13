"""
Run MT-Bench pairwise win-rate evaluation on Modal.

This wraps the vendored FastChat MT-Bench harness:
  1. Serve model A with vLLM and generate MT-Bench answers.
  2. Serve model B with vLLM and generate MT-Bench answers.
  3. Run FastChat pairwise-all judging with two judge models.
  4. Print win/loss/tie and adjusted win-rate via show_result.py.

The two default judges match the TBPO paper setting:
  - meta/llama3-70b-instruct
  - mistralai/Mixtral-8x22B-Instruct-v0.1

You must provide an OpenAI-compatible judge endpoint and key that can serve
those judge model names.

Smoke test on first 10 MT-Bench questions:
  modal run --detach modal_mtbench_pairwise.py::main \
    --model-a-path ducthang1703/llama3-arc-bpo-fullft-smoke-128 \
    --model-a-id llama3_arc_bpo \
    --model-b-path RLHFlow/LLaMA3-SFT-v2 \
    --model-b-id llama3_sft \
    --judge-api-base https://api.example.com/v1 \
    --judge-api-key sk_xxx \
    --first-n 10

Full run:
  modal run --detach modal_mtbench_pairwise.py::main \
    --model-a-path <candidate-full-model-or-merged-model> \
    --model-a-id <candidate_id> \
    --model-b-path <baseline-full-model-or-merged-model> \
    --model-b-id <baseline_id> \
    --judge-api-base <openai-compatible-base-url> \
    --judge-api-key <api-key>

Print current saved results:
  modal run modal_mtbench_pairwise.py::results --run-label <run_label>
"""

import modal


APP_NAME = "tbpo-mtbench-pairwise"
RUN_VERSION = "mtbench-pairwise-two-judges-v1"

FASTCHAT_SRC = "/root/FastChat"
VOLUME_ROOT = "/vol/output"
MTBENCH_ROOT = f"{VOLUME_ROOT}/mtbench_pairwise"


app = modal.App(APP_NAME)
output_volume = modal.Volume.from_name("tbpo-output", create_if_missing=True)


image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git")
    .pip_install(
        "torch>=2.6.0",
        "transformers>=4.48.0,<4.57.0",
        "accelerate",
        "huggingface_hub",
        "safetensors",
        "vllm>=0.8.5,<0.9.0",
        "openai>=1.0.0",
        "shortuuid",
        "pandas",
        "numpy",
        "requests",
        "tqdm",
        "sentencepiece",
    )
    .env(
        {
            "VLLM_USE_V1": "0",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        }
    )
    .add_local_dir(
        "mtbench/FastChat",
        remote_path=FASTCHAT_SRC,
        ignore=[
            ".git/**",
            "**/__pycache__/**",
            "*.pyc",
        ],
    )
)


_SHARED = dict(
    image=image,
    gpu="A100-80GB",
    volumes={VOLUME_ROOT: output_volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
)


def _sanitize_id(value: str) -> str:
    keep = []
    for char in value.strip().replace("/", "_"):
        if char.isalnum() or char in {"_", "-", "."}:
            keep.append(char)
        else:
            keep.append("_")
    return "".join(keep).strip("_") or "model"


def _run_label(model_a_id: str, model_b_id: str, first_n: int) -> str:
    suffix = f"first{first_n}" if first_n and first_n > 0 else "full"
    return f"{_sanitize_id(model_a_id)}__vs__{_sanitize_id(model_b_id)}__{suffix}"


@app.function(**_SHARED, timeout=60 * 60 * 24)
def run_pairwise(
    model_a_path: str,
    model_a_id: str,
    model_b_path: str,
    model_b_id: str,
    judge_api_base: str,
    judge_api_key: str,
    judge_models: str = "meta/llama3-70b-instruct,mistralai/Mixtral-8x22B-Instruct-v0.1",
    first_n: int = 0,
    run_label: str = "",
    answer_parallel: int = 16,
    judge_parallel: int = 8,
    max_tokens: int = 1024,
    force_temperature: float = 0.0,
    vllm_dtype: str = "float16",
    vllm_gpu_memory_utilization: float = 0.80,
    vllm_max_model_len: int = 4096,
    force_answer_regen: bool = False,
    force_judge_regen: bool = True,
):
    import os
    import shutil
    import signal
    import subprocess
    import time

    import requests

    if not model_a_path or not model_b_path:
        raise ValueError("model_a_path and model_b_path are required.")
    if not model_a_id:
        model_a_id = _sanitize_id(model_a_path)
    if not model_b_id:
        model_b_id = _sanitize_id(model_b_path)
    if model_a_id == model_b_id:
        raise ValueError("model_a_id and model_b_id must be different.")
    if not judge_api_base or not judge_api_key:
        raise ValueError("judge_api_base and judge_api_key are required.")

    run_label = run_label or _run_label(model_a_id, model_b_id, first_n)
    work_dir = f"{MTBENCH_ROOT}/{run_label}"
    data_dir = f"{work_dir}/data"
    os.makedirs(work_dir, exist_ok=True)

    if not os.path.isdir(data_dir):
        shutil.copytree(f"{FASTCHAT_SRC}/data", data_dir)

    env = os.environ.copy()
    env["PYTHONPATH"] = FASTCHAT_SRC + os.pathsep + env.get("PYTHONPATH", "")
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    print(f"[VERSION] {RUN_VERSION}")
    print(f"[WORKDIR] {work_dir}")
    print(f"[MODEL A] {model_a_id}: {model_a_path}")
    print(f"[MODEL B] {model_b_id}: {model_b_path}")
    print(f"[JUDGES] {judge_models}")
    print(f"[FIRST_N] {first_n or 'full'}")

    def wait_for_vllm(port: int, timeout_s: int = 1800):
        url = f"http://127.0.0.1:{port}/v1/models"
        start = time.time()
        last_error = None
        while time.time() - start < timeout_s:
            try:
                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    print(f"[VLLM READY] {url}")
                    return
                last_error = f"status={response.status_code}"
            except Exception as exc:
                last_error = repr(exc)
            time.sleep(10)
        raise TimeoutError(f"vLLM server did not become ready at {url}: {last_error}")

    def start_vllm(model_path: str, model_id: str, port: int):
        cmd = [
            "vllm",
            "serve",
            model_path,
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--served-model-name",
            model_id,
            "--dtype",
            vllm_dtype,
            "--gpu-memory-utilization",
            str(vllm_gpu_memory_utilization),
            "--max-model-len",
            str(vllm_max_model_len),
        ]
        print("[VLLM START] " + " ".join(cmd))
        proc = subprocess.Popen(cmd, cwd=work_dir, env=env)
        wait_for_vllm(port)
        return proc

    def stop_process(proc):
        if proc.poll() is not None:
            return
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=30)

    def generate_answers(model_path: str, model_id: str, port: int):
        answer_file = f"{data_dir}/mt_bench/model_answer/{model_id}.jsonl"
        if os.path.isfile(answer_file) and os.path.getsize(answer_file) > 0 and not force_answer_regen:
            print(f"[SKIP ANSWERS] {answer_file}")
            return answer_file

        if os.path.isfile(answer_file):
            os.remove(answer_file)

        proc = start_vllm(model_path, model_id, port)
        try:
            gen_env = env.copy()
            gen_env["OPENAI_API_BASE"] = f"http://127.0.0.1:{port}/v1"
            gen_env["OPENAI_API_KEY"] = "EMPTY"

            cmd = [
                "python",
                f"{FASTCHAT_SRC}/fastchat/llm_judge/gen_api_answer.py",
                "--model",
                model_id,
                "--openai-api-base",
                f"http://127.0.0.1:{port}/v1",
                "--parallel",
                str(answer_parallel),
                "--answer-file",
                answer_file,
                "--force-temperature",
                str(force_temperature),
                "--max-tokens",
                str(max_tokens),
            ]
            if first_n and first_n > 0:
                cmd.extend(["--question-begin", "0", "--question-end", str(first_n)])

            print("[GENERATE] " + " ".join(cmd))
            subprocess.run(cmd, check=True, cwd=work_dir, env=gen_env)
            output_volume.commit()
            return answer_file
        finally:
            stop_process(proc)

    generate_answers(model_a_path, model_a_id, 8000)
    generate_answers(model_b_path, model_b_id, 8001)

    judge_outputs = {}
    judges = [judge.strip() for judge in judge_models.split(",") if judge.strip()]
    for judge_model in judges:
        judgment_file = f"{data_dir}/mt_bench/model_judgment/{judge_model}_pair.jsonl"
        if os.path.isfile(judgment_file) and force_judge_regen:
            os.remove(judgment_file)

        judge_env = env.copy()
        judge_env["OPENAI_API_BASE"] = judge_api_base
        judge_env["OPENAI_API_KEY"] = judge_api_key

        cmd = [
            "python",
            f"{FASTCHAT_SRC}/fastchat/llm_judge/gen_judgment.py",
            "--mode",
            "pairwise-all",
            "--model-list",
            model_a_id,
            model_b_id,
            "--judge-model",
            judge_model,
            "--parallel",
            str(judge_parallel),
            "--openai-api-base",
            judge_api_base,
            "--openai-api-key",
            judge_api_key,
        ]
        if first_n and first_n > 0:
            cmd.extend(["--first-n", str(first_n)])

        print("[JUDGE] " + " ".join(cmd[:-1]) + " <api-key-hidden>")
        subprocess.run(cmd, check=True, cwd=work_dir, env=judge_env, input="\n", text=True)

        show_cmd = [
            "python",
            f"{FASTCHAT_SRC}/fastchat/llm_judge/show_result.py",
            "--mode",
            "pairwise-all",
            "--model-list",
            model_a_id,
            model_b_id,
            "--judge-model",
            judge_model,
        ]
        print("[RESULT] " + " ".join(show_cmd))
        completed = subprocess.run(
            show_cmd,
            check=True,
            cwd=work_dir,
            env=judge_env,
            text=True,
            capture_output=True,
        )
        print(completed.stdout)
        judge_outputs[judge_model] = completed.stdout
        output_volume.commit()

    print("[DONE] MT-Bench pairwise evaluation complete.")
    print(f"[SAVED] {work_dir}")
    return {
        "run_label": run_label,
        "work_dir": work_dir,
        "model_a_id": model_a_id,
        "model_b_id": model_b_id,
        "judges": judges,
        "results": judge_outputs,
    }


@app.function(image=image, volumes={VOLUME_ROOT: output_volume}, timeout=60 * 10)
def _read_results(run_label: str):
    import os
    import subprocess

    output_volume.reload()
    work_dir = f"{MTBENCH_ROOT}/{run_label}"
    data_dir = f"{work_dir}/data"
    if not os.path.isdir(data_dir):
        print(f"No saved MT-Bench run found at {work_dir}")
        return

    env = os.environ.copy()
    env["PYTHONPATH"] = FASTCHAT_SRC + os.pathsep + env.get("PYTHONPATH", "")

    print(f"[WORKDIR] {work_dir}")
    judgment_dir = f"{data_dir}/mt_bench/model_judgment"
    for root, _, files in os.walk(judgment_dir):
        for filename in files:
            if not filename.endswith("_pair.jsonl"):
                continue
            rel = os.path.relpath(os.path.join(root, filename), judgment_dir)
            judge_model = rel[: -len("_pair.jsonl")].replace(os.sep, "/")
            print(f"\n[JUDGE] {judge_model}")
            cmd = [
                "python",
                f"{FASTCHAT_SRC}/fastchat/llm_judge/show_result.py",
                "--mode",
                "pairwise-all",
                "--judge-model",
                judge_model,
            ]
            completed = subprocess.run(
                cmd,
                check=True,
                cwd=work_dir,
                env=env,
                text=True,
                capture_output=True,
            )
            print(completed.stdout)


@app.local_entrypoint()
def main(
    model_a_path: str,
    model_b_path: str,
    model_a_id: str = "",
    model_b_id: str = "",
    judge_api_base: str = "",
    judge_api_key: str = "",
    judge_models: str = "meta/llama3-70b-instruct,mistralai/Mixtral-8x22B-Instruct-v0.1",
    first_n: int = 0,
    run_label: str = "",
):
    """Launch MT-Bench pairwise-all evaluation."""
    model_a_id = model_a_id or _sanitize_id(model_a_path)
    model_b_id = model_b_id or _sanitize_id(model_b_path)
    run_label = run_label or _run_label(model_a_id, model_b_id, first_n)
    run_pairwise.spawn(
        model_a_path=model_a_path,
        model_a_id=model_a_id,
        model_b_path=model_b_path,
        model_b_id=model_b_id,
        judge_api_base=judge_api_base,
        judge_api_key=judge_api_key,
        judge_models=judge_models,
        first_n=first_n,
        run_label=run_label,
    )
    print("[LAUNCHED] MT-Bench pairwise run submitted.")
    print(f"  App:       {APP_NAME}")
    print(f"  Version:   {RUN_VERSION}")
    print(f"  Run label: {run_label}")
    print(f"  Results:   modal run modal_mtbench_pairwise.py::results --run-label {run_label}")


@app.local_entrypoint()
def results(run_label: str):
    """Print saved MT-Bench pairwise results from the Modal volume."""
    _read_results.remote(run_label=run_label)
