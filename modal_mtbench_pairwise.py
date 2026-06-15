"""
Run MT-Bench pairwise win-rate evaluation on Modal.

This wraps the vendored FastChat MT-Bench harness:
  1. Download/merge model A if it is a PEFT LoRA adapter, then serve with vLLM.
  2. Download/merge model B if it is a PEFT LoRA adapter, then serve with vLLM.
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
MODEL_ROOT = f"{VOLUME_ROOT}/generation_quality_models"


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
        "peft",
        "safetensors",
        "vllm>=0.8.5,<0.9.0",
        "fastapi==0.115.12",
        "starlette==0.46.2",
        "prometheus-fastapi-instrumentator==7.0.2",
        "openai>=1.0.0",
        "anthropic",
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
        for pattern in ("*.safetensors", "pytorch_model*.bin", "model*.bin")
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

    seen = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if _looks_like_full_model_dir(candidate):
            return candidate
    return None


def _prepare_model_if_needed(model_path: str, model_label: str):
    import os
    import shutil
    import subprocess

    safe_label = _sanitize_id(model_label or model_path)
    model_local = f"{MODEL_ROOT}/merged/{safe_label}"
    if os.path.isdir(model_local) and os.listdir(model_local):
        print(f"[SKIP MODEL] {model_local}")
        _patch_tokenizer_config(model_local)
        _patch_model_config(model_local)
        output_volume.commit()
        return model_local

    download_local = f"{MODEL_ROOT}/downloads/{safe_label}"
    if not os.path.isdir(download_local) or not os.listdir(download_local):
        print(f"[DOWNLOAD] {model_path}")
        subprocess.run(["hf", "download", model_path, "--local-dir", download_local], check=True)

    adapter_config = os.path.join(download_local, "adapter_config.json")
    if not os.path.isfile(adapter_config):
        full_model_dir = _find_full_model_dir(download_local)
        if full_model_dir is None:
            raise RuntimeError(f"{model_path} is neither a PEFT LoRA adapter nor a full HF model checkpoint.")
        if full_model_dir != download_local:
            print(f"[FULL MODEL] found nested checkpoint at {full_model_dir}")
            print(f"[FULL MODEL] copying to {model_local}")
            shutil.copytree(full_model_dir, model_local, dirs_exist_ok=True)
            full_model_dir = model_local
        else:
            print(f"[FULL MODEL] using downloaded model at {download_local}")
        _patch_tokenizer_config(full_model_dir)
        _patch_model_config(full_model_dir)
        output_volume.commit()
        return full_model_dir

    print(f"[LORA ADAPTER] merging {download_local}")
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
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base_model, download_local)
    model = model.merge_and_unload()
    os.makedirs(model_local, exist_ok=True)
    model.save_pretrained(model_local, safe_serialization=True)
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=True, trust_remote_code=True)
    tokenizer.save_pretrained(model_local)
    _patch_tokenizer_config(model_local)
    _patch_model_config(model_local)
    output_volume.commit()
    del model
    del base_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return model_local


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
    answer_parallel: int = 4,
    judge_parallel: int = 8,
    max_tokens: int = 1024,
    force_temperature: float = 0.0,
    vllm_dtype: str = "float16",
    vllm_gpu_memory_utilization: float = 0.55,
    vllm_max_model_len: int = 4096,
    vllm_max_num_seqs: int = 8,
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

    model_a_local = _prepare_model_if_needed(model_a_path, model_a_id)
    model_b_local = _prepare_model_if_needed(model_b_path, model_b_id)
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

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
            "--max-num-seqs",
            str(vllm_max_num_seqs),
            "--enforce-eager",
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

    generate_answers(model_a_local, model_a_id, 8000)
    generate_answers(model_b_local, model_b_id, 8001)

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
    answer_parallel: int = 4,
    judge_parallel: int = 8,
    max_tokens: int = 1024,
    vllm_gpu_memory_utilization: float = 0.55,
    vllm_max_model_len: int = 4096,
    vllm_max_num_seqs: int = 8,
    force_answer_regen: bool = False,
    force_judge_regen: bool = True,
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
        answer_parallel=answer_parallel,
        judge_parallel=judge_parallel,
        max_tokens=max_tokens,
        vllm_gpu_memory_utilization=vllm_gpu_memory_utilization,
        vllm_max_model_len=vllm_max_model_len,
        vllm_max_num_seqs=vllm_max_num_seqs,
        force_answer_regen=force_answer_regen,
        force_judge_regen=force_judge_regen,
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
