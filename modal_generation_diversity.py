"""
Evaluate generation diversity on UltraFeedback Binarized prompts.

This matches the Generation Quality diversity setting in TBPO.tex:
  - 100 held-out prompts from HuggingFaceH4/ultrafeedback_binarized test split
  - report Distinct-1, Self-BLEU, and predictive entropy

The model can be either a full HF model repo or a PEFT LoRA adapter repo. LoRA
adapters are downloaded, merged into their base model, and then served by vLLM.

Run a smoke test:
  modal run --detach modal_generation_diversity.py::main \
    --model-path ducthang1703/llama3-8b-arc-bpo-lora-noise20-16k \
    --model-label llama3_noise20_16k \
    --num-prompts 10

Run the paper-size setting:
  modal run --detach modal_generation_diversity.py::main \
    --model-path ducthang1703/llama3-arc-bpo-uniform-lora-full-bs64 \
    --model-label llama3_arc_bpo_lora_full_bs64 \
    --num-prompts 100

Force a clean rerun if a previous run wrote partial generations:
  modal run --detach modal_generation_diversity.py::main \
    --model-path ducthang1703/llama3-arc-bpo-uniform-lora-full-bs64 \
    --model-label llama3_arc_bpo_lora_full_bs64 \
    --num-prompts 100 \
    --force-regen

Print saved results:
  modal run modal_generation_diversity.py::results --run-label <run_label>
"""

import modal


APP_NAME = "tbpo-generation-diversity"
RUN_VERSION = "ultrafeedback-diversity-v1"

VOLUME_ROOT = "/vol/output"
RUN_ROOT = f"{VOLUME_ROOT}/generation_diversity"
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
        "datasets",
        "huggingface_hub",
        "peft",
        "safetensors",
        "vllm>=0.8.5,<0.9.0",
        "openai>=1.0.0",
        "numpy",
        "nltk",
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
        keep.append(char if char.isalnum() or char in {"_", "-", "."} else "_")
    return "".join(keep).strip("_") or "model"


def _run_label(model_label: str, num_prompts: int, samples_per_prompt: int) -> str:
    return f"{_sanitize_id(model_label)}__diversity__n{num_prompts}x{samples_per_prompt}"


def _extract_prompt(example) -> str:
    chosen = example.get("chosen")
    if isinstance(chosen, list) and chosen:
        first = chosen[0]
        if isinstance(first, dict):
            return first.get("content", "").strip()
        return str(first).strip()
    prompt = example.get("prompt")
    if prompt is not None:
        return str(prompt).strip()
    return ""


def _tokenize_words(text: str) -> list[str]:
    return [tok.lower() for tok in text.split() if tok.strip()]


def _distinct_1(texts: list[str]) -> float:
    toks = []
    for text in texts:
        toks.extend(_tokenize_words(text))
    if not toks:
        return float("nan")
    return len(set(toks)) / len(toks)


def _self_bleu(texts: list[str]) -> float:
    from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

    tokenized = [_tokenize_words(text) for text in texts if _tokenize_words(text)]
    if len(tokenized) < 2:
        return float("nan")
    smooth = SmoothingFunction().method1
    scores = []
    for idx, hyp in enumerate(tokenized):
        refs = [tokens for j, tokens in enumerate(tokenized) if j != idx and tokens]
        if refs and hyp:
            scores.append(sentence_bleu(refs, hyp, smoothing_function=smooth))
    return sum(scores) / len(scores) if scores else float("nan")


def _entropy_from_top_logprobs(top_logprobs) -> float | None:
    import math

    if not top_logprobs:
        return None
    probs = []
    for item in top_logprobs:
        logprob = None
        if isinstance(item, dict):
            logprob = item.get("logprob")
        else:
            logprob = getattr(item, "logprob", None)
        if logprob is not None:
            probs.append(math.exp(float(logprob)))
    if not probs:
        return None
    total = sum(probs)
    if total <= 0:
        return None
    norm = [p / total for p in probs]
    return -sum(p * math.log(max(p, 1e-12)) for p in norm)


def _extract_mean_entropy(choice) -> float | None:
    content = getattr(getattr(choice, "logprobs", None), "content", None)
    if not content:
        return None
    entropies = []
    for token_logprobs in content:
        top = getattr(token_logprobs, "top_logprobs", None)
        entropy = _entropy_from_top_logprobs(top)
        if entropy is not None:
            entropies.append(entropy)
    if not entropies:
        return None
    return sum(entropies) / len(entropies)


def _patch_tokenizer_config(model_dir: str):
    import json
    import os

    tokenizer_config = os.path.join(model_dir, "tokenizer_config.json")
    if not os.path.isfile(tokenizer_config):
        return
    with open(tokenizer_config, encoding="utf-8") as f:
        data = json.load(f)
    if data.get("tokenizer_class") == "TokenizersBackend":
        data["tokenizer_class"] = "PreTrainedTokenizerFast"
        with open(tokenizer_config, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print("[TOKENIZER] Patched tokenizer_class TokenizersBackend -> PreTrainedTokenizerFast")


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


def _prepare_model_if_needed(model_path: str, model_label: str):
    import os
    import subprocess

    model_local = f"{MODEL_ROOT}/merged/{model_label}"
    if os.path.isdir(model_local) and os.listdir(model_local):
        print(f"[SKIP MODEL] {model_local}")
        _patch_tokenizer_config(model_local)
        _patch_model_config(model_local)
        output_volume.commit()
        return model_local

    download_local = f"{MODEL_ROOT}/downloads/{model_label}"
    if not os.path.isdir(download_local) or not os.listdir(download_local):
        print(f"[DOWNLOAD] {model_path}")
        subprocess.run(["hf", "download", model_path, "--local-dir", download_local], check=True)

    adapter_config = f"{download_local}/adapter_config.json"
    if not os.path.isfile(adapter_config):
        print(f"[FULL MODEL] using downloaded model at {download_local}")
        _patch_tokenizer_config(download_local)
        _patch_model_config(download_local)
        output_volume.commit()
        return download_local

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


@app.function(**_SHARED, timeout=60 * 60 * 12)
def run_diversity_eval(
    model_path: str,
    model_label: str,
    num_prompts: int = 100,
    samples_per_prompt: int = 1,
    run_label: str = "",
    dataset_name: str = "HuggingFaceH4/ultrafeedback_binarized",
    split: str = "test_prefs",
    max_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
    vllm_dtype: str = "float16",
    vllm_gpu_memory_utilization: float = 0.70,
    vllm_max_model_len: int = 4096,
    force_regen: bool = False,
):
    import json
    import math
    import os
    import signal
    import subprocess
    import time

    import requests
    from datasets import load_dataset
    from openai import OpenAI

    if not model_path:
        raise ValueError("model_path is required.")
    if not model_label:
        model_label = _sanitize_id(model_path)
    if samples_per_prompt <= 0:
        raise ValueError("samples_per_prompt must be positive.")

    run_label = run_label or _run_label(model_label, num_prompts, samples_per_prompt)
    work_dir = f"{RUN_ROOT}/{run_label}"
    os.makedirs(work_dir, exist_ok=True)

    print(f"[VERSION] {RUN_VERSION}")
    print(f"[MODEL] {model_path}")
    print(f"[RUN] {run_label}")
    print(f"[PROMPTS] {num_prompts}, samples_per_prompt={samples_per_prompt}")
    model_local = _prepare_model_if_needed(model_path, model_label)

    ds = load_dataset(dataset_name, split=split)
    prompts = []
    seen = set()
    for example in ds:
        prompt = _extract_prompt(example)
        if prompt and prompt not in seen:
            prompts.append(prompt)
            seen.add(prompt)
        if len(prompts) >= num_prompts:
            break

    output_file = f"{work_dir}/generations.jsonl"
    expected_generations = len(prompts) * samples_per_prompt
    regenerate = force_regen or not os.path.isfile(output_file)
    if not regenerate:
        with open(output_file, encoding="utf-8") as f:
            existing_generations = sum(1 for line in f if line.strip())
        if existing_generations != expected_generations:
            print(
                f"[REGEN] existing generations={existing_generations}, "
                f"expected={expected_generations}; regenerating."
            )
            regenerate = True

    if regenerate:
        env = os.environ.copy()
        env["TOKENIZERS_PARALLELISM"] = "false"
        env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

        cmd = [
            "vllm",
            "serve",
            model_local,
            "--host",
            "127.0.0.1",
            "--port",
            "8000",
            "--served-model-name",
            model_label,
            "--dtype",
            vllm_dtype,
            "--gpu-memory-utilization",
            str(vllm_gpu_memory_utilization),
            "--max-model-len",
            str(vllm_max_model_len),
        ]
        print("[VLLM START] " + " ".join(cmd))
        proc = subprocess.Popen(cmd, cwd=work_dir, env=env)
        try:
            start = time.time()
            while time.time() - start < 1800:
                try:
                    if requests.get("http://127.0.0.1:8000/v1/models", timeout=5).status_code == 200:
                        break
                except Exception:
                    pass
                time.sleep(10)
            else:
                raise TimeoutError("vLLM server did not become ready.")

            client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="EMPTY")
            with open(output_file, "w", encoding="utf-8") as f:
                for prompt_idx, prompt in enumerate(prompts):
                    messages = [{"role": "user", "content": prompt}]
                    for sample_idx in range(samples_per_prompt):
                        response = client.chat.completions.create(
                            model=model_label,
                            messages=messages,
                            max_tokens=max_tokens,
                            temperature=temperature,
                            top_p=top_p,
                            logprobs=True,
                            top_logprobs=5,
                        )
                        choice = response.choices[0]
                        text = choice.message.content.strip()
                        entropy = _extract_mean_entropy(choice)
                        f.write(
                            json.dumps(
                                {
                                    "prompt_idx": prompt_idx,
                                    "sample_idx": sample_idx,
                                    "prompt": prompt,
                                    "generation": text,
                                    "predictive_entropy_top5": entropy,
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                        f.flush()
            output_volume.commit()
        finally:
            if proc.poll() is None:
                proc.send_signal(signal.SIGTERM)
                try:
                    proc.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=30)
    else:
        print(f"[SKIP GENERATIONS] {output_file}")

    rows = []
    with open(output_file, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    generations = [row["generation"] for row in rows]
    entropies = [
        row["predictive_entropy_top5"]
        for row in rows
        if row.get("predictive_entropy_top5") is not None
        and not math.isnan(float(row["predictive_entropy_top5"]))
    ]
    summary = {
        "model_path": model_path,
        "model_label": model_label,
        "n_prompts": len(prompts),
        "n_generations": len(generations),
        "samples_per_prompt": samples_per_prompt,
        "distinct_1": _distinct_1(generations),
        "self_bleu": _self_bleu(generations),
        "predictive_entropy_top5_mean": sum(entropies) / len(entropies) if entropies else None,
        "predictive_entropy_available": len(entropies),
        "avg_generation_len_words": (
            sum(len(text.split()) for text in generations) / max(len(generations), 1)
        ),
    }
    with open(f"{work_dir}/summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    output_volume.commit()
    print(json.dumps(summary, indent=2))
    print(f"[DONE] saved to {work_dir}")
    return summary


@app.function(image=image, volumes={VOLUME_ROOT: output_volume}, timeout=60 * 5)
def _read_results(run_label: str):
    import json
    import os

    output_volume.reload()
    summary_file = f"{RUN_ROOT}/{run_label}/summary.json"
    if not os.path.isfile(summary_file):
        print(f"No summary found at {summary_file}")
        return
    with open(summary_file, encoding="utf-8") as f:
        summary = json.load(f)
    print(f"[RUN] {run_label}")
    print(json.dumps(summary, indent=2))


@app.local_entrypoint()
def main(
    model_path: str,
    model_label: str = "",
    num_prompts: int = 100,
    samples_per_prompt: int = 1,
    run_label: str = "",
    force_regen: bool = False,
):
    model_label = model_label or _sanitize_id(model_path)
    run_label = run_label or _run_label(model_label, num_prompts, samples_per_prompt)
    run_diversity_eval.spawn(
        model_path=model_path,
        model_label=model_label,
        num_prompts=num_prompts,
        samples_per_prompt=samples_per_prompt,
        run_label=run_label,
        force_regen=force_regen,
    )
    print("[LAUNCHED] Generation diversity eval submitted.")
    print(f"  Run label: {run_label}")
    print(f"  Results:   modal run modal_generation_diversity.py::results --run-label {run_label}")


@app.local_entrypoint()
def results(run_label: str):
    _read_results.remote(run_label)
