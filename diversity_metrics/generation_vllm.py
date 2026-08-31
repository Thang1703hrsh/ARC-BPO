#!/usr/bin/env python3
import argparse
import json
import math
import os
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

from tqdm import tqdm
from vllm import LLM, SamplingParams

# ----------------------------
# Helpers
# ----------------------------


def read_prompts(path: str) -> List[Dict[str, Any]]:
    """
    Supports:
      - .txt: one prompt per line
      - .jsonl: each line has {"prompt_messages": [...]} or {"prompt": "..."} (optionally an "id")
    Returns list of {"id": int, "prompt": str, "messages": list | None}
    """
    prompts: List[Dict[str, Any]] = []
    if path.endswith(".txt"):
        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                p = line.strip()
                if p:
                    prompts.append({"id": i, "source_id": i, "prompt": p, "messages": None})
    else:
        # assume jsonl
        with open(path, "r", encoding="utf-8") as f:
            i = 0
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if "_meta" in obj:
                    continue
                pid = obj.get("id", i)
                source_id = obj.get("source_id", pid)
                messages = obj.get("prompt_messages", None)
                prompt_text = obj.get("prompt", "")
                if messages is None and not prompt_text:
                    raise ValueError("JSONL lines must contain 'prompt_messages' or 'prompt' field.")
                prompts.append(
                    {
                        "id": int(pid),
                        "source_id": source_id,
                        "prompt": prompt_text,
                        "messages": messages,
                    }
                )
                i += 1
    if not prompts:
        raise ValueError("No prompts found.")
    return prompts


def entropy_from_logprobs(logprobs_dict: Dict[int, float], top_k_approx: bool = True) -> float:
    """
    Compute entropy from logprobs dict {token_id: logprob}.
    If top_k_approx=True, we renormalize the top-k probs to sum to 1.
    """
    if not logprobs_dict:
        return 0.0

    logprobs = list(logprobs_dict.values())

    # Convert to probs
    probs = [math.exp(lp) for lp in logprobs]

    if top_k_approx:
        # Renormalize to account for missing probability mass
        total = sum(probs)
        if total > 0:
            probs = [p / total for p in probs]

    # Compute entropy: -sum(p * log(p))
    entropy = 0.0
    for p in probs:
        if p > 0:
            entropy -= p * math.log(p)

    return entropy


@dataclass
class SampleRecord:
    text: str
    token_ids: List[int]
    token_entropies: List[float]
    gen_len: int
    has_eos: bool


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF model name or local path")
    ap.add_argument("--prompts", required=True, help="prompts.txt or prompts.jsonl")
    ap.add_argument("--out", required=True, help="output JSONL file")
    ap.add_argument("--k", type=int, default=5, help="num samples per prompt (paper uses 5)")
    ap.add_argument("--batch_size", type=int, default=64, help="prompts per batch (vLLM handles internal batching)")
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--top_k", type=int, default=0)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--tensor_parallel_size", type=int, default=1, help="Number of GPUs for tensor parallelism")
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    ap.add_argument("--max_model_len", type=int, default=4096)
    ap.add_argument("--logprobs_k", type=int, default=20, help="Number of top logprobs to return for entropy estimation")
    ap.add_argument("--trust_remote_code", action="store_true")
    ap.add_argument("--dtype", type=str, default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    args = ap.parse_args()

    # Initialize vLLM with tensor parallelism
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        trust_remote_code=args.trust_remote_code,
        dtype=args.dtype,
        seed=args.seed,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
    )

    tokenizer = llm.get_tokenizer()

    # Sampling parameters
    sampling_params = SamplingParams(
        n=args.k,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k if args.top_k > 0 else -1,
        max_tokens=args.max_new_tokens,
        logprobs=args.logprobs_k,
    )

    prompts = read_prompts(args.prompts)

    # Prepare prompt texts with chat template
    prompt_texts = []
    for p in prompts:
        if p["messages"] is not None:
            text = tokenizer.apply_chat_template(
                p["messages"],
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            text = p["prompt"]
        prompt_texts.append(text)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    with open(args.out, "w", encoding="utf-8") as f:
        # Write metadata
        meta = {
            "model": args.model,
            "k": args.k,
            "batch_size": args.batch_size,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "seed": args.seed,
            "tensor_parallel_size": args.tensor_parallel_size,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "max_model_len": args.max_model_len,
            "logprobs_k": args.logprobs_k,
            "backend": "vllm",
            "entropy_estimator": "renormalized_top_k",
        }
        f.write(json.dumps({"_meta": meta}, ensure_ascii=False) + "\n")

        eos_id = tokenizer.eos_token_id

        # Process in batches
        for start in tqdm(range(0, len(prompts), args.batch_size), desc="Generating"):
            batch_prompts = prompts[start : start + args.batch_size]
            batch_texts = prompt_texts[start : start + args.batch_size]

            # Generate with vLLM
            outputs = llm.generate(batch_texts, sampling_params)

            for prompt_obj, rendered_prompt, output in zip(batch_prompts, batch_texts, outputs):
                rec: Dict[str, Any] = {
                    "id": prompt_obj["id"],
                    "source_id": prompt_obj["source_id"],
                    "prompt": prompt_obj["prompt"],
                    "prompt_messages": prompt_obj["messages"],
                    "rendered_prompt": rendered_prompt,
                    "samples": [],
                }

                for completion in output.outputs:
                    token_ids = list(completion.token_ids)
                    text = completion.text

                    # Compute token entropies from logprobs
                    token_entropies = []
                    if completion.logprobs:
                        for step_logprobs in completion.logprobs:
                            if step_logprobs:
                                # step_logprobs is a dict {token_id: Logprob}
                                logprobs_dict = {tid: lp.logprob for tid, lp in step_logprobs.items()}
                                ent = entropy_from_logprobs(logprobs_dict)
                                token_entropies.append(ent)
                            else:
                                token_entropies.append(0.0)

                    # Check for EOS
                    has_eos = eos_id is not None and eos_id in token_ids

                    sample = SampleRecord(
                        text=text,
                        token_ids=token_ids,
                        token_entropies=token_entropies,
                        gen_len=len(token_ids),
                        has_eos=has_eos,
                    )
                    rec["samples"].append(asdict(sample))

                f.write(json.dumps(rec, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
