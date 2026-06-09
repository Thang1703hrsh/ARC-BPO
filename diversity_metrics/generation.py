#!/usr/bin/env python3
import argparse
import json
import os
import random
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    set_seed,
)

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
                    prompts.append({"id": i, "prompt": p, "messages": None})
    else:
        # assume jsonl
        with open(path, "r", encoding="utf-8") as f:
            i = 0
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                pid = obj.get("id", i)
                messages = obj.get("prompt_messages", None)
                prompt_text = obj.get("prompt", "")
                if messages is None and not prompt_text:
                    raise ValueError("JSONL lines must contain 'prompt_messages' or 'prompt' field.")
                prompts.append({"id": int(pid), "prompt": prompt_text, "messages": messages})
                i += 1
    if not prompts:
        raise ValueError("No prompts found.")
    return prompts


@torch.inference_mode()
def step_entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """
    logits: [batch, vocab]
    returns: [batch] entropy in nats
    """
    logp = F.log_softmax(logits, dim=-1)
    p = logp.exp()
    return -(p * logp).sum(dim=-1)


def trim_to_eos_or_pad(
    token_ids: List[int],
    entropies: List[float],
    eos_token_id: Optional[int],
    pad_token_id: Optional[int],
) -> Tuple[List[int], List[float], bool]:
    """
    Trim token_ids and entropies to:
      - first EOS inclusive, if present
      - else strip trailing PADs, if pad_token_id is known
      - else keep as-is
    Returns (trimmed_token_ids, trimmed_entropies, has_eos)
    """
    has_eos = False
    if eos_token_id is not None:
        try:
            eos_pos = token_ids.index(eos_token_id)
            has_eos = True
            cut = eos_pos + 1
            return token_ids[:cut], entropies[:cut], has_eos
        except ValueError:
            pass

    if pad_token_id is not None:
        # strip trailing pads
        cut = len(token_ids)
        while cut > 0 and token_ids[cut - 1] == pad_token_id:
            cut -= 1
        return token_ids[:cut], entropies[:cut], has_eos

    return token_ids, entropies[: len(token_ids)], has_eos


@dataclass
class SampleRecord:
    text: str
    token_ids: List[int]
    token_entropies: List[float]
    gen_len: int
    has_eos: bool


def load_model_and_tokenizer(
    model_name_or_path: str,
    dtype: str,
    device_map: str,
    trust_remote_code: bool,
):
    config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=trust_remote_code)

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path, use_fast=True, trust_remote_code=trust_remote_code
    )

    # Choose model class
    if getattr(config, "is_encoder_decoder", False):
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name_or_path,
            torch_dtype=getattr(torch, dtype),
            device_map=device_map,
            trust_remote_code=trust_remote_code,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=getattr(torch, dtype),
            device_map=device_map,
            trust_remote_code=trust_remote_code,
        )
        # For batched generation with decoder-only, left padding is strongly preferred.
        tokenizer.padding_side = "left"

    # Ensure PAD token exists
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer has no pad_token_id and no eos_token_id; cannot pad.")
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()
    return model, tokenizer, config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF model name or local path")
    ap.add_argument("--prompts", required=True, help="prompts.txt or prompts.jsonl")
    ap.add_argument("--out", required=True, help="output JSONL file")
    ap.add_argument("--k", type=int, default=5, help="num samples per prompt (paper uses 5)")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--max_prompt_tokens", type=int, default=2048)
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--top_k", type=int, default=0)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument(
        "--dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"]
    )
    ap.add_argument("--device_map", type=str, default="auto", help='e.g. "auto" or "cuda:0"')
    ap.add_argument("--trust_remote_code", action="store_true")
    args = ap.parse_args()

    set_seed(args.seed)
    random.seed(args.seed)

    model, tokenizer, config = load_model_and_tokenizer(
        args.model, args.dtype, args.device_map, args.trust_remote_code
    )

    prompts = read_prompts(args.prompts)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        meta = {
            "model": args.model,
            "k": args.k,
            "batch_size": args.batch_size,
            "max_prompt_tokens": args.max_prompt_tokens,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "seed": args.seed,
            "dtype": args.dtype,
            "is_encoder_decoder": bool(getattr(config, "is_encoder_decoder", False)),
        }
        f.write(json.dumps({"_meta": meta}, ensure_ascii=False) + "\n")

        is_encdec = meta["is_encoder_decoder"]
        eos_id = tokenizer.eos_token_id
        pad_id = tokenizer.pad_token_id

        for start in tqdm(range(0, len(prompts), args.batch_size), desc="Generating"):
            batch = prompts[start : start + args.batch_size]

            # Apply chat template if messages are available, otherwise use raw prompt
            batch_text = []
            for b in batch:
                if b["messages"] is not None:
                    text = tokenizer.apply_chat_template(
                        b["messages"],
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                else:
                    text = b["prompt"]
                batch_text.append(text)

            inputs = tokenizer(
                batch_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.max_prompt_tokens,
            )

            # Move to the model device (device_map=auto => inputs can be on any cuda; model handles it)
            # For safety: send to first parameter device if possible.
            try:
                dev = next(model.parameters()).device
                inputs = {k: v.to(dev) for k, v in inputs.items()}
            except StopIteration:
                pass

            # For decoder-only, generated tokens start after the (padded) input length.
            input_len_padded = inputs["input_ids"].shape[1] if not is_encdec else None

            gen = model.generate(
                **inputs,
                do_sample=True,
                num_return_sequences=args.k,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k if args.top_k > 0 else None,
                max_new_tokens=args.max_new_tokens,
                return_dict_in_generate=True,
                output_scores=True,
            )

            sequences = gen.sequences  # [batch*k, ...]
            scores_list = gen.scores  # list length = gen_steps; each [batch*k, vocab]

            # Compute per-step entropy: [batch*k, gen_steps]
            ent_steps = []
            for step_logits in scores_list:
                ent = step_entropy_from_logits(step_logits)  # [batch*k]
                ent_steps.append(ent.detach().float().cpu())
            ent_matrix = (
                torch.stack(ent_steps, dim=1) if ent_steps else torch.empty((sequences.size(0), 0))
            )

            # Group back into per-prompt
            # HF generate with num_return_sequences groups outputs by prompt in order.
            for i, prompt_obj in enumerate(batch):
                rec: Dict[str, Any] = {
                    "id": prompt_obj["id"],
                    "prompt": prompt_obj["prompt"],
                    "samples": [],
                }

                for j in range(args.k):
                    idx = i * args.k + j
                    seq_ids = sequences[idx].detach().cpu().tolist()

                    if is_encdec:
                        gen_token_ids = seq_ids
                    else:
                        gen_token_ids = (
                            seq_ids[input_len_padded:] if input_len_padded is not None else seq_ids
                        )

                    entropies = ent_matrix[idx].tolist()
                    gen_token_ids, entropies, has_eos = trim_to_eos_or_pad(
                        gen_token_ids, entropies, eos_id, pad_id
                    )

                    text = tokenizer.decode(gen_token_ids, skip_special_tokens=True)

                    sample = SampleRecord(
                        text=text,
                        token_ids=gen_token_ids,
                        token_entropies=entropies,
                        gen_len=len(gen_token_ids),
                        has_eos=has_eos,
                    )
                    rec["samples"].append(asdict(sample))

                f.write(json.dumps(rec, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
