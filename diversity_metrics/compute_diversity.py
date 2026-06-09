#!/usr/bin/env python3
import argparse
import json
import math
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

import sacrebleu

# ----------------------------
# Load
# ----------------------------


def iter_generation_file(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "_meta" in obj:
                continue
            yield obj


# ----------------------------
# Distinct-1
# ----------------------------

_word_re = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def distinct_1_from_text(text: str) -> float:
    toks = _word_re.findall(text.lower())
    if not toks:
        return 0.0
    return len(set(toks)) / len(toks)


# ----------------------------
# Self-BLEU
# ----------------------------


def self_bleu_ordered(responses: List[str]) -> float:
    """
    Paper says "BLEU score for each pair of responses" :contentReference[oaicite:5]{index=5}.
    This implements ordered pairs i != j and averages (non-symmetric, but common in self-BLEU usage).
    """
    if len(responses) < 2:
        return 0.0
    scores = []
    for i in range(len(responses)):
        for j in range(len(responses)):
            if i == j:
                continue
            s = sacrebleu.sentence_bleu(responses[i], [responses[j]]).score
            scores.append(s)
    return sum(scores) / len(scores)


# ----------------------------
# Predictive entropy
# ----------------------------


def mean_predictive_entropy(
    sample: Dict[str, Any], exclude_eos: bool = False, eos_token_id: Optional[int] = None
) -> float:
    ent = sample.get("token_entropies", [])
    tok_ids = sample.get("token_ids", [])
    if not ent or not tok_ids:
        return 0.0

    n = min(len(ent), len(tok_ids))
    ent = ent[:n]
    tok_ids = tok_ids[:n]

    if exclude_eos and eos_token_id is not None and n > 0 and tok_ids[-1] == eos_token_id:
        ent = ent[:-1]

    if not ent:
        return 0.0
    return sum(ent) / len(ent)


# ----------------------------
# Main metric computation
# ----------------------------


def compute_metrics(
    path: str, distinct_sample_index: int = 0, exclude_eos_in_entropy: bool = False
) -> Dict[str, Any]:
    distinct_vals: List[float] = []
    self_bleu_vals: List[float] = []
    entropy_vals: List[float] = []

    num_prompts = 0

    for obj in iter_generation_file(path):
        num_prompts += 1
        samples = obj["samples"]
        texts = [s["text"] for s in samples]

        # Distinct-1: paper uses 1 sample per prompt :contentReference[oaicite:6]{index=6}
        idx = min(distinct_sample_index, len(samples) - 1)
        distinct_vals.append(distinct_1_from_text(texts[idx]))

        # Self-BLEU: paper uses 5 samples and averages over pairs :contentReference[oaicite:7]{index=7}
        self_bleu_vals.append(self_bleu_ordered(texts))

        # Predictive entropy: paper samples 5 responses per prompt :contentReference[oaicite:8]{index=8}
        per_prompt_ent = []
        for s in samples:
            per_prompt_ent.append(
                mean_predictive_entropy(s, exclude_eos=exclude_eos_in_entropy, eos_token_id=None)
            )
        entropy_vals.append(sum(per_prompt_ent) / len(per_prompt_ent) if per_prompt_ent else 0.0)

    def mean_std(xs: List[float]) -> Tuple[float, float]:
        if not xs:
            return 0.0, 0.0
        m = sum(xs) / len(xs)
        v = sum((x - m) ** 2 for x in xs) / max(1, (len(xs) - 1))
        return m, math.sqrt(v)

    entropy_mean, entropy_std = mean_std(entropy_vals)
    selfbleu_mean, selfbleu_std = mean_std(self_bleu_vals)
    distinct_mean, distinct_std = mean_std(distinct_vals)

    return {
        "num_prompts": num_prompts,
        "predictive_entropy_mean_nats": entropy_mean,
        "predictive_entropy_std_nats": entropy_std,
        "self_bleu_mean": selfbleu_mean,
        "self_bleu_std": selfbleu_std,
        "distinct_1_mean": distinct_mean,
        "distinct_1_std": distinct_std,
        # reminder of intended directions from the paper table:
        # Entropy ↑, Self-BLEU ↓, Distinct-1 ↑ :contentReference[oaicite:9]{index=9}
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--infile", required=True, help="JSONL produced by generate_samples.py")
    ap.add_argument("--out", default=None, help="Optional output JSON file for metrics")
    ap.add_argument(
        "--distinct_sample_index", type=int, default=0, help="Which sample to use for Distinct-1"
    )
    ap.add_argument(
        "--exclude_eos_in_entropy", action="store_true", help="Drop last token entropy if it is EOS"
    )
    args = ap.parse_args()

    metrics = compute_metrics(
        args.infile,
        distinct_sample_index=args.distinct_sample_index,
        exclude_eos_in_entropy=args.exclude_eos_in_entropy,
    )

    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
