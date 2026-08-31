#!/usr/bin/env python3
"""Create a reproducible diversity prompt set from held-out preference data."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional


def extract_prompt(example: Mapping[str, Any], source_index: int) -> Optional[Dict[str, Any]]:
    chosen = example.get("chosen")
    rejected = example.get("rejected")
    if not isinstance(chosen, list) or not isinstance(rejected, list):
        return None
    if len(chosen) != 2 or len(rejected) != 2 or chosen[:-1] != rejected[:-1]:
        return None
    prompt_messages = chosen[:-1]
    if not prompt_messages or any(not isinstance(message, dict) for message in prompt_messages):
        return None
    if prompt_messages[-1].get("role") != "user":
        return None
    return {
        "id": source_index,
        "source_id": example.get("id", example.get("prompt_id", source_index)),
        "prompt_messages": prompt_messages,
    }


def prepare_prompt_records(
    dataset: Iterable[Mapping[str, Any]],
    max_prompts: int,
    shuffle: bool,
    seed: int,
    include_indices: Optional[set[int]] = None,
) -> List[Dict[str, Any]]:
    indexed = list(enumerate(dataset))
    if include_indices is not None:
        indexed = [item for item in indexed if item[0] in include_indices]
    if shuffle:
        random.Random(seed).shuffle(indexed)
    records: List[Dict[str, Any]] = []
    for source_index, example in indexed:
        record = extract_prompt(example, source_index)
        if record is None:
            continue
        record["id"] = len(records)
        records.append(record)
        if max_prompts > 0 and len(records) >= max_prompts:
            break
    if not records:
        raise RuntimeError("No valid two-turn held-out prompts were found.")
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--dataset_revision", default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max_prompts", type=int, default=0, help="0 uses all valid prompts.")
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--indices_file",
        help="Optional JSON list of dataset row indices to reproduce the credit-drift subset.",
    )
    parser.add_argument("--allow_train_split", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.max_prompts < 0:
        raise ValueError("max_prompts cannot be negative.")
    if "train" in args.split.lower() and not args.allow_train_split:
        raise ValueError("Diversity prompts must come from a held-out split.")

    from datasets import load_dataset

    dataset_path = Path(args.dataset)
    if dataset_path.is_file():
        dataset = load_dataset("json", data_files={"heldout": str(dataset_path)}, split="heldout")
    else:
        dataset = load_dataset(args.dataset, split=args.split, revision=args.dataset_revision)
    include_indices = None
    if args.indices_file:
        raw_indices = json.loads(Path(args.indices_file).read_text(encoding="utf-8"))
        if not isinstance(raw_indices, list):
            raise ValueError("indices_file must contain a JSON list.")
        include_indices = {int(index) for index in raw_indices}
        if len(include_indices) != len(raw_indices):
            raise ValueError("indices_file contains duplicate dataset indices.")
    records = prepare_prompt_records(
        dataset,
        args.max_prompts,
        args.shuffle,
        args.seed,
        include_indices=include_indices,
    )

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    metadata = {
        "dataset": args.dataset,
        "split": args.split,
        "dataset_revision": args.dataset_revision,
        "num_prompts": len(records),
        "max_prompts": args.max_prompts,
        "shuffle": args.shuffle,
        "seed": args.seed,
        "indices_file": args.indices_file,
    }
    output.with_suffix(output.suffix + ".meta.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(records)} held-out prompts to {output}")


if __name__ == "__main__":
    main()
