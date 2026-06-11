import random
from collections import defaultdict
from typing import Callable, Dict, Iterator, List, Optional, Union

import datasets
import numpy as np
import torch
import tqdm
from torch.nn.utils.rnn import pad_sequence

from arc_bpo_chunking import chunk_preference_pair
from utils import TemporarilySeededRandom


def _distribute_score_over_chunks(
    response_score: Optional[float],
    chunk_spans: List,
) -> Optional[List[float]]:
    """Spread one response-level reward score across its chunks.

    Each chunk receives a share proportional to its token length, so the
    per-chunk advantages sum back to the response score. Returns None if there
    is no score (the loss then falls back to a uniform shape for this response).
    This is a detached, deterministic data-side proxy (spec sec. 7).
    """
    if response_score is None or not chunk_spans:
        return None
    total_tokens = sum(end - start for start, end in chunk_spans)
    if total_tokens <= 0:
        return None
    score = float(response_score)
    return [score * (end - start) / total_tokens for start, end in chunk_spans]


def _find_subsequence(sequence: List[int], subsequence: List[int], start: int = 0) -> Optional[int]:
    if not subsequence:
        return start
    last = len(sequence) - len(subsequence)
    for idx in range(start, last + 1):
        if sequence[idx : idx + len(subsequence)] == subsequence:
            return idx
    return None


def _response_token_bounds(
    template_text: str,
    response_text: str,
    sequence_tokens: Dict,
    raw_response_tokens: Dict,
    prompt_token_count: int,
    tokenizer,
) -> tuple[int, int]:
    """Find assistant-content token bounds inside a templated sequence."""
    if not response_text:
        return prompt_token_count, prompt_token_count

    content_char_start = template_text.rfind(response_text)
    if content_char_start >= 0:
        content_char_end = content_char_start + len(response_text)
        start = len(
            tokenizer(template_text[:content_char_start], add_special_tokens=False)["input_ids"]
        )
        end = len(tokenizer(template_text[:content_char_end], add_special_tokens=False)["input_ids"])
        start = min(max(start, prompt_token_count), len(sequence_tokens["input_ids"]))
        end = min(max(end, start), len(sequence_tokens["input_ids"]))
        return start, end

    raw_ids = raw_response_tokens["input_ids"]
    start = _find_subsequence(sequence_tokens["input_ids"], raw_ids, start=prompt_token_count)
    if start is None:
        start = prompt_token_count
    return start, min(start + len(raw_ids), len(sequence_tokens["input_ids"]))


def _response_mask(length: int, start: int, end: int) -> List[int]:
    mask = [0] * length
    for idx in range(start, end):
        mask[idx] = 1
    return mask


def get_dataset_from_hf(
    hf_dataset_repo_name: str,
    split: str,
    silent: bool = False,
    cache_dir: Optional[str] = None,
):
    data: Dict[str, Dict[str, List]] = defaultdict(lambda: defaultdict(list))
    data_iter: Iterator[Dict]

    print(f"Loading {hf_dataset_repo_name} dataset ({split} split) from HF...")
    data_iter = datasets.load_dataset(hf_dataset_repo_name, split=split, cache_dir=cache_dir)

    for example in tqdm.tqdm(data_iter, desc=f"Processing {hf_dataset_repo_name}", disable=silent):
        assert len(example["chosen"]) == 2, "(Chosen) Only support 2 turns for now"
        assert len(example["rejected"]) == 2, "(Rejected) Only support 2 turns for now"
        assert example["chosen"][:-1] == example["rejected"][:-1], (
            f"Prompt in chosen and rejected do not match: "
            f"{example['chosen'][:-1]} vs {example['rejected'][:-1]}"
        )
        prompt = example["chosen"][0]
        chosen = example["chosen"][-1]
        rejected = example["rejected"][-1]
        assert type(prompt) is dict and prompt["role"] == "user"
        assert type(chosen) is dict and chosen["role"] == "assistant"
        assert type(rejected) is dict and rejected["role"] == "assistant"

        responses = [[chosen], [rejected]]
        prompt_str = prompt["content"]

        n_responses = len(data[prompt_str]["responses"])
        data[prompt_str]["prompt_dict"] = [prompt]
        data[prompt_str]["pairs"].append((n_responses, n_responses + 1))
        data[prompt_str]["responses"].extend(responses)
        data[prompt_str]["sft_target"] = [chosen]

        # Optional response-level reward scores (e.g. ArmoRM in
        # princeton-nlp/llama3-ultrafeedback-armorm). Kept aligned with
        # `responses` so each response index has a matching score (or None).
        chosen_score = example.get("score_chosen")
        rejected_score = example.get("score_rejected")
        data[prompt_str]["response_scores"].extend([chosen_score, rejected_score])

    return data


def tokenize_batch_element(
    prompt: list[dict],
    chosen: list[dict],
    rejected: list[dict],
    tokenizer,
    max_length: int,
    max_tokens_per_chunk: int = 64,
    min_tokens_per_chunk: int = 4,
    chosen_score: Optional[float] = None,
    rejected_score: Optional[float] = None,
) -> Optional[Dict]:
    """Tokenize a single batch element"""
    assert len(prompt) == 1 and len(chosen) == 1 and len(rejected) == 1
    # Data quality check: we don't want EOS appear at the middle of the prompt or response
    raw_prompt_tokens = tokenizer(prompt[0]["content"], add_special_tokens=False)
    raw_chosen_tokens = tokenizer(chosen[0]["content"], add_special_tokens=False)
    raw_rejected_tokens = tokenizer(rejected[0]["content"], add_special_tokens=False)

    if (
        tokenizer.eos_token_id in raw_prompt_tokens["input_ids"]
        or tokenizer.eos_token_id in raw_chosen_tokens["input_ids"]
        or tokenizer.eos_token_id in raw_rejected_tokens["input_ids"]
    ):
        return None

    # assert tokenizer.eos_token_id not in raw_prompt_tokens["input_ids"], (
    #     f"Prompt contains EOS token: {prompt}"
    # )
    # assert tokenizer.eos_token_id not in raw_chosen_tokens["input_ids"], (
    #     f"Chosen response contains EOS token: {chosen}"
    # )
    # assert tokenizer.eos_token_id not in raw_rejected_tokens["input_ids"], (
    #     f"Rejected response contains EOS token: {rejected}"
    # )

    chosen_message = prompt + chosen
    rejected_message = prompt + rejected

    chosen_template_message = tokenizer.apply_chat_template(
        chosen_message, add_generation_prompt=False, tokenize=False
    )
    rejected_template_message = tokenizer.apply_chat_template(
        rejected_message, add_generation_prompt=False, tokenize=False
    )
    prompt_template_message = tokenizer.apply_chat_template(
        prompt, add_generation_prompt=False, tokenize=False
    )

    prompt_sequence_tokens = tokenizer(prompt_template_message, add_special_tokens=False)
    chosen_sequence_tokens = tokenizer(chosen_template_message, add_special_tokens=False)
    rejected_sequence_tokens = tokenizer(rejected_template_message, add_special_tokens=False)

    chosen_response_start, chosen_response_end = _response_token_bounds(
        chosen_template_message,
        chosen[0]["content"],
        chosen_sequence_tokens,
        raw_chosen_tokens,
        len(prompt_sequence_tokens["input_ids"]),
        tokenizer,
    )
    rejected_response_start, rejected_response_end = _response_token_bounds(
        rejected_template_message,
        rejected[0]["content"],
        rejected_sequence_tokens,
        raw_rejected_tokens,
        len(prompt_sequence_tokens["input_ids"]),
        tokenizer,
    )

    chunk_spans = chunk_preference_pair(
        chosen,
        rejected,
        tokenizer,
        max_tokens_per_chunk=max_tokens_per_chunk,
        min_tokens_per_chunk=min_tokens_per_chunk,
    )

    # discard the sample if too long
    longer_response_length = max(
        len(chosen_sequence_tokens["input_ids"]), len(rejected_sequence_tokens["input_ids"])
    )
    if longer_response_length > max_length:
        return None

    # Create labels (we don't want to compute loss on prompt tokens)
    chosen_sequence_tokens["labels"] = chosen_sequence_tokens["input_ids"][:]
    chosen_sequence_tokens["labels"][: len(prompt_sequence_tokens["input_ids"])] = [-100] * len(
        prompt_sequence_tokens["input_ids"]
    )
    rejected_sequence_tokens["labels"] = rejected_sequence_tokens["input_ids"][:]
    rejected_sequence_tokens["labels"][: len(prompt_sequence_tokens["input_ids"])] = [-100] * len(
        prompt_sequence_tokens["input_ids"]
    )

    batch = {}

    batch["prompt"] = prompt
    batch["chosen"] = prompt + chosen
    batch["rejected"] = prompt + rejected
    batch["chosen_response_only"] = chosen
    batch["rejected_response_only"] = rejected
    batch["chosen_chunk_spans"] = chunk_spans["chosen_chunk_spans"]
    batch["rejected_chunk_spans"] = chunk_spans["rejected_chunk_spans"]
    # Detached per-chunk advantage proxy from response-level reward scores
    # (e.g. ArmoRM). None when the dataset has no scores -> uniform fallback.
    batch["chosen_adv_proxy"] = _distribute_score_over_chunks(
        chosen_score, chunk_spans["chosen_chunk_spans"]
    )
    batch["rejected_adv_proxy"] = _distribute_score_over_chunks(
        rejected_score, chunk_spans["rejected_chunk_spans"]
    )
    batch["chosen_response_token_start"] = chosen_response_start
    batch["chosen_response_token_end"] = chosen_response_end
    batch["rejected_response_token_start"] = rejected_response_start
    batch["rejected_response_token_end"] = rejected_response_end
    batch["chosen_response_mask"] = _response_mask(
        len(chosen_sequence_tokens["input_ids"]),
        chosen_response_start,
        chosen_response_end,
    )
    batch["rejected_response_mask"] = _response_mask(
        len(rejected_sequence_tokens["input_ids"]),
        rejected_response_start,
        rejected_response_end,
    )

    for k, toks in {
        "chosen": chosen_sequence_tokens,
        "rejected": rejected_sequence_tokens,
        "prompt": prompt_sequence_tokens,
    }.items():
        for type_key, tokens in toks.items():
            if type_key == "token_type_ids":
                continue
            batch[f"{k}_{type_key}"] = tokens

    return batch


def get_collate_fn(tokenizer) -> Callable[[List[Dict]], Dict[str, Union[List, torch.Tensor]]]:
    """Returns a collate function for the given tokenizer.

    The collate function takes a list of examples (dicts, where values are lists of
      ints [tokens] or strings [the original texts]) and returns a batch of examples,
      PyTorch tensors padded to the maximum length. Strings are passed through."""

    def collate_fn(batch):
        # first, pad everything to the same length
        padded_batch = {}
        for k in batch[0].keys():
            if (
                k.endswith("_input_ids")
                or k.endswith("_attention_mask")
                or k.endswith("_labels")
                or k.endswith("_response_mask")
            ):
                if "prompt" in k:  # adapted from https://stackoverflow.com/questions/73256206
                    to_pad = [torch.LongTensor(ex[k][::-1]) for ex in batch]
                else:
                    to_pad = [torch.LongTensor(ex[k]) for ex in batch]
                if k.endswith("_input_ids"):
                    padding_value = tokenizer.pad_token_id
                elif k.endswith("_labels"):
                    padding_value = -100
                elif k.endswith("_response_mask"):
                    padding_value = 0
                elif k.endswith("_attention_mask"):
                    padding_value = 0
                else:
                    raise ValueError(f"Unexpected key in batch '{k}'")

                padded_batch[k] = pad_sequence(
                    to_pad, batch_first=True, padding_value=padding_value
                )
                if "prompt" in k:  # for the prompt, flip back so padding is on left side
                    padded_batch[k] = padded_batch[k].flip(dims=[1])
            else:
                padded_batch[k] = [ex[k] for ex in batch]

        # import ipdb; ipdb.set_trace()

        return padded_batch

    return collate_fn


def get_batch_iterator(
    hf_dataset_repo_names,
    tokenizer,
    split: str = "train",
    batch_size: int = 1,
    shuffle: bool = True,
    max_length: int = 512,
    sft_mode: bool = False,
    n_epochs: Optional[int] = None,
    n_examples: Optional[int] = None,
    seed: int = 0,
    silent: bool = False,
    cache_dir: Optional[str] = None,
    min_tokens_per_chunk: int = 4,
    max_tokens_per_chunk: int = 64,
) -> Iterator[Dict]:
    assert n_epochs is not None or n_examples is not None, (
        "Must specify either n_epochs or n_examples"
    )
    if silent:
        datasets.logging.disable_progress_bar()
        datasets.logging.set_verbosity_error()

    with TemporarilySeededRandom(seed):
        permutation_seeds = iter(np.random.randint(0, 2**32, size=1000000).tolist())
        flat_data = []

        for prompt, data in get_dataset_from_hf(
            hf_dataset_repo_names,
            split,
            silent=silent,
            cache_dir=cache_dir,
        ).items():
            flat_data.append(
                (
                    prompt,
                    data["prompt_dict"],
                    data["responses"],
                    data["pairs"],
                    data["sft_target"],
                    data.get("response_scores", []),
                )
            )

    collate_fn = get_collate_fn(tokenizer)

    epoch_idx = 0
    example_idx = 0
    done = False
    while True:
        if n_epochs is not None and epoch_idx >= n_epochs:
            if not silent:
                print(f"Finished generating {n_epochs} epochs on {split} split")
            break
        if shuffle:
            with TemporarilySeededRandom(next(permutation_seeds)):
                random.shuffle(flat_data)

        batch = []
        for (
            prompt,
            prompt_dict,
            responses,
            pairs,
            sft_target,
            response_scores,
        ) in flat_data:
            if done:
                break
            if sft_mode:
                batch_element = tokenize_batch_element(
                    prompt_dict,
                    sft_target,
                    sft_target,
                    tokenizer,
                    max_length,
                    max_tokens_per_chunk=max_tokens_per_chunk,
                    min_tokens_per_chunk=min_tokens_per_chunk,
                )
                if batch_element is None:
                    continue
                batch_element = {k: v for k, v in batch_element.items() if "rejected" not in k}
                batch.append(batch_element)
                example_idx += 1
                if len(batch) == batch_size:
                    yield collate_fn(batch)
                    if n_examples is not None and example_idx >= n_examples:
                        if not silent:
                            print(f"Finished generating {n_examples} examples on {split} split")
                        done = True

                    batch = []
            else:
                for index, p in enumerate(pairs):
                    if done:
                        break
                    chosen_score = (
                        response_scores[p[0]] if p[0] < len(response_scores) else None
                    )
                    rejected_score = (
                        response_scores[p[1]] if p[1] < len(response_scores) else None
                    )
                    batch_element = tokenize_batch_element(
                        prompt_dict,
                        responses[p[0]],
                        responses[p[1]],
                        tokenizer,
                        max_length,
                        max_tokens_per_chunk=max_tokens_per_chunk,
                        min_tokens_per_chunk=min_tokens_per_chunk,
                        chosen_score=chosen_score,
                        rejected_score=rejected_score,
                    )
                    if batch_element is None:
                        continue
                    batch.append(batch_element)
                    example_idx += 1
                    if len(batch) == batch_size:
                        yield collate_fn(batch)
                        if n_examples is not None and example_idx >= n_examples:
                            if not silent:
                                print(f"FINISHED {n_examples} EXAMPLES on {split} split")
                            done = True
                        batch = []
        if done:
            break

        epoch_idx += 1
