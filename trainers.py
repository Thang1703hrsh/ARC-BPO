import torch

torch.backends.cuda.matmul.allow_tf32 = True
import contextlib
import functools
import json
import math
import os
import random
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch.distributed as dist
import torch.nn as nn
import tqdm
import transformers
import wandb
from omegaconf import DictConfig
from torch.distributed.fsdp import (
    BackwardPrefetch,
    CPUOffload,
    MixedPrecision,
    ShardingStrategy,
    StateDictType,
)
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
)
from torch.distributed.fsdp.api import FullStateDictConfig
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers import (
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)

from loss.h_function import make_h
from loss.loss import (
    arc_bpo_pair_loss,
    bregman_loss,
    preference_loss,
    tdpo_loss,
    tisdpo_loss,
)
from loss.loss_utils import (
    Q_tbpo_get_batch_logps,
    _get_batch_logps,
    _get_batch_logps_tisdpo,
    _tdpo_get_batch_logps,
    compute_exact_chunk_log_ratios,
    compute_entropy,
    compute_kl,
    compute_response_token_logps,
)
from preference_datasets import get_batch_iterator
from utils import (
    all_gather_if_needed,
    compute_tbpo_loss_mask,
    concatenated_inputs,
    formatted_dict,
    get_block_class_from_model,
    pad_to_length,
    rank0_print,
    slice_and_move_batch_for_device,
)


def _unwrap(m):
    return m.module if hasattr(m, "module") else m


def _estimate_epoch_steps_fast(
    hf_dataset_repo_names: Union[str, List[str]],
    split: str,
    n_epochs: int,
    batch_size: int,
    cache_dir: Optional[str] = None,
    dataset_revision: Optional[str] = None,
    sft_mode: bool = False,
) -> int:
    """Cheap scheduler-step estimate without tokenizing the full dataset."""
    import datasets

    if isinstance(hf_dataset_repo_names, str):
        repo_names = [hf_dataset_repo_names]
    else:
        repo_names = list(hf_dataset_repo_names)

    total_examples = 0
    for repo_name in repo_names:
        ds = datasets.load_dataset(
            repo_name,
            split=split,
            cache_dir=cache_dir,
            revision=dataset_revision,
        )
        if sft_mode:
            # SFT groups duplicate prompts in get_dataset_from_hf, so raw rows are
            # a slight upper bound. Preference training has one pair per raw row.
            total_examples += len({example["chosen"][0]["content"] for example in ds})
        else:
            total_examples += len(ds)

    return max(1, math.ceil(total_examples * n_epochs / batch_size))


class BasicTrainer(object):
    def __init__(
        self,
        policy: nn.Module,
        config: DictConfig,
        seed: int,
        run_dir: str,
        baseline_head: Optional[nn.Module] = None,
        reference_model: Optional[nn.Module] = None,
        rank: int = 0,
        world_size: int = 1,
    ):
        """A trainer for a language model, supporting either SFT or DPO training.

        If multiple GPUs are present, naively splits the model across them, effectively
        offering N times available memory, but without any parallel computation.
        """
        self.seed = seed
        self.rank = rank
        self.world_size = world_size
        self.config = config
        self.run_dir = run_dir
        self.base_data_dir = config.base_data_dir
        self.policy_clip_hits = 0
        self.baseline_clip_hits = 0

        tokenizer_name_or_path = config.model.tokenizer_name_or_path or config.model.name_or_path
        rank0_print(f"Loading tokenizer {tokenizer_name_or_path}")
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            tokenizer_name_or_path,
            revision=getattr(config.model, "revision", None),
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        data_iterator_kwargs = dict(
            hf_dataset_repo_names=config.datasets,
            tokenizer=self.tokenizer,
            shuffle=True,
            max_length=config.max_length,
            sft_mode=config.loss.name == "sft",
            seed=seed,
            dataset_revision=getattr(config, "dataset_revision", None),
            # Deterministic chunker floor/ceiling for ARC-BPO (experiments.md: min 4 / max 64).
            min_tokens_per_chunk=int(getattr(config.loss, "min_tokens_per_chunk", 4)),
            max_tokens_per_chunk=int(getattr(config.loss, "max_tokens_per_chunk", 64)),
        )

        if self.config.n_examples is not None:
            # exact given how your iterator yields full batches
            total_steps = math.ceil(self.config.n_examples / self.config.batch_size)
        elif self.config.n_epochs is not None:
            # Avoid materializing/tokenizing the full train iterator before FSDP
            # setup. On large full-data runs, rank 0 can spend >10 minutes
            # counting while other ranks block in NCCL broadcast and time out.
            rank0_print("Estimating total_steps from raw dataset length...")
            total_steps = _estimate_epoch_steps_fast(
                self.config.datasets,
                split=self.config.dataset_train_split,
                n_epochs=self.config.n_epochs,
                batch_size=self.config.batch_size,
                cache_dir=getattr(self.config, "cache_dir", None),
                dataset_revision=getattr(self.config, "dataset_revision", None),
                sft_mode=self.config.loss.name == "sft",
            )
        else:
            raise ValueError("Need either n_examples or n_epochs to compute total_steps")
        self.total_steps = int(total_steps)
        rank0_print(f"Computed total_steps={self.total_steps}")
        self.eval_every = self.config.batch_size * 10
        configured_save_every = int(getattr(self.config, "save_every_examples", 0) or 0)
        self.save_every = configured_save_every if configured_save_every > 0 else self.eval_every * 10
        self.next_save_example = self.save_every

        self.policy = policy
        self.reference_model = reference_model
        self.baseline_head = baseline_head

        self.train_iterator = get_batch_iterator(
            **data_iterator_kwargs,
            split=config.dataset_train_split,
            n_epochs=config.n_epochs,
            n_examples=config.n_examples,
            batch_size=config.batch_size,
            silent=rank != 0,
            skip_examples=int(getattr(config, "skip_examples", 0)),
            label_noise_rate=float(getattr(config, "label_noise_rate", 0.0)),
            label_noise_seed=getattr(config, "label_noise_seed", None),
            label_noise_indices_path=getattr(config, "label_noise_indices_path", None),
        )
        rank0_print("Loaded train data iterator")
        if int(config.n_eval_examples) > 0:
            self.eval_iterator = get_batch_iterator(
                **data_iterator_kwargs,
                split=config.dataset_test_split,
                n_examples=config.n_eval_examples,
                batch_size=config.eval_batch_size,
                silent=rank != 0,
            )
            self.eval_batches = list(self.eval_iterator)
            rank0_print(
                f"Loaded {len(self.eval_batches)} eval batches of size {config.eval_batch_size}"
            )
        else:
            self.eval_iterator = None
            self.eval_batches = []
            rank0_print("Training eval disabled because n_eval_examples <= 0")

    def get_batch_samples(self, batch: Dict[str, torch.LongTensor]) -> Tuple[str, str]:
        """Generate samples from the policy (and reference model, if doing DPO training) for the given batch of inputs."""

        # FSDP generation according to https://github.com/pytorch/pytorch/issues/100069
        ctx = lambda: (
            FSDP.summon_full_params(self.policy, writeback=False, recurse=False)
            if "FSDP" in self.config.trainer
            else contextlib.nullcontext()
        )
        with ctx():
            policy_output = self.policy.generate(
                batch["prompt_input_ids"],
                attention_mask=batch["prompt_attention_mask"],
                max_length=self.config.max_length,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        if self.config.loss.name in {"dpo", "ipo", "tdpo", "tisdpo"}:
            ctx = lambda: (
                FSDP.summon_full_params(self.reference_model, writeback=False, recurse=False)
                if "FSDP" in self.config.trainer
                else contextlib.nullcontext()
            )
            with ctx():
                reference_output = self.reference_model.generate(
                    batch["prompt_input_ids"],
                    attention_mask=batch["prompt_attention_mask"],
                    max_length=self.config.max_length,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

        policy_output = pad_to_length(
            policy_output, self.config.max_length, self.tokenizer.pad_token_id
        )
        policy_output = all_gather_if_needed(policy_output, self.rank, self.world_size)
        policy_output_decoded = self.tokenizer.batch_decode(policy_output, skip_special_tokens=True)

        if self.config.loss.name in {"dpo", "ipo", "tdpo", "tisdpo"}:
            reference_output = pad_to_length(
                reference_output, self.config.max_length, self.tokenizer.pad_token_id
            )
            reference_output = all_gather_if_needed(reference_output, self.rank, self.world_size)
            reference_output_decoded = self.tokenizer.batch_decode(
                reference_output, skip_special_tokens=True
            )
        else:
            reference_output_decoded = []

        return policy_output_decoded, reference_output_decoded

    def concatenated_forward(
        self, model: nn.Module, batch: Dict[str, Union[List, torch.LongTensor]]
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.

        We do this to avoid doing two forward passes, because it's faster for FSDP.
        """

        concatenated_batch = concatenated_inputs(batch)
        # dict_keys(['concatenated_weight', 'concatenated_input_ids', 'concatenated_attention_mask', 'concatenated_labels'])
        all_logits = model(
            concatenated_batch["concatenated_input_ids"],
            attention_mask=concatenated_batch["concatenated_attention_mask"],
            use_cache=False,
        ).logits.to(torch.float32)
        all_logps = _get_batch_logps(
            all_logits,
            concatenated_batch["concatenated_labels"],
            average_log_prob=False,
            token_level=self.config.loss.token_level,
        )
        chosen_logps = all_logps[: batch["chosen_input_ids"].shape[0]]
        rejected_logps = all_logps[batch["chosen_input_ids"].shape[0] :]
        return chosen_logps, rejected_logps

    def tisdpo_concatenated_forward(
        self,
        model: nn.Module,
        reference_model: nn.Module,
        batch: Dict[str, Union[List, torch.LongTensor]],
    ):
        """Run the policy model and the reference model on the given batch of inputs, concatenating the chosen and rejected inputs together.

        We do this to avoid doing two forward passes, because it's faster for FSDP.
        """
        concatenated_batch = concatenated_inputs(batch)
        all_logits = model(
            concatenated_batch["concatenated_input_ids"],
            attention_mask=concatenated_batch["concatenated_attention_mask"],
            use_cache=False,
        ).logits.to(torch.float32)

        with torch.no_grad():
            reference_all_logits = reference_model(
                concatenated_batch["concatenated_input_ids"],
                attention_mask=concatenated_batch["concatenated_attention_mask"],
                use_cache=False,
            ).logits.to(torch.float32)

        all_logps_margin, all_position_kl, all_logps = _get_batch_logps_tisdpo(
            all_logits,
            reference_all_logits,
            concatenated_batch["concatenated_labels"],
            concatenated_batch["concatenated_weight"],
            average_log_prob=False,
        )

        chosen_logps_margin = all_logps_margin[: batch["chosen_input_ids"].shape[0]]
        rejected_logps_margin = all_logps_margin[batch["chosen_input_ids"].shape[0] :]
        chosen_position_kl = all_position_kl[: batch["chosen_input_ids"].shape[0]]
        rejected_position_kl = all_position_kl[batch["chosen_input_ids"].shape[0] :]

        chosen_logps = all_logps[: batch["chosen_input_ids"].shape[0]].detach()
        rejected_logps = all_logps[batch["chosen_input_ids"].shape[0] :].detach()

        return (
            chosen_logps_margin,
            rejected_logps_margin,
            chosen_position_kl,
            rejected_position_kl,
            chosen_logps,
            rejected_logps,
        )

    def tdpo_concatenated_forward(
        self,
        model: nn.Module,
        reference_model: nn.Module,
        batch: Dict[str, Union[List, torch.LongTensor]],
    ):
        """Run the policy model and the reference model on the given batch of inputs, concatenating the chosen and rejected inputs together.

        We do this to avoid doing two forward passes, because it's faster for FSDP.
        """
        concatenated_batch = concatenated_inputs(batch)
        all_logits = model(
            concatenated_batch["concatenated_input_ids"],
            attention_mask=concatenated_batch["concatenated_attention_mask"],
            use_cache=False,
        ).logits.to(torch.float32)

        with torch.no_grad():
            reference_all_logits = reference_model(
                concatenated_batch["concatenated_input_ids"],
                attention_mask=concatenated_batch["concatenated_attention_mask"],
                use_cache=False,
            ).logits.to(torch.float32)
        all_logps_margin, all_position_kl, all_logps = _tdpo_get_batch_logps(
            all_logits,
            reference_all_logits,
            concatenated_batch["concatenated_labels"],
            average_log_prob=False,
        )

        chosen_logps_margin = all_logps_margin[: batch["chosen_input_ids"].shape[0]]
        rejected_logps_margin = all_logps_margin[batch["chosen_input_ids"].shape[0] :]
        chosen_position_kl = all_position_kl[: batch["chosen_input_ids"].shape[0]]
        rejected_position_kl = all_position_kl[batch["chosen_input_ids"].shape[0] :]

        chosen_logps = all_logps[: batch["chosen_input_ids"].shape[0]].detach()
        rejected_logps = all_logps[batch["chosen_input_ids"].shape[0] :].detach()

        return (
            chosen_logps_margin,
            rejected_logps_margin,
            chosen_position_kl,
            rejected_position_kl,
            chosen_logps,
            rejected_logps,
        )

    def Q_tbpo_concatenated_forward(
        self,
        model: nn.Module,
        reference_model: nn.Module,
        batch: Dict[str, Union[List, torch.LongTensor]],
    ):
        """
        Compute R_theta
        """
        assert self.baseline_head is not None, "Q_tbpo requires baseline_head"
        concatenated_batch = concatenated_inputs(batch)

        # Use forward hook to capture last hidden state without storing all intermediate layers
        # This avoids storing all 32 intermediate hidden states (~17GB memory savings)
        # The hook captures the output of the base model (MistralModel) before lm_head
        last_hidden_state_captured = None

        def capture_last_hidden_state(module, input, output):
            nonlocal last_hidden_state_captured
            # output is BaseModelOutputWithPast, output[0] is last_hidden_state
            last_hidden_state_captured = output[0]

        # Register hook on the base transformer (the module that emits last_hidden_state).
        # The attribute path differs by wrapping:
        #   - PEFT (LoRA):  PeftModel -> LoraModel -> CausalLM -> transformer  (3 levels)
        #   - Full FT:                                CausalLM -> transformer  (1 level)
        # FSDP forwards attribute access to the wrapped module, so it is transparent.
        def _get_base_transformer(m):
            if hasattr(m, "get_base_model"):
                m = m.get_base_model()  # unwrap PEFT to underlying AutoModelForCausalLM
            return m.model  # CausalLM exposes the transformer as `.model`

        hook_handle = _get_base_transformer(model).register_forward_hook(
            capture_last_hidden_state
        )

        try:
            # Forward pass with output_hidden_states=False to avoid storing all layers
            outputs = model(
                concatenated_batch["concatenated_input_ids"],
                attention_mask=concatenated_batch["concatenated_attention_mask"],
                output_hidden_states=False,  # Don't store all 32 intermediate layers
                use_cache=False,
            )
            all_logits = outputs.logits.to(torch.float32)
            all_last_hidden_states = last_hidden_state_captured.to(torch.float32)
        finally:
            # Always remove the hook
            hook_handle.remove()

        with torch.no_grad():
            reference_all_logits = reference_model(
                concatenated_batch["concatenated_input_ids"],
                attention_mask=concatenated_batch["concatenated_attention_mask"],
                use_cache=False,
            ).logits.to(torch.float32)

        loss_mask = compute_tbpo_loss_mask(batch, concatenated_batch)

        all_logps_margin, all_logps = Q_tbpo_get_batch_logps(
            all_logits,
            reference_all_logits,
            concatenated_batch["concatenated_labels"],
        )

        chosen_logps_margin = all_logps_margin[: batch["chosen_input_ids"].shape[0]]
        rejected_logps_margin = all_logps_margin[batch["chosen_input_ids"].shape[0] :]

        all_last_hidden_states_detached = all_last_hidden_states.detach()
        all_baselines = self.baseline_head(all_last_hidden_states_detached)
        chosen_baselines = all_baselines[: batch["chosen_input_ids"].shape[0]]
        rejected_baselines = all_baselines[batch["chosen_input_ids"].shape[0] :]
        # align to token-prediction positions: logits[:, :-1] predicts labels[:, 1:]
        b_chosen = chosen_baselines[:, :-1]
        b_rejected = rejected_baselines[:, :-1]
        log_w = b_rejected - b_chosen
        mean_logw_per_prompt = (log_w * loss_mask).sum(-1) / loss_mask.sum(-1)
        # center w
        log_w = log_w - mean_logw_per_prompt.unsqueeze(-1)
        log_w = log_w.clamp(self.config.model.baseline_l, self.config.model.baseline_u)

        # R_theta
        beta = self.config.loss.beta
        log_R = beta * (rejected_logps_margin - chosen_logps_margin + log_w)

        token_chosen_logps = beta * all_logps[: batch["chosen_input_ids"].shape[0]].detach()
        token_rejected_logps = beta * all_logps[batch["chosen_input_ids"].shape[0] :].detach()
        token_chosen_rewards = beta * (token_chosen_logps + b_chosen)
        token_rejected_rewards = beta * (token_rejected_logps + b_rejected)

        chosen_logps = token_chosen_logps.sum(-1)
        rejected_logps = token_rejected_logps.sum(-1)

        # Token-level diagnostics (policy entropy and KL drift vs reference) over the
        # same token positions used by the TBPO loss mask (min(|yw|,|yl|)).
        batch_size = batch["chosen_input_ids"].shape[0]
        chosen_logits = all_logits[:batch_size].detach()
        chosen_reference_logits = reference_all_logits[:batch_size].detach()
        chosen_labels = concatenated_batch["concatenated_labels"][:batch_size]

        denom = loss_mask.sum(-1).clamp_min(1)

        entropy, _ = compute_entropy(chosen_logits, chosen_labels)
        entropy_per_prompt = (entropy * loss_mask).sum(-1) / denom  # (batch_size,)

        kl_forward, _ = compute_kl(
            chosen_logits, chosen_reference_logits, chosen_labels, direction="ref_to_policy"
        )
        kl_forward_per_prompt = (kl_forward * loss_mask).sum(-1) / denom  # (batch_size,)

        kl_reverse, _ = compute_kl(
            chosen_logits, chosen_reference_logits, chosen_labels, direction="policy_to_ref"
        )
        kl_reverse_per_prompt = (kl_reverse * loss_mask).sum(-1) / denom  # (batch_size,)

        return (
            log_R,
            chosen_logps,
            rejected_logps,
            token_chosen_rewards,
            token_rejected_rewards,
            loss_mask,
            entropy_per_prompt,
            kl_forward_per_prompt,
            kl_reverse_per_prompt,
        )

    def A_tbpo_concatenated_forward(
        self,
        model: nn.Module,
        reference_model: nn.Module,
        batch: Dict[str, Union[List, torch.LongTensor]],
    ):
        """Compute R_theta"""

        concatenated_batch = concatenated_inputs(batch)
        all_logits = model(
            concatenated_batch["concatenated_input_ids"],
            attention_mask=concatenated_batch["concatenated_attention_mask"],
            use_cache=False,
        ).logits.to(torch.float32)
        with torch.no_grad():
            reference_all_logits = reference_model(
                concatenated_batch["concatenated_input_ids"],
                attention_mask=concatenated_batch["concatenated_attention_mask"],
                use_cache=False,
            ).logits.to(torch.float32)

        loss_mask = compute_tbpo_loss_mask(batch, concatenated_batch)

        all_logps_margin, all_logps = Q_tbpo_get_batch_logps(
            all_logits,
            reference_all_logits,
            concatenated_batch["concatenated_labels"],
        )
        all_position_kl, _ = compute_kl(
            all_logits,
            reference_all_logits,
            concatenated_batch["concatenated_labels"],
            direction="ref_to_policy",
        )
        batch_size = batch["chosen_input_ids"].shape[0]

        # Split chosen/rejected
        chosen_logps_margin = all_logps_margin[:batch_size]
        rejected_logps_margin = all_logps_margin[batch_size:]
        chosen_position_kl = all_position_kl[:batch_size]
        rejected_position_kl = all_position_kl[batch_size:]

        # Compute log R_θ using advantage formulation
        # log R = β * ((Δlogps_l - Δlogps_w) + (KL_l - KL_w))
        # Note: rejected_logps_margin - chosen_logps_margin is the odds ratio direction
        beta = self.config.loss.beta
        delta_logps = rejected_logps_margin - chosen_logps_margin
        delta_kl = rejected_position_kl - chosen_position_kl
        log_R = beta * (delta_logps + delta_kl)

        # Token-level log probabilities for rewards (detached)
        token_chosen_logps = all_logps[:batch_size].detach()
        token_rejected_logps = all_logps[batch_size:].detach()

        # Token-level rewards: advantage = logps_margin + kl
        # Scale by beta for consistency with Q_tbpo
        token_chosen_rewards = beta * (chosen_logps_margin.detach() + chosen_position_kl.detach())
        token_rejected_rewards = beta * (
            rejected_logps_margin.detach() + rejected_position_kl.detach()
        )

        # Sequence-level log probabilities (sum over tokens, masked)
        chosen_logps = (token_chosen_logps * loss_mask).sum(-1)
        rejected_logps = (token_rejected_logps * loss_mask).sum(-1)

        # Token-level diagnostics
        chosen_logits = all_logits[:batch_size].detach()
        chosen_reference_logits = reference_all_logits[:batch_size].detach()
        chosen_labels = concatenated_batch["concatenated_labels"][:batch_size]

        denom = loss_mask.sum(-1).clamp_min(1)

        entropy, _ = compute_entropy(chosen_logits, chosen_labels)
        entropy_per_prompt = (entropy * loss_mask).sum(-1) / denom

        kl_forward, _ = compute_kl(
            chosen_logits, chosen_reference_logits, chosen_labels, direction="ref_to_policy"
        )
        kl_forward_per_prompt = (kl_forward * loss_mask).sum(-1) / denom

        kl_reverse, _ = compute_kl(
            chosen_logits, chosen_reference_logits, chosen_labels, direction="policy_to_ref"
        )
        kl_reverse_per_prompt = (kl_reverse * loss_mask).sum(-1) / denom

        # Also return the per-position KL for logging
        kl_chosen_per_prompt = (chosen_position_kl * loss_mask).sum(-1) / denom
        kl_rejected_per_prompt = (rejected_position_kl * loss_mask).sum(-1) / denom

        return (
            log_R,
            chosen_logps,
            rejected_logps,
            token_chosen_rewards,
            token_rejected_rewards,
            loss_mask,
            entropy_per_prompt,
            kl_forward_per_prompt,
            kl_reverse_per_prompt,
            kl_chosen_per_prompt,
            kl_rejected_per_prompt,
        )

    def BPO_SBA_concatenated_forward(
        self,
        model: nn.Module,
        reference_model: nn.Module,
        batch: Dict[str, Union[List, torch.LongTensor]],
    ):
        """Compute sequence-level log R_θ for BPO-SBA loss.

        This implements the pure BPO-SBA(λ) loss from the paper:
        R_θ(x, y_w, y_l) = [π_θ(y_l|x)π_ref(y_w|x) / π_θ(y_w|x)π_ref(y_l|x)]^β

        Returns sequence-level log_R to plug directly into the h function.
        """
        concatenated_batch = concatenated_inputs(batch)

        # Forward pass through policy
        all_logits = model(
            concatenated_batch["concatenated_input_ids"],
            attention_mask=concatenated_batch["concatenated_attention_mask"],
            use_cache=False,
        ).logits.to(torch.float32)

        # Forward pass through reference (no grad)
        with torch.no_grad():
            reference_all_logits = reference_model(
                concatenated_batch["concatenated_input_ids"],
                attention_mask=concatenated_batch["concatenated_attention_mask"],
                use_cache=False,
            ).logits.to(torch.float32)

        # Get sequence-level log probabilities (sum over all tokens)
        policy_all_logps = _get_batch_logps(
            all_logits,
            concatenated_batch["concatenated_labels"],
            average_log_prob=False,
            token_level=False,
        )
        reference_all_logps = _get_batch_logps(
            reference_all_logits,
            concatenated_batch["concatenated_labels"],
            average_log_prob=False,
            token_level=False,
        )

        batch_size = batch["chosen_input_ids"].shape[0]

        # Split into chosen/rejected
        policy_chosen_logps = policy_all_logps[:batch_size]
        policy_rejected_logps = policy_all_logps[batch_size:]
        reference_chosen_logps = reference_all_logps[:batch_size]
        reference_rejected_logps = reference_all_logps[batch_size:]

        # Sequence-level log probability margins: log π_θ(y|x) - log π_ref(y|x)
        chosen_logps_margin = policy_chosen_logps - reference_chosen_logps
        rejected_logps_margin = policy_rejected_logps - reference_rejected_logps

        # Sequence-level log_R = β * (rejected_margin - chosen_margin)
        # This is log of: R_θ = [π_θ(y_l|x)π_ref(y_w|x) / π_θ(y_w|x)π_ref(y_l|x)]^β
        beta = self.config.loss.beta
        log_R = beta * (rejected_logps_margin - chosen_logps_margin)  # (batch,)
        chosen_rewards = beta * chosen_logps_margin.detach()
        rejected_rewards = beta * rejected_logps_margin.detach()

        return (
            log_R,
            policy_chosen_logps,
            policy_rejected_logps,
            chosen_rewards,
            rejected_rewards,
        )

    def arc_bpo_concatenated_forward(
        self,
        model: nn.Module,
        reference_model: nn.Module,
        batch: Dict[str, Union[List, torch.LongTensor]],
        loss_config: DictConfig,
    ):
        """Compute ARC-BPO one-sided chunk losses without TBPO token matching."""
        concatenated_batch = concatenated_inputs(batch)

        all_logits = model(
            concatenated_batch["concatenated_input_ids"],
            attention_mask=concatenated_batch["concatenated_attention_mask"],
            use_cache=False,
        ).logits.to(torch.float32)

        with torch.no_grad():
            reference_all_logits = reference_model(
                concatenated_batch["concatenated_input_ids"],
                attention_mask=concatenated_batch["concatenated_attention_mask"],
                use_cache=False,
            ).logits.to(torch.float32)

        policy_token_logps, response_mask = compute_response_token_logps(
            all_logits,
            concatenated_batch["concatenated_labels"],
            concatenated_batch["concatenated_response_mask"],
        )
        reference_token_logps, _ = compute_response_token_logps(
            reference_all_logits,
            concatenated_batch["concatenated_labels"],
            concatenated_batch["concatenated_response_mask"],
        )

        batch_size = batch["chosen_input_ids"].shape[0]
        losses = []
        outputs = []
        policy_chosen_logps = []
        policy_rejected_logps = []

        delta_star_cfg = getattr(loss_config, "delta_star", 2.0)
        if torch.is_tensor(delta_star_cfg) and delta_star_cfg.numel() > 1:
            delta_star_values = list(delta_star_cfg)
        elif isinstance(delta_star_cfg, (list, tuple)):
            delta_star_values = delta_star_cfg
        else:
            delta_star_values = [delta_star_cfg] * batch_size

        chosen_advantages = batch.get("chosen_adv_proxy", [None] * batch_size)
        rejected_advantages = batch.get("rejected_adv_proxy", [None] * batch_size)

        for idx in range(batch_size):
            chosen_idx = idx
            rejected_idx = idx + batch_size

            chosen_policy = policy_token_logps[chosen_idx][response_mask[chosen_idx]]
            chosen_reference = reference_token_logps[chosen_idx][response_mask[chosen_idx]]
            rejected_policy = policy_token_logps[rejected_idx][response_mask[rejected_idx]]
            rejected_reference = reference_token_logps[rejected_idx][response_mask[rejected_idx]]

            chosen_chunk_log_ratios, chosen_seq_log_ratio = compute_exact_chunk_log_ratios(
                chosen_policy,
                chosen_reference,
                batch["chosen_chunk_spans"][idx],
                beta=loss_config.beta,
            )
            rejected_chunk_log_ratios, rejected_seq_log_ratio = compute_exact_chunk_log_ratios(
                rejected_policy,
                rejected_reference,
                batch["rejected_chunk_spans"][idx],
                beta=loss_config.beta,
            )

            pair_loss, pair_outputs = arc_bpo_pair_loss(
                chosen_chunk_log_ratios,
                rejected_chunk_log_ratios,
                delta_star_values[idx],
                chosen_advantages=chosen_advantages[idx],
                rejected_advantages=rejected_advantages[idx],
                temperature=float(getattr(loss_config, "T", 2.0)),
                kappa=float(getattr(loss_config, "kappa", 2.0)),
                use_advantage_shape=bool(getattr(loss_config, "use_advantage_shape", False)),
                fallback_to_uniform=bool(getattr(loss_config, "fallback_to_uniform_shape", True)),
                winsorize_advantages=bool(
                    getattr(loss_config, "winsorize_advantages", True)
                ),
                lam=float(getattr(loss_config, "sba_lambda", 1.0)),
                s=float(getattr(loss_config, "sba_scale", 4.0)),
                exp_clip=float(getattr(loss_config, "exp_clip", 30.0)),
            )
            pair_outputs["chosen_seq_log_ratio"] = chosen_seq_log_ratio.detach()
            pair_outputs["rejected_seq_log_ratio"] = rejected_seq_log_ratio.detach()
            pair_outputs["chosen_num_chunks"] = torch.tensor(
                chosen_chunk_log_ratios.numel(), device=chosen_policy.device, dtype=torch.float32
            )
            pair_outputs["rejected_num_chunks"] = torch.tensor(
                rejected_chunk_log_ratios.numel(), device=chosen_policy.device, dtype=torch.float32
            )

            losses.append(pair_loss)
            outputs.append(pair_outputs)
            policy_chosen_logps.append(chosen_policy.sum())
            policy_rejected_logps.append(rejected_policy.sum())

        return (
            torch.stack(losses),
            torch.stack(policy_chosen_logps),
            torch.stack(policy_rejected_logps),
            outputs,
        )

    def get_batch_metrics(
        self, batch: Dict[str, Union[List, torch.LongTensor]], loss_config: DictConfig, train=True
    ):
        metrics = {}
        train_test = "train" if train else "eval"

        if loss_config.name in {"dpo", "ipo"}:
            policy_chosen_logps, policy_rejected_logps = self.concatenated_forward(
                self.policy, batch
            )
            with torch.no_grad():
                reference_chosen_logps, reference_rejected_logps = self.concatenated_forward(
                    self.reference_model, batch
                )

            if loss_config.name == "dpo":
                loss_kwargs = {
                    "beta": loss_config.beta,
                    "reference_free": loss_config.reference_free,
                    "label_smoothing": loss_config.label_smoothing,
                    "ipo": False,
                }
            elif loss_config.name == "ipo":
                loss_kwargs = {"beta": loss_config.beta, "ipo": True}
            else:
                raise ValueError(f"unknown loss {loss_config.name}")

            losses, chosen_rewards, rejected_rewards = preference_loss(
                policy_chosen_logps,
                policy_rejected_logps,
                reference_chosen_logps,
                reference_rejected_logps,
                **loss_kwargs,
            )

            reward_accuracies = (chosen_rewards > rejected_rewards).float()

            chosen_rewards = all_gather_if_needed(chosen_rewards, self.rank, self.world_size)
            rejected_rewards = all_gather_if_needed(rejected_rewards, self.rank, self.world_size)
            reward_accuracies = all_gather_if_needed(reward_accuracies, self.rank, self.world_size)

            metrics[f"rewards_{train_test}/chosen"] = chosen_rewards.cpu().numpy().tolist()
            metrics[f"rewards_{train_test}/rejected"] = rejected_rewards.cpu().numpy().tolist()
            metrics[f"rewards_{train_test}/accuracies"] = reward_accuracies.cpu().numpy().tolist()
            metrics[f"rewards_{train_test}/margins"] = (
                (chosen_rewards - rejected_rewards).cpu().numpy().tolist()
            )

            policy_rejected_logps = all_gather_if_needed(
                policy_rejected_logps.detach(), self.rank, self.world_size
            )
            metrics[f"logps_{train_test}/rejected"] = policy_rejected_logps.cpu().numpy().tolist()
        elif loss_config.name == "tdpo":
            (
                chosen_logps_margin,
                rejected_logps_margin,
                chosen_position_kl,
                rejected_position_kl,
                policy_chosen_logps,
                policy_rejected_logps,
            ) = self.tdpo_concatenated_forward(self.policy, self.reference_model, batch)
            losses, chosen_rewards, rejected_rewards = tdpo_loss(
                chosen_logps_margin,
                rejected_logps_margin,
                chosen_position_kl,
                rejected_position_kl,
                beta=loss_config.beta,
                alpha=loss_config.alpha,
                if_tdpo2=loss_config.if_tdpo2,
            )

            reward_accuracies = (chosen_rewards > rejected_rewards).float()

            chosen_rewards = all_gather_if_needed(chosen_rewards, self.rank, self.world_size)
            rejected_rewards = all_gather_if_needed(rejected_rewards, self.rank, self.world_size)
            reward_accuracies = all_gather_if_needed(reward_accuracies, self.rank, self.world_size)

            metrics[f"rewards_{train_test}/chosen"] = chosen_rewards.cpu().numpy().tolist()
            metrics[f"rewards_{train_test}/rejected"] = rejected_rewards.cpu().numpy().tolist()
            metrics[f"rewards_{train_test}/accuracies"] = reward_accuracies.cpu().numpy().tolist()
            metrics[f"rewards_{train_test}/margins"] = (
                (chosen_rewards - rejected_rewards).cpu().numpy().tolist()
            )

            all_device_chosen_position_kl = all_gather_if_needed(
                chosen_position_kl.detach(), self.rank, self.world_size
            )
            all_device_rejected_position_kl = all_gather_if_needed(
                rejected_position_kl.detach(), self.rank, self.world_size
            )

            metrics[f"kl_{train_test}/chosen"] = (
                all_device_chosen_position_kl.cpu().numpy().tolist()
            )
            metrics[f"kl_{train_test}/rejected"] = (
                all_device_rejected_position_kl.cpu().numpy().tolist()
            )
            metrics[f"kl_{train_test}/margin"] = (
                (all_device_chosen_position_kl - all_device_rejected_position_kl)
                .cpu()
                .numpy()
                .tolist()
            )

            policy_rejected_logps = all_gather_if_needed(
                policy_rejected_logps.detach(), self.rank, self.world_size
            )
            metrics[f"logps_{train_test}/rejected"] = policy_rejected_logps.cpu().numpy().tolist()
        elif loss_config.name == "tisdpo":
            (
                chosen_logps_margin,
                rejected_logps_margin,
                chosen_position_kl,
                rejected_position_kl,
                policy_chosen_logps,
                policy_rejected_logps,
            ) = self.tisdpo_concatenated_forward(self.policy, self.reference_model, batch)
            losses, chosen_rewards, rejected_rewards = tisdpo_loss(
                chosen_logps_margin,
                rejected_logps_margin,
                chosen_position_kl,
                rejected_position_kl,
                beta=loss_config.beta,
                alpha=loss_config.alpha,
                token_level=loss_config.token_level,
            )

            reward_accuracies = (chosen_rewards > rejected_rewards).float()

            chosen_rewards = all_gather_if_needed(chosen_rewards, self.rank, self.world_size)
            rejected_rewards = all_gather_if_needed(rejected_rewards, self.rank, self.world_size)
            reward_accuracies = all_gather_if_needed(reward_accuracies, self.rank, self.world_size)

            metrics[f"rewards_{train_test}/chosen"] = chosen_rewards.cpu().numpy().tolist()
            metrics[f"rewards_{train_test}/rejected"] = rejected_rewards.cpu().numpy().tolist()
            metrics[f"rewards_{train_test}/accuracies"] = reward_accuracies.cpu().numpy().tolist()
            metrics[f"rewards_{train_test}/margins"] = (
                (chosen_rewards - rejected_rewards).cpu().numpy().tolist()
            )

            all_device_chosen_position_kl = all_gather_if_needed(
                chosen_position_kl.detach(), self.rank, self.world_size
            )
            all_device_rejected_position_kl = all_gather_if_needed(
                rejected_position_kl.detach(), self.rank, self.world_size
            )

            metrics[f"kl_{train_test}/chosen"] = (
                all_device_chosen_position_kl.cpu().numpy().tolist()
            )
            metrics[f"kl_{train_test}/rejected"] = (
                all_device_rejected_position_kl.cpu().numpy().tolist()
            )
            metrics[f"kl_{train_test}/margin"] = (
                (all_device_chosen_position_kl - all_device_rejected_position_kl)
                .cpu()
                .numpy()
                .tolist()
            )

            policy_rejected_logps = all_gather_if_needed(
                policy_rejected_logps.detach(), self.rank, self.world_size
            )
            metrics[f"logps_{train_test}/rejected"] = policy_rejected_logps.cpu().numpy().tolist()

        elif loss_config.name == "sft":
            policy_chosen_logits = self.policy(
                batch["chosen_input_ids"],
                attention_mask=batch["chosen_attention_mask"],
                use_cache=False,
            ).logits.to(torch.float32)
            policy_chosen_logps = _get_batch_logps(
                policy_chosen_logits,
                batch["chosen_labels"],
                average_log_prob=False,
                token_level=False,
            )

            losses = -policy_chosen_logps

        elif loss_config.name == "arc_bpo":
            (
                losses,
                policy_chosen_logps,
                policy_rejected_logps,
                arc_outputs,
            ) = self.arc_bpo_concatenated_forward(
                self.policy,
                self.reference_model,
                batch,
                loss_config,
            )

            policy_rejected_logps = all_gather_if_needed(
                policy_rejected_logps.detach(), self.rank, self.world_size
            )
            metrics[f"logps_{train_test}/rejected"] = policy_rejected_logps.cpu().numpy().tolist()

            delta_star = torch.as_tensor(
                getattr(loss_config, "delta_star", 2.0),
                device=losses.device,
                dtype=losses.dtype,
            )
            for metric_name in (
                "target_margin",
                "model_margin",
                "chosen_seq_log_ratio",
                "rejected_seq_log_ratio",
                "chosen_num_chunks",
                "rejected_num_chunks",
            ):
                values = torch.stack([o[metric_name].detach() for o in arc_outputs])
                values = all_gather_if_needed(values, self.rank, self.world_size)
                metrics[f"arc_bpo_{train_test}/{metric_name}"] = values.cpu().numpy().tolist()

            calibration_errors = torch.stack(
                [(o["target_margin"] - delta_star).abs().detach() for o in arc_outputs]
            )
            calibration_errors = all_gather_if_needed(
                calibration_errors, self.rank, self.world_size
            )
            metrics[f"arc_bpo_{train_test}/target_calibration_error"] = (
                calibration_errors.cpu().numpy().tolist()
            )

        elif loss_config.name == "BPO_SBA":
            # BPO-SBA(λ): Sequence-level BPO with SBA divergence
            # This uses pure sequence-level odds ratio matching from the paper.
            (
                log_R,
                policy_chosen_logps,
                policy_rejected_logps,
                chosen_rewards,
                rejected_rewards,
            ) = self.BPO_SBA_concatenated_forward(self.policy, self.reference_model, batch)

            # Apply h function directly to sequence-level log_R
            h_func = make_h(**loss_config.bregman_loss)
            per_sample_loss = h_func.loss_from_logR(log_R)  # (batch,)

            # Apply label smoothing if specified
            label_smoothing = float(getattr(loss_config, "label_smoothing", 0.0))
            if label_smoothing > 0:
                per_sample_loss_flipped = h_func.loss_from_logR(-log_R)
                per_sample_loss = (
                    per_sample_loss * (1 - label_smoothing)
                    + per_sample_loss_flipped * label_smoothing
                )

            losses = per_sample_loss

            reward_accuracies = (chosen_rewards > rejected_rewards).float()
            chosen_rewards = all_gather_if_needed(chosen_rewards, self.rank, self.world_size)
            rejected_rewards = all_gather_if_needed(rejected_rewards, self.rank, self.world_size)
            reward_accuracies = all_gather_if_needed(
                reward_accuracies, self.rank, self.world_size
            )

            metrics[f"rewards_{train_test}/chosen"] = chosen_rewards.cpu().numpy().tolist()
            metrics[f"rewards_{train_test}/rejected"] = rejected_rewards.cpu().numpy().tolist()
            metrics[f"rewards_{train_test}/accuracies"] = reward_accuracies.cpu().numpy().tolist()
            metrics[f"rewards_{train_test}/margins"] = (
                (chosen_rewards - rejected_rewards).cpu().numpy().tolist()
            )

            policy_rejected_logps = all_gather_if_needed(
                policy_rejected_logps.detach(), self.rank, self.world_size
            )

            # ---- log_R statistics ----
            log_R_gathered = all_gather_if_needed(log_R.detach(), self.rank, self.world_size)

            metrics[f"log_R_{train_test}/mean"] = log_R_gathered.mean().item()
            metrics[f"log_R_{train_test}/std"] = log_R_gathered.std(unbiased=False).item()
            metrics[f"log_R_{train_test}/max"] = log_R_gathered.max().item()
            metrics[f"log_R_{train_test}/min"] = log_R_gathered.min().item()
            metrics[f"log_R_{train_test}/frac_lt_0"] = (log_R_gathered < 0).float().mean().item()
            metrics[f"log_R_{train_test}/frac_gt_0"] = (log_R_gathered > 0).float().mean().item()
            metrics[f"log_R_{train_test}/values"] = log_R_gathered.cpu().numpy().tolist()

            # Loss statistics
            losses_gathered = all_gather_if_needed(losses.detach(), self.rank, self.world_size)
            metrics[f"loss_{train_test}/per_sample"] = losses_gathered.cpu().numpy().tolist()

            metrics[f"logps_{train_test}/rejected"] = policy_rejected_logps.cpu().numpy().tolist()

        elif loss_config.name in {"Q_tbpo", "A_tbpo"}:
            # Unified handling for Q_tbpo and A_tbpo
            # Difference: Q_tbpo uses baseline head, A_tbpo uses KL divergence
            kl_chosen_per_prompt = None
            kl_rejected_per_prompt = None

            if loss_config.name == "Q_tbpo":
                (
                    log_R,
                    policy_chosen_logps,
                    policy_rejected_logps,
                    token_chosen_rewards,
                    token_rejected_rewards,
                    loss_mask,
                    entropy_per_prompt,
                    kl_forward_per_prompt,
                    kl_reverse_per_prompt,
                ) = self.Q_tbpo_concatenated_forward(self.policy, self.reference_model, batch)
            else:  # A_tbpo
                (
                    log_R,
                    policy_chosen_logps,
                    policy_rejected_logps,
                    token_chosen_rewards,
                    token_rejected_rewards,
                    loss_mask,
                    entropy_per_prompt,
                    kl_forward_per_prompt,
                    kl_reverse_per_prompt,
                    kl_chosen_per_prompt,
                    kl_rejected_per_prompt,
                ) = self.A_tbpo_concatenated_forward(self.policy, self.reference_model, batch)

            label_smoothing = float(getattr(loss_config, "label_smoothing", 0.0))
            losses, per_token_loss = bregman_loss(
                log_R,
                loss_mask,
                h_func=make_h(**loss_config.bregman_loss),
                label_smoothing=label_smoothing,
            )

            policy_rejected_logps = all_gather_if_needed(
                policy_rejected_logps.detach(), self.rank, self.world_size
            )

            # ---- log_R statistics (masked) ----
            # Gather across ranks to get consistent stats in multi-GPU runs.
            log_R_gathered = all_gather_if_needed(log_R.detach(), self.rank, self.world_size)
            loss_mask_gathered = all_gather_if_needed(loss_mask, self.rank, self.world_size).bool()
            min_lengths_gathered = loss_mask_gathered.sum(dim=-1)

            # Per-sequence masked mean of log_R
            log_R_per_seq = (log_R_gathered * loss_mask_gathered).sum(
                -1
            ) / min_lengths_gathered.clamp_min(1)
            metrics[f"log_R_{train_test}/mean_per_sequence"] = log_R_per_seq.cpu().numpy().tolist()

            # Summary stats across sequences (distribution of per-seq means)
            metrics[f"log_R_{train_test}/mean_per_sequence_median"] = log_R_per_seq.median().item()
            metrics[f"log_R_{train_test}/mean_per_sequence_p10"] = log_R_per_seq.quantile(
                0.1
            ).item()
            metrics[f"log_R_{train_test}/mean_per_sequence_p90"] = log_R_per_seq.quantile(
                0.9
            ).item()

            # Masked token-level stats over all (unmasked) tokens in the gathered batch
            log_R_tokens = log_R_gathered[loss_mask_gathered]
            if log_R_tokens.numel() > 0:
                metrics[f"log_R_{train_test}/mean"] = log_R_tokens.mean().item()
                metrics[f"log_R_{train_test}/std"] = log_R_tokens.std(unbiased=False).item()
                metrics[f"log_R_{train_test}/max"] = log_R_tokens.max().item()
                metrics[f"log_R_{train_test}/min"] = log_R_tokens.min().item()
                metrics[f"log_R_{train_test}/frac_lt_0"] = (log_R_tokens < 0).float().mean().item()
                metrics[f"log_R_{train_test}/frac_gt_0"] = (log_R_tokens > 0).float().mean().item()
            else:
                metrics[f"log_R_{train_test}/mean"] = 0.0
                metrics[f"log_R_{train_test}/std"] = 0.0
                metrics[f"log_R_{train_test}/max"] = 0.0
                metrics[f"log_R_{train_test}/min"] = 0.0
                metrics[f"log_R_{train_test}/frac_lt_0"] = 0.0
                metrics[f"log_R_{train_test}/frac_gt_0"] = 0.0

            # Log minimum completion length distribution (T = min(|yw|, |yl|) per pair)
            metrics[f"min_lengths_{train_test}/values"] = (
                min_lengths_gathered.cpu().numpy().tolist()
            )

            # Per-sequence token-level loss statistics
            per_token_loss_gathered = all_gather_if_needed(
                per_token_loss.detach(), self.rank, self.world_size
            )

            # For each sequence, compute statistics over its tokens (excluding masked positions)
            per_seq_stats = []
            for seq_idx in range(per_token_loss_gathered.shape[0]):
                seq_mask = loss_mask_gathered[seq_idx]
                seq_losses = per_token_loss_gathered[seq_idx][seq_mask.bool()]

                if seq_losses.numel() > 0:
                    per_seq_stats.append(
                        {
                            "mean": seq_losses.mean().item(),
                            "median": seq_losses.median().item(),
                            "p90": seq_losses.quantile(0.9).item(),
                        }
                    )
                else:
                    per_seq_stats.append({"mean": 0.0, "median": 0.0, "p90": 0.0})

            # Extract lists for each statistic
            metrics[f"per_token_loss_{train_test}/mean_per_sequence"] = [
                s["mean"] for s in per_seq_stats
            ]
            metrics[f"per_token_loss_{train_test}/median_per_sequence"] = [
                s["median"] for s in per_seq_stats
            ]
            metrics[f"per_token_loss_{train_test}/p90_per_sequence"] = [
                s["p90"] for s in per_seq_stats
            ]

            # Per-sequence token reward statistics (mask already applied in reward computation)
            token_chosen_rewards_gathered = all_gather_if_needed(
                token_chosen_rewards.detach(), self.rank, self.world_size
            )
            token_rejected_rewards_gathered = all_gather_if_needed(
                token_rejected_rewards.detach(), self.rank, self.world_size
            )

            # Compute mean, median, p95 per sequence for chosen rewards
            chosen_reward_means = token_chosen_rewards_gathered.mean(dim=-1).cpu().numpy().tolist()
            chosen_reward_medians = (
                token_chosen_rewards_gathered.median(dim=-1).values.cpu().numpy().tolist()
            )
            chosen_reward_p95 = (
                token_chosen_rewards_gathered.quantile(0.95, dim=-1).cpu().numpy().tolist()
            )

            metrics[f"token_chosen_rewards_{train_test}/mean_per_sequence"] = chosen_reward_means
            metrics[f"token_chosen_rewards_{train_test}/median_per_sequence"] = (
                chosen_reward_medians
            )
            metrics[f"token_chosen_rewards_{train_test}/p95_per_sequence"] = chosen_reward_p95

            # Compute mean, median, p95 per sequence for rejected rewards
            rejected_reward_means = (
                token_rejected_rewards_gathered.mean(dim=-1).cpu().numpy().tolist()
            )
            rejected_reward_medians = (
                token_rejected_rewards_gathered.median(dim=-1).values.cpu().numpy().tolist()
            )
            rejected_reward_p95 = (
                token_rejected_rewards_gathered.quantile(0.95, dim=-1).cpu().numpy().tolist()
            )

            metrics[f"token_rejected_rewards_{train_test}/mean_per_sequence"] = (
                rejected_reward_means
            )
            metrics[f"token_rejected_rewards_{train_test}/median_per_sequence"] = (
                rejected_reward_medians
            )
            metrics[f"token_rejected_rewards_{train_test}/p95_per_sequence"] = rejected_reward_p95

            entropy_per_prompt_gathered = all_gather_if_needed(
                entropy_per_prompt.detach(), self.rank, self.world_size
            )
            metrics[f"entropy_{train_test}/per_prompt"] = (
                entropy_per_prompt_gathered.cpu().numpy().tolist()
            )

            kl_forward_per_prompt_gathered = all_gather_if_needed(
                kl_forward_per_prompt.detach(), self.rank, self.world_size
            )
            metrics[f"kl_forward_{train_test}/per_prompt"] = (
                kl_forward_per_prompt_gathered.cpu().numpy().tolist()
            )

            kl_reverse_per_prompt_gathered = all_gather_if_needed(
                kl_reverse_per_prompt.detach(), self.rank, self.world_size
            )
            metrics[f"kl_reverse_{train_test}/per_prompt"] = (
                kl_reverse_per_prompt_gathered.cpu().numpy().tolist()
            )

            metrics[f"logps_{train_test}/rejected"] = policy_rejected_logps.cpu().numpy().tolist()

            if kl_chosen_per_prompt is not None:
                kl_chosen_gathered = all_gather_if_needed(
                    kl_chosen_per_prompt.detach(), self.rank, self.world_size
                )
                kl_rejected_gathered = all_gather_if_needed(
                    kl_rejected_per_prompt.detach(), self.rank, self.world_size
                )
                metrics[f"advantage_kl_{train_test}/chosen"] = (
                    kl_chosen_gathered.cpu().numpy().tolist()
                )
                metrics[f"advantage_kl_{train_test}/rejected"] = (
                    kl_rejected_gathered.cpu().numpy().tolist()
                )
                metrics[f"advantage_kl_{train_test}/margin"] = (
                    (kl_rejected_gathered - kl_chosen_gathered).cpu().numpy().tolist()
                )

        policy_chosen_logps = all_gather_if_needed(
            policy_chosen_logps.detach(), self.rank, self.world_size
        )
        metrics[f"logps_{train_test}/chosen"] = policy_chosen_logps.cpu().numpy().tolist()

        all_devices_losses = all_gather_if_needed(losses.detach(), self.rank, self.world_size)
        metrics[f"loss/{train_test}"] = all_devices_losses.cpu().numpy().tolist()

        return losses.mean(), metrics

    def train(self):
        rank0_print(f"Using {self.config.optimizer} optimizer")
        policy_params = [p for p in self.policy.parameters() if p.requires_grad]
        self.policy_optimizer = getattr(torch.optim, self.config.optimizer)(
            policy_params, lr=self.config.lr, weight_decay=self.config.weight_decay
        )

        self.baseline_optimizer = None
        if self.baseline_head is not None:
            self.baseline_optimizer = getattr(torch.optim, self.config.baseline_optimizer)(
                self.baseline_head.parameters(),
                lr=self.config.baseline_lr,
                weight_decay=self.config.baseline_weight_decay,
            )
        if self.config.scheduler == "linear":
            self.scheduler = get_linear_schedule_with_warmup(
                self.policy_optimizer,
                num_warmup_steps=int(self.config.warmup_ratio * self.total_steps),
                num_training_steps=self.total_steps,
            )
        elif self.config.scheduler == "cosine":
            self.scheduler = get_cosine_schedule_with_warmup(
                self.policy_optimizer,
                num_warmup_steps=int(self.config.warmup_ratio * self.total_steps),
                num_training_steps=self.total_steps,
            )

        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

        if self.config.loss.name in {
            "dpo",
            "ipo",
            "tdpo",
            "tisdpo",
            "Q_tbpo",
            "A_tbpo",
            "BPO_SBA",
            "arc_bpo",
        }:
            self.reference_model.eval()

        self.example_counter = 0
        self.batch_counter = 0
        last_log = None

        for batch in self.train_iterator:
            #### BEGIN EVALUATION ####
            # if self.batch_counter == 0:
            #     capture_memory_snapshot(self.rank, step=self.batch_counter, output_dir=self.run_dir)

            if self.eval_batches and self.example_counter % self.eval_every == 0 and (
                self.example_counter > 0 or self.config.do_first_eval
            ):
                rank0_print(f"Running evaluation after {self.example_counter} train examples")
                self.policy.eval()

                all_eval_metrics = defaultdict(list)
                if self.config.sample_during_eval:
                    all_policy_samples, all_reference_samples = [], []
                    policy_text_table = wandb.Table(columns=["step", "prompt", "sample"])
                    if self.config.loss.name in {"dpo", "ipo", "tdpo", "tisdpo"}:
                        reference_text_table = wandb.Table(columns=["step", "prompt", "sample"])

                for eval_batch in (
                    tqdm.tqdm(self.eval_batches, desc="Computing eval metrics")
                    if self.rank == 0
                    else self.eval_batches
                ):
                    local_eval_batch = slice_and_move_batch_for_device(
                        eval_batch, self.rank, self.world_size, self.rank
                    )
                    with torch.no_grad():
                        _, eval_metrics = self.get_batch_metrics(
                            local_eval_batch, self.config.loss, train=False
                        )

                    for k, v in eval_metrics.items():
                        if isinstance(v, list):
                            all_eval_metrics[k].extend(v)
                        else:
                            all_eval_metrics[k].append(v)

                if self.config.sample_during_eval:
                    if self.config.n_eval_model_samples < self.config.eval_batch_size:
                        rank0_print(
                            f"Warning: n_eval_model_samples ({self.config.n_eval_model_samples}) < eval_batch_size ({self.config.eval_batch_size}). Sampling from the first complete eval batch of prompts."
                        )
                        sample_batches = self.eval_batches[:1]
                    else:
                        n_sample_batches = (
                            self.config.n_eval_model_samples // self.config.eval_batch_size
                        )
                        sample_batches = self.eval_batches[:n_sample_batches]
                    for eval_batch in (
                        tqdm.tqdm(sample_batches, desc="Generating samples...")
                        if self.rank == 0
                        else sample_batches
                    ):
                        local_eval_batch = slice_and_move_batch_for_device(
                            eval_batch, self.rank, self.world_size, self.rank
                        )
                        policy_samples, reference_samples = self.get_batch_samples(local_eval_batch)

                        all_policy_samples.extend(policy_samples)
                        all_reference_samples.extend(reference_samples)

                        for prompt, sample in zip(eval_batch["prompt"], policy_samples):
                            policy_text_table.add_data(self.example_counter, prompt, sample)
                        if self.config.loss.name in {"dpo", "ipo", "tdpo", "tisdpo"}:
                            for prompt, sample in zip(eval_batch["prompt"], reference_samples):
                                reference_text_table.add_data(self.example_counter, prompt, sample)

                mean_eval_metrics = {k: sum(v) / len(v) for k, v in all_eval_metrics.items()}
                rank0_print(
                    f"eval after {self.example_counter}: {formatted_dict(mean_eval_metrics)}"
                )
                if self.config.sample_during_eval:
                    rank0_print(json.dumps(all_policy_samples[:10], indent=2))
                    if self.config.loss.name in {"dpo", "ipo", "tdpo", "tisdpo"}:
                        rank0_print(json.dumps(all_reference_samples[:10], indent=2))

                if self.config.wandb.enabled and self.rank == 0:
                    wandb.log(mean_eval_metrics, step=self.example_counter)

                    if self.config.sample_during_eval:
                        wandb.log({"policy_samples": policy_text_table}, step=self.example_counter)
                        if self.config.loss.name in {"dpo", "ipo", "tdpo", "tisdpo"}:
                            wandb.log(
                                {"reference_samples": reference_text_table},
                                step=self.example_counter,
                            )
            #### END EVALUATION ####

            #### BEGIN TRAINING ####
            self.policy.train()

            start_time = time.time()
            batch_metrics = defaultdict(list)
            for microbatch_idx in range(self.config.gradient_accumulation_steps):
                global_microbatch = slice_and_move_batch_for_device(
                    batch, microbatch_idx, self.config.gradient_accumulation_steps, self.rank
                )
                local_microbatch = slice_and_move_batch_for_device(
                    global_microbatch, self.rank, self.world_size, self.rank
                )
                loss, metrics = self.get_batch_metrics(
                    local_microbatch, self.config.loss, train=True
                )
                (loss / self.config.gradient_accumulation_steps).backward()
                # try:
                #     loss, metrics = self.get_batch_metrics(
                #         local_microbatch, self.config.loss, train=True
                #     )
                #     (loss / self.config.gradient_accumulation_steps).backward()
                # except torch.cuda.OutOfMemoryError:
                #     capture_memory_snapshot(self.rank, step=self.batch_counter, prefix="oom_snapshot", output_dir=self.run_dir)
                #     raise

                for k, v in metrics.items():
                    if isinstance(v, list):
                        batch_metrics[k].extend(v)
                    else:
                        batch_metrics[k].append(v)

            policy_norm, baseline_norm = self.clip_gradient()
            self.policy_optimizer.step()
            self.scheduler.step()
            self.policy_optimizer.zero_grad(set_to_none=True)
            if self.baseline_optimizer is not None:
                self.baseline_optimizer.step()
                self.baseline_optimizer.zero_grad(set_to_none=True)

            step_time = time.time() - start_time
            examples_per_second = self.config.batch_size / step_time
            batch_metrics["examples_per_second"].append(examples_per_second)
            batch_metrics["policy_grad_norm"].append(policy_norm)
            batch_metrics["policy_clip_frac"].append(self.policy_clip_hits / self.total_steps)
            if self.baseline_head is not None:
                batch_metrics["baseline_grad_norm"].append(baseline_norm)
                batch_metrics["baseline_clip_frac"].append(
                    self.baseline_clip_hits / self.total_steps
                )

            self.batch_counter += 1
            self.example_counter += self.config.batch_size

            if (
                self.config.save_checkpoint
                and self.save_every > 0
                and self.example_counter > 0
                and self.example_counter >= self.next_save_example
            ):
                rank0_print(
                    f"Saving checkpoint after {self.example_counter} train examples "
                    f"(update {self.batch_counter})"
                )
                self.save_checkpoint(step=self.batch_counter)
                while self.next_save_example <= self.example_counter:
                    self.next_save_example += self.save_every

            if last_log is None or time.time() - last_log > self.config.minimum_log_interval_secs:
                mean_train_metrics = {k: sum(v) / len(v) for k, v in batch_metrics.items()}
                mean_train_metrics["counters/examples"] = self.example_counter
                mean_train_metrics["counters/updates"] = self.batch_counter
                mean_train_metrics["lr"] = self.scheduler.get_last_lr()[0]
                rank0_print(
                    f"train stats after {self.example_counter} examples: {formatted_dict(mean_train_metrics)}"
                )

                if self.config.wandb.enabled and self.rank == 0:
                    wandb.log(mean_train_metrics, step=self.example_counter)

                last_log = time.time()
            else:
                rank0_print(
                    f"skipping logging after {self.example_counter} examples to avoid logging too frequently"
                )
            #### END TRAINING ####

    def clip_gradient(self):
        """Clip the gradient norm of the parameters of a non-FSDP policy."""
        max_norm = self.config.max_grad_norm
        policy_norm = torch.nn.utils.clip_grad_norm_(
            list(self.policy.parameters()), max_norm
        ).item()
        baseline_norm = 0.0
        if self.baseline_head is not None:
            baseline_norm = torch.nn.utils.clip_grad_norm_(
                list(self.baseline_head.parameters()), max_norm
            ).item()
        if self.rank == 0:
            if policy_norm > max_norm:
                self.policy_clip_hits += 1
            if self.baseline_head is not None and baseline_norm > max_norm:
                self.baseline_clip_hits += 1
        return policy_norm, baseline_norm

    def write_state_dict(
        self,
        step: int,
        state: Dict[str, torch.Tensor],
        metrics: Dict,
        filename: str,
        dir_name: Optional[str] = None,
    ):
        """Write a checkpoint to disk."""
        if dir_name is None:
            dir_name = os.path.join(self.run_dir, "LATEST")

        os.makedirs(dir_name, exist_ok=True)
        output_path = os.path.join(dir_name, filename)
        rank0_print(f"writing checkpoint to {output_path}...")
        torch.save(
            {
                "step_idx": step,
                "state": state,
                "metrics": metrics if metrics is not None else {},
            },
            output_path,
        )

    def save_checkpoint(self, step: int, output_dir=None):
        if output_dir is None:
            model_save_dir = os.path.join(self.run_dir, f"step-{step}")
        else:
            model_save_dir = output_dir

        if self.rank == 0:
            os.makedirs(model_save_dir, exist_ok=True)
        if dist.is_available() and dist.is_initialized():
            dist.barrier()

        self.save(output_dir=model_save_dir)
        if dist.is_available() and dist.is_initialized():
            dist.barrier()

    def save(self, output_dir: Optional[str] = None, metrics: Optional[Dict] = None):
        """Save policy and tokenizer to disk."""
        if output_dir is None:
            model_save_dir = os.path.join(self.run_dir, "LATEST")
        else:
            model_save_dir = output_dir

        os.makedirs(model_save_dir, exist_ok=True)

        use_lora = bool(getattr(self.config.model, "use_lora", False))

        # --- save policy ---
        policy_to_save = _unwrap(self.policy)  # handles DDP; BasicTrainer isn't FSDP-wrapped
        if use_lora:
            # Save adapter-only
            adapter_dir = os.path.join(model_save_dir, "adapter")
            os.makedirs(adapter_dir, exist_ok=True)
            policy_to_save.save_pretrained(adapter_dir, safe_serialization=True)
        else:
            # Save full model
            policy_to_save.save_pretrained(model_save_dir, safe_serialization=True)

        # baseline head
        if self.baseline_head is not None:
            head = _unwrap(self.baseline_head)
            torch.save(head.state_dict(), os.path.join(model_save_dir, "baseline_head.pt"))

        # Save tokenizer alongside the model
        self.tokenizer.save_pretrained(model_save_dir)

        # Save metrics separately
        if metrics is not None:
            metrics_file = os.path.join(model_save_dir, "training_metrics.json")
            with open(metrics_file, "w") as f:
                json.dump({"step": self.example_counter, "metrics": metrics}, f)


class FSDPTrainer(BasicTrainer):
    def __init__(
        self,
        policy: nn.Module,
        config: DictConfig,
        seed: int,
        run_dir: str,
        baseline_head: Optional[nn.Module] = None,
        reference_model: Optional[nn.Module] = None,
        rank: int = 0,
        world_size: int = 1,
        transform_config=None,
    ):
        """A trainer subclass that uses PyTorch FSDP to shard the model across multiple GPUs.

        This trainer will shard both the policy and reference model across all available GPUs.
        Models are sharded at the block level, where the block class name is provided in the config.
        """

        super().__init__(
            policy,
            config,
            seed,
            run_dir,
            baseline_head,
            reference_model,
            rank,
            world_size,
        )
        assert config.model.block_name is not None, (
            "must specify model.block_name (e.g., GPT2Block or GPTNeoXLayer) for FSDP"
        )
        wrap_class = get_block_class_from_model(policy, config.model.block_name)
        if getattr(config.model, "use_lora", False):
            from peft.utils.other import fsdp_auto_wrap_policy

            model_auto_wrap_policy = fsdp_auto_wrap_policy(policy)
        else:
            model_auto_wrap_policy = functools.partial(
                transformer_auto_wrap_policy, transformer_layer_cls={wrap_class}
            )

        shared_fsdp_kwargs = dict(
            auto_wrap_policy=model_auto_wrap_policy,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            cpu_offload=CPUOffload(offload_params=False),
            backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
            device_id=rank,
            ignored_modules=None,
            limit_all_gathers=False,
            use_orig_params=False,
            sync_module_states=False,
        )

        rank0_print("Sharding policy...")
        mp_dtype = (
            getattr(torch, config.model.fsdp_policy_mp)
            if config.model.fsdp_policy_mp is not None
            else None
        )
        policy_mp_policy = MixedPrecision(
            param_dtype=mp_dtype, reduce_dtype=mp_dtype, buffer_dtype=mp_dtype
        )
        self.policy = FSDP(policy, **shared_fsdp_kwargs, mixed_precision=policy_mp_policy)

        if self.baseline_head is not None:
            self.baseline_head = self.baseline_head.to(rank)
            self.baseline_head = torch.nn.parallel.DistributedDataParallel(
                self.baseline_head, device_ids=[rank], output_device=rank
            )

        if config.activation_checkpointing:
            rank0_print("Attempting to enable activation checkpointing...")
            try:
                # use activation checkpointing, according to:
                # https://pytorch.org/blog/scaling-multimodal-foundation-models-in-torchmultimodal-with-pytorch-distributed/
                #
                # first, verify we have FSDP activation support ready by importing:
                from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
                    CheckpointImpl,
                    apply_activation_checkpointing,
                    checkpoint_wrapper,
                )

                non_reentrant_wrapper = functools.partial(
                    checkpoint_wrapper,
                    offload_to_cpu=False,
                    checkpoint_impl=CheckpointImpl.NO_REENTRANT,
                )
            except Exception as e:
                rank0_print("FSDP activation checkpointing not available:", e)
            else:
                check_fn = lambda submodule: isinstance(submodule, wrap_class)
                rank0_print("Applying activation checkpointing wrapper to policy...")
                apply_activation_checkpointing(
                    self.policy, checkpoint_wrapper_fn=non_reentrant_wrapper, check_fn=check_fn
                )
                rank0_print("FSDP activation checkpointing enabled!")

        if config.loss.name in {
            "dpo",
            "ipo",
            "tdpo",
            "tisdpo",
            "Q_tbpo",
            "A_tbpo",
            "BPO_SBA",
            "arc_bpo",
        }:
            rank0_print("Sharding reference model...")
            self.reference_model = FSDP(reference_model, **shared_fsdp_kwargs)

        print("Loaded model on rank", rank)
        dist.barrier()

    def clip_gradient(self):
        """Clip the gradient norm of the parameters of an FSDP policy, gathering the gradients across all GPUs."""
        max_norm = self.config.max_grad_norm
        policy_norm = self.policy.clip_grad_norm_(max_norm).item()
        baseline_norm = 0.0
        if self.baseline_head is not None:
            baseline_norm = torch.nn.utils.clip_grad_norm_(
                self.baseline_head.parameters(), max_norm
            ).item()
        if self.rank == 0:
            if policy_norm > max_norm:
                self.policy_clip_hits += 1
            if self.baseline_head is not None and baseline_norm > max_norm:
                self.baseline_clip_hits += 1
        return policy_norm, baseline_norm

    def save_checkpoint(self, step: int, output_dir=None):
        if output_dir is None:
            model_save_dir = os.path.join(self.run_dir, f"step-{step}")
        else:
            model_save_dir = output_dir
        if self.rank == 0:
            os.makedirs(model_save_dir, exist_ok=True)
        dist.barrier()
        self.save(output_dir=model_save_dir)
        dist.barrier()

    def save(self, output_dir=None, metrics=None):
        """Save policy and tokenizer state to disk, gathering from all processes and saving only on the rank 0 process."""
        use_lora = bool(getattr(self.config.model, "use_lora", False))
        save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(
            self.policy, StateDictType.FULL_STATE_DICT, state_dict_config=save_policy
        ):
            policy_state_dict = self.policy.state_dict()

        if self.rank == 0:
            # Save model using transformers save_pretrained
            if output_dir is None:
                model_save_dir = os.path.join(self.run_dir, "LATEST")
            else:
                model_save_dir = output_dir

            os.makedirs(model_save_dir, exist_ok=True)

            underlying = self.policy.module

            if use_lora:
                adapter_dir = os.path.join(model_save_dir, "adapter")
                os.makedirs(adapter_dir, exist_ok=True)

                # Let PEFT filter the full FSDP state dict itself. Passing an
                # already-filtered adapter state can be filtered a second time
                # by PEFT and produce an empty 40-byte safetensors file.
                lora_keys = [k for k in policy_state_dict if "lora_" in k]
                if not lora_keys:
                    sample_keys = list(policy_state_dict.keys())[:20]
                    raise RuntimeError(
                        "LoRA save requested but no lora_ tensors were found in "
                        f"the gathered FSDP state dict. Sample keys: {sample_keys}"
                    )
                print(f"[SAVE] Found {len(lora_keys)} LoRA tensors in full state dict.")

                underlying.save_pretrained(
                    adapter_dir,
                    safe_serialization=True,
                    state_dict=policy_state_dict,
                )
                adapter_file = os.path.join(adapter_dir, "adapter_model.safetensors")
                if os.path.isfile(adapter_file) and os.path.getsize(adapter_file) <= 1024:
                    raise RuntimeError(
                        f"Saved LoRA adapter is unexpectedly small "
                        f"({os.path.getsize(adapter_file)} bytes): {adapter_file}. "
                        "The adapter state was not written correctly."
                    )
            else:
                # Full model save (HF format)
                underlying.save_pretrained(
                    model_save_dir,
                    safe_serialization=True,
                    state_dict=policy_state_dict,
                )

            if self.baseline_head is not None:
                head = _unwrap(self.baseline_head)  # DDP-wrapped in your __init__
                torch.save(head.state_dict(), os.path.join(model_save_dir, "baseline_head.pt"))

            # Save tokenizer alongside the model
            self.tokenizer.save_pretrained(model_save_dir)

            # Save metrics separately
            if metrics is not None:
                metrics_file = os.path.join(model_save_dir, "training_metrics.json")
                with open(metrics_file, "w") as f:
                    json.dump({"step": self.example_counter, "metrics": metrics}, f)

        del policy_state_dict
        dist.barrier()
