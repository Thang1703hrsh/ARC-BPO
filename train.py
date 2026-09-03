import torch

torch.backends.cuda.matmul.allow_tf32 = True
import os
import resource
import socket
from typing import Set

import hydra
import torch.distributed as dist
import torch.multiprocessing as mp
import transformers
import wandb
from omegaconf import DictConfig, OmegaConf
from peft import LoraConfig, PeftModel, TaskType, get_peft_model

import trainers
from baseline_head import BaselineHead
from huggingface_utils import push_final_checkpoint_to_hub
from utils import (
    build_exp_name,
    disable_dropout,
    get_local_dir,
    get_local_run_dir,
    get_open_port,
    init_distributed,
    rank0_print,
)

OmegaConf.register_new_resolver(
    "get_local_run_dir", lambda exp_name, local_dir: get_local_run_dir(exp_name, local_dir)
)
OmegaConf.register_new_resolver(
    "build_exp_name",
    lambda loss_name, model_name, datasets: build_exp_name(
        loss_name,
        model_name,
        datasets,
    ),
)


def default_lora_targets(model_name: str):
    # good default for Llama/Mistral/Qwen2-style decoders
    return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def build_lora_config(cfg):
    targets = cfg.model.lora_target_modules or default_lora_targets(cfg.model.name_or_path)
    return LoraConfig(
        r=cfg.model.lora_r,
        lora_alpha=cfg.model.lora_alpha,
        lora_dropout=cfg.model.lora_dropout,
        bias=cfg.model.lora_bias,
        target_modules=targets,
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
    )


def worker_main(
    rank: int,
    world_size: int,
    config: DictConfig,
):
    """Main function for each worker process (may be only 1 for BasicTrainer/TensorParallelTrainer)."""
    if "FSDP" in config.trainer:
        init_distributed(rank, world_size, port=config.fsdp_port)

    if config.debug:
        os.environ["WANDB_MODE"] = "disabled"

    if rank == 0 and config.wandb.enabled:
        os.environ["WANDB_CACHE_DIR"] = get_local_dir(config.output_dir)
        wandb.init(
            entity=config.wandb.entity,
            project=config.wandb.project,
            config=OmegaConf.to_container(config),
            dir=get_local_dir(config.output_dir),
            name=config.exp_name,
        )

    # if torch.cuda.is_available():  # All ranks
    #     torch.cuda.memory._record_memory_history(max_entries=100000)

    rank0_print("building policy")
    model_kwargs = {"device_map": "balanced"} if config.trainer == "BasicTrainer" else {}
    policy_dtype = getattr(torch, config.model.policy_dtype)
    model_revision = getattr(config.model, "revision", None)
    policy = transformers.AutoModelForCausalLM.from_pretrained(
        config.model.name_or_path,
        revision=model_revision,
        low_cpu_mem_usage=True,
        torch_dtype=policy_dtype,
        **model_kwargs,
    )
    if config.model.use_baseline_head:
        hidden_size = getattr(policy.config, "hidden_size", None) or getattr(
            policy.config, "n_embd"
        )
        baseline_head = BaselineHead(
            hidden_size=hidden_size,
            mlp_dim=config.model.baseline_mlp_dim,
            dropout=config.model.baseline_dropout,
        )
    else:
        baseline_head = None
    disable_dropout(policy)
    if config.model.use_lora:
        adapter_path = getattr(config.model, "adapter_path", None)
        if adapter_path:
            rank0_print(f"loading trainable LoRA adapter from {adapter_path}")
            policy = PeftModel.from_pretrained(policy, adapter_path, is_trainable=True)
        else:
            lora_cfg = build_lora_config(config)
            policy = get_peft_model(policy, lora_cfg)
    elif getattr(config.model, "adapter_path", None):
        raise ValueError("model.adapter_path requires model.use_lora=true")

    if rank == 0:
        if config.model.use_lora:
            policy.print_trainable_parameters()
        else:
            n_params = sum(p.numel() for p in policy.parameters())
            n_trainable = sum(p.numel() for p in policy.parameters() if p.requires_grad)
            print(
                f"trainable params: {n_trainable:,} || "
                f"all params: {n_params:,} || "
                f"trainable%: {100.0 * n_trainable / max(n_params, 1):.4f}"
            )

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
        rank0_print("building reference model")
        reference_model_dtype = getattr(torch, config.model.reference_dtype)
        reference_model = transformers.AutoModelForCausalLM.from_pretrained(
            config.model.name_or_path,
            revision=model_revision,
            low_cpu_mem_usage=True,
            torch_dtype=reference_model_dtype,
            **model_kwargs,
        )
        disable_dropout(reference_model)
    else:
        reference_model = None

    TrainerClass = getattr(trainers, config.trainer)
    print(f"Creating trainer on process {rank} with world size {world_size}")
    trainer = TrainerClass(
        policy,
        config,
        config.seed,
        config.local_run_dir,
        baseline_head=baseline_head,
        reference_model=reference_model,
        rank=rank,
        world_size=world_size,
    )

    trainer.train()

    if dist.is_initialized():
        dist.barrier()
    torch.cuda.synchronize()
    if rank == 0:
        print("Finished training", rank)

    trainer.save()
    if rank == 0:
        upload = push_final_checkpoint_to_hub(
            config,
            config.local_run_dir,
            use_lora=bool(getattr(config.model, "use_lora", False)),
        )
        if upload["status"] == "uploaded":
            print(f"Pushed final checkpoint to {upload['url']}")
    if dist.is_initialized():
        dist.barrier()
    # clean up
    if dist.is_initialized():
        dist.destroy_process_group()


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(config: DictConfig):
    """Main entry point for training. Validates config, creates/initializes model(s), and kicks off worker process(es)."""

    # Now resolve hydra references with the updated transform config
    OmegaConf.resolve(config)

    missing_keys: Set[str] = OmegaConf.missing_keys(config)
    if missing_keys:
        raise ValueError(f"Got missing keys in config:\n{missing_keys}")

    # if config.eval_every % config.batch_size != 0:
    #     print("WARNING: eval_every must be divisible by batch_size")
    #     print("Setting eval_every to", config.eval_every - config.eval_every % config.batch_size)
    #     config.eval_every = config.eval_every - config.eval_every % config.batch_size

    if "FSDP" in config.trainer and config.fsdp_port is None:
        free_port = get_open_port()
        print("no FSDP port specified; using open port for FSDP:", free_port)
        config.fsdp_port = free_port

    print(OmegaConf.to_yaml(config))

    config_path = os.path.join(config.local_run_dir, "config.yaml")
    with open(config_path, "w") as f:
        OmegaConf.save(config, f)

    print("=" * 80)
    print(f"Writing to {socket.gethostname()}:{config.local_run_dir}")
    print("=" * 80)

    os.environ["XDG_CACHE_HOME"] = get_local_dir(config.output_dir)

    if "FSDP" in config.trainer:
        world_size = torch.cuda.device_count()
        print("starting", world_size, "processes for FSDP training")
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
        print(f"setting RLIMIT_NOFILE soft limit to {hard} from {soft}")
        mp.spawn(
            worker_main,
            nprocs=world_size,
            args=(world_size, config),
            join=True,
        )
    else:
        print("starting single-process worker")
        worker_main(0, 1, config)


if __name__ == "__main__":
    main()
