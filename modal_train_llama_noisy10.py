"""
Run ARC-BPO training for Llama-3-8B with 10% noisy preference labels on Modal.

This is for the label-noise robustness setting:
  randomly swap chosen/rejected in 10% of training preference pairs.

Full run, public HF adapter upload:
  modal run --detach modal_train_llama_noisy10.py::main \
    --hf-repo-id ducthang1703/llama3-8b-arc-bpo-lora-noise10-full

Smoke test:
  modal run --detach modal_train_llama_noisy10.py::main \
    --n-examples 128 \
    --hf-repo-id ducthang1703/llama3-8b-arc-bpo-lora-noise10-smoke-128

Inspect saved Modal volume outputs:
  modal run modal_train_llama_noisy10.py::ls_outputs
"""

import modal


APP_NAME = "tbpo-arc-bpo-llama-noisy10-train"
RUN_VERSION = "arc-bpo-llama3-noisy10-h100x4-v1"

REPO_DIR = "/root/arc_bpo"
VOLUME_ROOT = "/vol/output"
OUTPUT_DIR = f"{VOLUME_ROOT}/train_runs_llama_noisy10"
GPU_CONFIG = "H100:4"

app = modal.App(APP_NAME)
output_volume = modal.Volume.from_name("tbpo-output", create_if_missing=True)


def _gpu_count(gpu_config: str) -> int:
    if ":" not in gpu_config:
        return 1
    try:
        return int(gpu_config.rsplit(":", 1)[1])
    except ValueError:
        return 1


def _bool_arg(flag: bool) -> str:
    return "true" if flag else "false"


image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git")
    .pip_install(
        "torch",
        "transformers>=4.31.0",
        "datasets>=2.12.0",
        "accelerate>=0.20.3",
        "peft",
        "hydra-core>=1.3.2",
        "omegaconf>=2.3.0",
        "wandb>=0.15.4",
        "sentencepiece>=0.1.99",
        "protobuf>=4.23.3",
        "tqdm>=4.65.0",
        "huggingface_hub",
        "safetensors",
    )
    .env(
        {
            "TOKENIZERS_PARALLELISM": "false",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        }
    )
    .add_local_dir(
        ".",
        remote_path=REPO_DIR,
        ignore=[
            ".git/**",
            "output/**",
            "wandb/**",
            "mtbench/**",
            "**/__pycache__/**",
            ".venv/**",
            "*.pyc",
        ],
    )
)


_SECRETS = [
    modal.Secret.from_name("huggingface-secret"),
]


@app.function(
    image=image,
    gpu=GPU_CONFIG,
    volumes={VOLUME_ROOT: output_volume},
    secrets=_SECRETS,
    timeout=60 * 60 * 18,
)
def train(
    hf_repo_id: str,
    hf_private: bool = False,
    hf_upload_adapter_only: bool = True,
    # ARC-BPO loss hyperparameters
    beta: float = 0.1,
    delta_star: float = 2.5,
    arc_t: float = 2.0,
    kappa: float = 2.0,
    sba_lambda: float = 1.0,
    sba_scale: float = 4.0,
    exp_clip: float = 30.0,
    use_advantage_shape: bool = False,
    fallback_to_uniform_shape: bool = True,
    min_tokens_per_chunk: int = 4,
    max_tokens_per_chunk: int = 64,
    # Label-noise robustness setting
    label_noise_rate: float = 0.10,
    label_noise_seed: int = 0,
    # Data / optimization
    model_config: str = "llama_8b",
    datasets_raw: str = "princeton-nlp/llama3-ultrafeedback-armorm",
    train_split: str = "train",
    test_split: str = "test",
    batch_size: int = 64,
    grad_accum: int = 4,
    lr: str = "5e-7",
    weight_decay: float = 0.0,
    max_grad_norm: float = 10.0,
    optimizer: str = "RMSprop",
    scheduler: str = "cosine",
    warmup_ratio: float = 0.05,
    max_length: int = 2048,
    n_epochs: int = 1,
    n_examples: int = 0,
    eval_batch_size: int = 16,
    n_eval_examples: int = 0,
    save_checkpoint: bool = True,
    do_first_eval: bool = False,
    activation_checkpointing: bool = True,
    use_lora: bool = True,
    wandb_enabled: bool = False,
):
    import os
    import subprocess

    if not hf_repo_id:
        raise ValueError("hf_repo_id is required so the noisy run is saved to Hugging Face.")
    if not 0.0 <= label_noise_rate <= 1.0:
        raise ValueError(f"label_noise_rate must be in [0, 1], got {label_noise_rate}")

    world_size = _gpu_count(GPU_CONFIG)
    min_global_batch = grad_accum * world_size
    if batch_size < min_global_batch:
        raise ValueError(
            "FSDP would create an empty per-rank microbatch: "
            f"batch_size={batch_size}, grad_accum={grad_accum}, "
            f"world_size={world_size}. Use batch_size >= {min_global_batch}."
        )
    if batch_size % min_global_batch != 0:
        raise ValueError(
            "batch_size must be divisible by grad_accum * world_size: "
            f"batch_size={batch_size}, divisor={min_global_batch}."
        )

    os.chdir(REPO_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not wandb_enabled:
        os.environ["WANDB_MODE"] = "disabled"
    os.environ["HYDRA_FULL_ERROR"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    print(f"[VERSION] {RUN_VERSION}")
    print(f"[GPU] {GPU_CONFIG}")
    print(f"[MODEL] {model_config}")
    print(f"[DATASET] {datasets_raw}:{train_split}")
    print(f"[LABEL NOISE] rate={label_noise_rate:.2%} seed={label_noise_seed}")
    print(f"[OUTPUT] {OUTPUT_DIR}")
    print(f"[HF SAVE] {hf_repo_id}")

    cmd = [
        "python3",
        "train.py",
        f"model={model_config}",
        f"model.use_lora={_bool_arg(use_lora)}",
        "model.use_baseline_head=false",
        "loss=arc_bpo",
        f"loss.beta={beta}",
        f"loss.delta_star={delta_star}",
        f"loss.T={arc_t}",
        f"loss.kappa={kappa}",
        f"loss.sba_lambda={sba_lambda}",
        f"loss.sba_scale={sba_scale}",
        f"loss.exp_clip={exp_clip}",
        f"loss.use_advantage_shape={_bool_arg(use_advantage_shape)}",
        f"loss.fallback_to_uniform_shape={_bool_arg(fallback_to_uniform_shape)}",
        f"loss.min_tokens_per_chunk={min_tokens_per_chunk}",
        f"loss.max_tokens_per_chunk={max_tokens_per_chunk}",
        f"datasets={datasets_raw}",
        f"dataset_train_split={train_split}",
        f"dataset_test_split={test_split}",
        "trainer=FSDPTrainer",
        f"batch_size={batch_size}",
        f"gradient_accumulation_steps={grad_accum}",
        f"lr={lr}",
        f"weight_decay={weight_decay}",
        f"max_grad_norm={max_grad_norm}",
        f"optimizer={optimizer}",
        f"scheduler={scheduler}",
        f"warmup_ratio={warmup_ratio}",
        f"max_length={max_length}",
        f"n_epochs={n_epochs}",
        f"label_noise_rate={label_noise_rate}",
        f"label_noise_seed={label_noise_seed}",
        f"eval_batch_size={eval_batch_size}",
        f"n_eval_examples={n_eval_examples}",
        f"save_checkpoint={_bool_arg(save_checkpoint)}",
        f"do_first_eval={_bool_arg(do_first_eval)}",
        f"activation_checkpointing={_bool_arg(activation_checkpointing)}",
        f"output_dir={OUTPUT_DIR}",
        f"wandb.enabled={_bool_arg(wandb_enabled)}",
    ]
    if n_examples and n_examples > 0:
        cmd.append(f"n_examples={n_examples}")
        cmd.append("n_epochs=null")

    print("[RUN] " + " ".join(cmd))
    subprocess.run(cmd, check=True)

    output_volume.commit()
    print("[DONE] Training finished; outputs committed to Modal volume.")

    from huggingface_hub import HfApi, upload_folder

    latest_dirs = []
    for root, dirs, _ in os.walk(OUTPUT_DIR):
        if "LATEST" in dirs:
            latest_path = os.path.join(root, "LATEST")
            latest_dirs.append((os.path.getmtime(latest_path), latest_path))
    if not latest_dirs:
        raise RuntimeError(f"No LATEST checkpoint found under {OUTPUT_DIR}")

    latest_path = max(latest_dirs)[1]
    upload_path = latest_path
    if use_lora and hf_upload_adapter_only:
        upload_path = os.path.join(latest_path, "adapter")
        adapter_config = os.path.join(upload_path, "adapter_config.json")
        adapter_model = os.path.join(upload_path, "adapter_model.safetensors")
        if not os.path.isfile(adapter_config):
            raise RuntimeError(f"No LoRA adapter_config.json found at {upload_path}")
        if not os.path.isfile(adapter_model) or os.path.getsize(adapter_model) <= 1024:
            raise RuntimeError(
                f"LoRA adapter_model.safetensors is missing or too small: {adapter_model}"
            )

    print(f"[HF UPLOAD] repo={hf_repo_id}")
    print(f"[HF UPLOAD] folder={upload_path}")
    api = HfApi()
    api.create_repo(hf_repo_id, private=hf_private, exist_ok=True)
    upload_folder(
        repo_id=hf_repo_id,
        folder_path=upload_path,
        commit_message=(
            "Upload ARC-BPO Llama-3-8B label-noise run "
            f"(noise={label_noise_rate:.2%}, seed={label_noise_seed})"
        ),
    )
    print(f"[HF UPLOAD] Done: https://huggingface.co/{hf_repo_id}")

    return {
        "output_dir": OUTPUT_DIR,
        "hf_repo_id": hf_repo_id,
        "label_noise_rate": label_noise_rate,
        "label_noise_seed": label_noise_seed,
    }


@app.function(image=image, volumes={VOLUME_ROOT: output_volume}, timeout=60 * 5)
def ls_outputs():
    import os

    output_volume.reload()
    if not os.path.isdir(OUTPUT_DIR):
        print(f"No outputs yet at {OUTPUT_DIR}")
        return
    for root, dirs, files in os.walk(OUTPUT_DIR):
        depth = root[len(OUTPUT_DIR) :].count(os.sep)
        if depth > 2:
            continue
        indent = "  " * depth
        print(f"{indent}{os.path.basename(root) or OUTPUT_DIR}/")
        for f in files:
            print(f"{indent}  {f}")


@app.local_entrypoint()
def main(
    hf_repo_id: str = "ducthang1703/llama3-8b-arc-bpo-lora-noise10-full",
    hf_private: bool = False,
    batch_size: int = 64,
    grad_accum: int = 4,
    n_epochs: int = 1,
    n_examples: int = 0,
    label_noise_rate: float = 0.10,
    label_noise_seed: int = 0,
    n_eval_examples: int = 0,
    do_first_eval: bool = False,
    use_lora: bool = True,
    wandb_enabled: bool = False,
):
    print(f"[LAUNCH] {APP_NAME} ({RUN_VERSION})")
    print(
        f"[MODE] Llama-3-8B ARC-BPO noisy training "
        f"(noise={label_noise_rate:.2%}, "
        f"n_examples={n_examples or 'full epoch'}, batch_size={batch_size})"
    )
    train.spawn(
        hf_repo_id=hf_repo_id,
        hf_private=hf_private,
        batch_size=batch_size,
        grad_accum=grad_accum,
        n_epochs=n_epochs,
        n_examples=n_examples,
        label_noise_rate=label_noise_rate,
        label_noise_seed=label_noise_seed,
        n_eval_examples=n_eval_examples,
        do_first_eval=do_first_eval,
        use_lora=use_lora,
        wandb_enabled=wandb_enabled,
    )
    print("[LAUNCHED] Noisy ARC-BPO Llama training is running on Modal.")
    print("[RESULTS] Inspect outputs with: modal run modal_train_llama_noisy10.py::ls_outputs")
