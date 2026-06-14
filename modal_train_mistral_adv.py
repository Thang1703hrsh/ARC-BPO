"""
Run ARC-BPO advantage-shape training for Mistral-7B on Modal.

This is the Mistral counterpart of modal_train_llama_adv.py. It keeps the
Mistral SFT initialization but switches the preference data to the ArmoRM-scored
UltraFeedback variant so ARC-BPO can construct real per-chunk advantage-shape
targets. Using HuggingFaceH4/ultrafeedback_binarized with
use_advantage_shape=true would mostly fall back to uniform because that dataset
does not carry score_chosen / score_rejected.

Smoke test:
  modal run --detach modal_train_mistral_adv.py::main --smoke \
    --hf-repo-id ducthang1703/mistral-arc-bpo-adv-lora-smoke-128 \
    --no-hf-private

16k LoRA run:
  modal run --detach modal_train_mistral_adv.py::main \
    --n-examples 16000 \
    --batch-size 32 \
    --grad-accum 4 \
    --n-eval-examples 0 \
    --no-do-first-eval \
    --hf-repo-id ducthang1703/mistral-arc-bpo-adv-lora-16k \
    --no-hf-private

Full epoch LoRA run:
  modal run --detach modal_train_mistral_adv.py::main \
    --batch-size 32 \
    --grad-accum 4 \
    --n-eval-examples 0 \
    --no-do-first-eval \
    --hf-repo-id ducthang1703/mistral-arc-bpo-adv-lora-full \
    --no-hf-private

Inspect outputs:
  modal run modal_train_mistral_adv.py::ls_outputs
"""

import modal


APP_NAME = "tbpo-arc-bpo-mistral-train-adv"
RUN_VERSION = "arc-bpo-mistral-adv-h100x4-hub-upload-v1"

REPO_DIR = "/root/arc_bpo"
VOLUME_ROOT = "/vol/output"
OUTPUT_DIR = f"{VOLUME_ROOT}/train_runs_mistral_adv"
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


image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git")
    .pip_install(
        "torch>=2.6.0",
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
    timeout=60 * 60 * 12,
)
def train(
    # --- ARC-BPO loss hyperparameters ---
    beta: float = 0.1,
    delta_star: float = 2.5,
    arc_t: float = 2.0,
    kappa: float = 2.0,
    sba_lambda: float = 1.0,
    sba_scale: float = 4.0,
    exp_clip: float = 30.0,
    use_advantage_shape: bool = True,
    fallback_to_uniform_shape: bool = True,
    min_tokens_per_chunk: int = 4,
    max_tokens_per_chunk: int = 64,
    # --- data / optimization ---
    model_config: str = "mistral_7b",
    datasets_raw: str = "princeton-nlp/llama3-ultrafeedback-armorm",
    train_split: str = "train",
    test_split: str = "test",
    batch_size: int = 32,
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
    n_eval_examples: int = 64,
    save_checkpoint: bool = True,
    save_every_examples: int = 5000,
    do_first_eval: bool = True,
    activation_checkpointing: bool = True,
    use_lora: bool = True,
    wandb_enabled: bool = False,
    hf_repo_id: str = "",
    hf_private: bool = True,
    hf_upload_adapter_only: bool = True,
):
    import os
    import subprocess

    os.chdir(REPO_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"[VERSION] {RUN_VERSION}")
    print(f"[GPU] {GPU_CONFIG}")
    print(
        "[SHAPE] advantage "
        f"(use_advantage_shape={use_advantage_shape}, T={arc_t}, kappa={kappa}, "
        f"fallback={fallback_to_uniform_shape})"
    )
    print(f"[DATASET] {datasets_raw} ({train_split}/{test_split})")
    print(f"[REPO] {REPO_DIR}")
    print(f"[OUTPUT] {OUTPUT_DIR}")

    world_size = _gpu_count(GPU_CONFIG)
    min_global_batch = grad_accum * world_size
    if batch_size < min_global_batch:
        raise ValueError(
            "FSDP would create an empty per-rank microbatch: "
            f"batch_size={batch_size}, grad_accum={grad_accum}, "
            f"world_size={world_size}. Use batch_size >= {min_global_batch}, "
            "or reduce --grad-accum / GPU count."
        )
    if batch_size % min_global_batch != 0:
        raise ValueError(
            "batch_size must be divisible by grad_accum * world_size: "
            f"batch_size={batch_size}, divisor={min_global_batch}."
        )

    if not wandb_enabled:
        os.environ["WANDB_MODE"] = "disabled"
    os.environ["HYDRA_FULL_ERROR"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    def b(flag: bool) -> str:
        return "true" if flag else "false"

    cmd = [
        "python3",
        "train.py",
        f"model={model_config}",
        f"model.use_lora={b(use_lora)}",
        "model.use_baseline_head=false",
        "loss=arc_bpo",
        f"loss.beta={beta}",
        f"loss.delta_star={delta_star}",
        f"loss.T={arc_t}",
        f"loss.kappa={kappa}",
        f"loss.sba_lambda={sba_lambda}",
        f"loss.sba_scale={sba_scale}",
        f"loss.exp_clip={exp_clip}",
        f"loss.use_advantage_shape={b(use_advantage_shape)}",
        f"loss.fallback_to_uniform_shape={b(fallback_to_uniform_shape)}",
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
        f"eval_batch_size={eval_batch_size}",
        f"n_eval_examples={n_eval_examples}",
        f"save_checkpoint={b(save_checkpoint)}",
        f"save_every_examples={save_every_examples}",
        f"do_first_eval={b(do_first_eval)}",
        f"activation_checkpointing={b(activation_checkpointing)}",
        f"output_dir={OUTPUT_DIR}",
        f"wandb.enabled={b(wandb_enabled)}",
    ]

    if n_examples and n_examples > 0:
        cmd.append(f"n_examples={n_examples}")
        cmd.append("n_epochs=null")

    print("[RUN] " + " ".join(cmd))
    subprocess.run(cmd, check=True)

    output_volume.commit()
    print("[DONE] Training finished; outputs committed to volume.")

    if hf_repo_id:
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
            commit_message=f"Upload ARC-BPO advantage-shape Mistral checkpoint ({RUN_VERSION})",
        )
        print(f"[HF UPLOAD] Done: https://huggingface.co/{hf_repo_id}")

    return OUTPUT_DIR


@app.function(
    image=image,
    volumes={VOLUME_ROOT: output_volume},
    timeout=60 * 5,
)
def ls_outputs():
    """List the Mistral advantage-shape training run directories saved in the volume."""
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


@app.function(image=image, gpu="A100", timeout=60 * 5)
def _check_versions():
    import torch
    import transformers

    print(f"torch: {torch.__version__}")
    print(f"torch cuda: {torch.version.cuda}")
    print(f"cuda available: {torch.cuda.is_available()}")
    print(f"gpu count: {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        print(f"gpu 0: {torch.cuda.get_device_name(0)}")
    print(f"transformers: {transformers.__version__}")


@app.local_entrypoint()
def check_versions():
    """Print torch/CUDA/transformers versions inside the Modal image."""
    _check_versions.remote()


@app.local_entrypoint()
def main(
    delta_star: float = 2.5,
    arc_t: float = 2.0,
    kappa: float = 2.0,
    batch_size: int = 32,
    grad_accum: int = 4,
    n_epochs: int = 1,
    n_examples: int = 0,
    eval_batch_size: int = 16,
    n_eval_examples: int = 64,
    max_length: int = 2048,
    save_every_examples: int = 5000,
    do_first_eval: bool = True,
    fallback_to_uniform_shape: bool = True,
    use_lora: bool = True,
    wandb_enabled: bool = False,
    hf_repo_id: str = "",
    hf_private: bool = True,
    hf_upload_adapter_only: bool = True,
    smoke: bool = False,
):
    """Launch ARC-BPO Mistral training with ArmoRM advantage shape on Modal."""
    if smoke:
        if n_examples == 0:
            n_examples = 128
        batch_size = min(batch_size, 16)
        grad_accum = min(grad_accum, 4)
        n_eval_examples = min(n_eval_examples, 16)
        eval_batch_size = min(eval_batch_size, 8)
        do_first_eval = True

    print(f"[LAUNCH] {APP_NAME} ({RUN_VERSION})")
    print(
        f"[MODE] {'SMOKE TEST' if smoke else 'FULL RUN'} "
        f"(n_examples={n_examples or 'full epoch'}, batch_size={batch_size}, "
        f"T={arc_t}, kappa={kappa}, use_lora={use_lora})"
    )
    train.spawn(
        delta_star=delta_star,
        arc_t=arc_t,
        kappa=kappa,
        batch_size=batch_size,
        grad_accum=grad_accum,
        n_epochs=n_epochs,
        n_examples=n_examples,
        eval_batch_size=eval_batch_size,
        n_eval_examples=n_eval_examples,
        max_length=max_length,
        save_every_examples=save_every_examples,
        do_first_eval=do_first_eval,
        use_advantage_shape=True,
        fallback_to_uniform_shape=fallback_to_uniform_shape,
        use_lora=use_lora,
        wandb_enabled=wandb_enabled,
        hf_repo_id=hf_repo_id,
        hf_private=hf_private,
        hf_upload_adapter_only=hf_upload_adapter_only,
    )
    print("[LAUNCHED] Advantage-shape Mistral training is running on Modal.")
    print("[RESULTS] Inspect outputs with: modal run modal_train_mistral_adv.py::ls_outputs")
