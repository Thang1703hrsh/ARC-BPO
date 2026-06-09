import os
import subprocess
from pathlib import Path

import modal


APP_NAME = "tbpo-sft-mistral-ultrafeedback"
REMOTE_REPO_DIR = "/root/TBPO"
VOLUME_DIR = "/vol"


app = modal.App(APP_NAME)

cache_volume = modal.Volume.from_name("tbpo-cache", create_if_missing=True)
output_volume = modal.Volume.from_name("tbpo-output", create_if_missing=True)


def _ignore_repo_files(path: Path) -> bool:
    parts = set(path.parts)
    return bool(
        parts
        & {
            ".git",
            "__pycache__",
            "output",
            "winrate_results",
            "processed_data",
            "env",
        }
    )


image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-devel-ubuntu22.04",
        add_python="3.11",
    )
    .apt_install("git", "build-essential")
    .pip_install(
        "torch==2.6.0+cu124",
        "torchvision==0.21.0+cu124",
        "torchaudio==2.6.0+cu124",
        extra_index_url="https://download.pytorch.org/whl/cu124",
    )
    .pip_install(
        "transformers==4.52.4",
        "datasets==3.5.0",
        "accelerate==1.7.0",
        "peft==0.14.0",
        "hydra-core==1.3.2",
        "omegaconf==2.3.0",
        "wandb==0.20.1",
        "tqdm",
        "sentencepiece==0.2.0",
        "protobuf",
    )
    .env(
        {
            "HF_HOME": f"{VOLUME_DIR}/cache/huggingface",
            "TRANSFORMERS_CACHE": f"{VOLUME_DIR}/cache/huggingface/transformers",
            "HF_DATASETS_CACHE": f"{VOLUME_DIR}/cache/huggingface/datasets",
            "XDG_CACHE_HOME": f"{VOLUME_DIR}/cache",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTHONUNBUFFERED": "1",
        }
    )
    .workdir(REMOTE_REPO_DIR)
    .add_local_dir(
        ".",
        REMOTE_REPO_DIR,
        ignore=_ignore_repo_files,
    )
)


@app.function(
    image=image,
    gpu="H100:2",
    timeout=60 * 60 * 24,
    volumes={
        f"{VOLUME_DIR}/cache": cache_volume,
        f"{VOLUME_DIR}/output": output_volume,
    },
)
def train_sft_mistral_ultrafeedback(
    use_lora: bool = False,
    n_epochs: int = 1,
    batch_size: int = 8,
    grad_accum: int = 4,
    lr: str = "2e-4",
    max_length: int = 2048,
    do_first_eval: bool = True,
    n_eval_examples: int = 64,
    wandb_enabled: bool = False,
):
    """Run the SFT baseline from TBPO.tex on 2 Modal H100 GPUs."""
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    os.environ.setdefault("WANDB_MODE", "disabled" if not wandb_enabled else "online")

    cmd = [
        "python",
        "train.py",
        "model=mistral_7b",
        f"model.use_lora={str(use_lora).lower()}",
        "model.use_baseline_head=false",
        "loss=sft",
        "datasets=HuggingFaceH4/ultrafeedback_binarized",
        "dataset_train_split=train_prefs",
        "dataset_test_split=test_prefs",
        "trainer=FSDPTrainer",
        f"batch_size={batch_size}",
        "eval_batch_size=8",
        f"gradient_accumulation_steps={grad_accum}",
        f"lr={lr}",
        "weight_decay=0.0",
        "max_grad_norm=10.0",
        "optimizer=AdamW",
        "scheduler=cosine",
        "warmup_ratio=0.05",
        "minimum_log_interval_secs=1.0",
        f"max_length={max_length}",
        f"n_epochs={n_epochs}",
        f"n_eval_examples={n_eval_examples}",
        f"do_first_eval={str(do_first_eval).lower()}",
        "activation_checkpointing=true",
        f"output_dir={VOLUME_DIR}/output",
        f"wandb.enabled={str(wandb_enabled).lower()}",
    ]

    print("Running command:")
    print(" ".join(cmd))
    subprocess.run(cmd, cwd=REMOTE_REPO_DIR, check=True)

    output_volume.commit()
    cache_volume.commit()
    return f"Finished. Check the Modal Volume tbpo-output mounted at {VOLUME_DIR}/output."


@app.local_entrypoint()
def main(
    use_lora: bool = False,
    n_epochs: int = 1,
    batch_size: int = 8,
    grad_accum: int = 4,
    lr: str = "2e-4",
    max_length: int = 2048,
    do_first_eval: bool = True,
    n_eval_examples: int = 64,
    wandb_enabled: bool = False,
):
    print(
        train_sft_mistral_ultrafeedback.remote(
            use_lora=use_lora,
            n_epochs=n_epochs,
            batch_size=batch_size,
            grad_accum=grad_accum,
            lr=lr,
            max_length=max_length,
            do_first_eval=do_first_eval,
            n_eval_examples=n_eval_examples,
            wandb_enabled=wandb_enabled,
        )
    )
