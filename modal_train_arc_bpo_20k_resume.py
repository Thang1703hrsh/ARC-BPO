"""
Resumable ARC-BPO LoRA training in 20,000-example stages.

This Modal app is intended for interrupted/limited-resource workflows:

1. Train exactly one 20k valid-example stage.
2. Upload the resulting LoRA adapter to Hugging Face.
3. Later, load that adapter as trainable and continue on the next 20k examples.

Supported model families:
  llama    -> RLHFlow/LLaMA3-SFT-v2 + princeton-nlp/llama3-ultrafeedback-armorm
  mistral  -> HuggingFaceH4/mistral-7b-sft-alpha + HuggingFaceH4/ultrafeedback_binarized
  qwen     -> Qwen/Qwen2.5-7B-Instruct + HuggingFaceH4/ultrafeedback_binarized

First Mistral stage, examples [0, 20000):
  modal run --detach modal_train_arc_bpo_20k_resume.py::main \
    --model-family mistral \
    --stage-index 0 \
    --hf-repo-id ducthang1703/mistral-arc-bpo-uniform-lora-20k \
    --no-hf-private

Second Mistral stage, examples [20000, 40000), continuing from the previous adapter:
  modal run --detach modal_train_arc_bpo_20k_resume.py::main \
    --model-family mistral \
    --stage-index 1 \
    --resume-adapter-repo ducthang1703/mistral-arc-bpo-uniform-lora-20k \
    --hf-repo-id ducthang1703/mistral-arc-bpo-uniform-lora-40k \
    --no-hf-private

Continue from an existing 10k adapter and train the next 20k examples [10000, 30000):
  modal run --detach modal_train_arc_bpo_20k_resume.py::main \
    --model-family mistral \
    --start-example 10000 \
    --resume-adapter-repo ducthang1703/mistral-arc-bpo-uniform-lora-10k \
    --hf-repo-id ducthang1703/mistral-arc-bpo-uniform-lora-30k \
    --no-hf-private

Inspect saved volume outputs:
  modal run modal_train_arc_bpo_20k_resume.py::ls_outputs
"""

from dataclasses import dataclass

import modal


APP_NAME = "tbpo-arc-bpo-20k-resume-train"
RUN_VERSION = "arc-bpo-20k-resume-v1"

REPO_DIR = "/root/arc_bpo"
VOLUME_ROOT = "/vol/output"
OUTPUT_ROOT = f"{VOLUME_ROOT}/resumable_arc_bpo_20k"
GPU_CONFIG = "H100:4"

EXAMPLES_PER_STAGE = 20_000


app = modal.App(APP_NAME)
output_volume = modal.Volume.from_name("tbpo-output", create_if_missing=True)


@dataclass(frozen=True)
class ModelPreset:
    model_config: str
    datasets_raw: str
    train_split: str
    test_split: str
    delta_star: float
    output_name: str


MODEL_PRESETS = {
    "llama": ModelPreset(
        model_config="llama_8b",
        datasets_raw="princeton-nlp/llama3-ultrafeedback-armorm",
        train_split="train",
        test_split="test",
        delta_star=2.5,
        output_name="llama",
    ),
    "mistral": ModelPreset(
        model_config="mistral_7b",
        datasets_raw="HuggingFaceH4/ultrafeedback_binarized",
        train_split="train_prefs",
        test_split="test_prefs",
        delta_star=2.5,
        output_name="mistral",
    ),
    "qwen": ModelPreset(
        model_config="qwen2_5_7b_instruct",
        datasets_raw="HuggingFaceH4/ultrafeedback_binarized",
        train_split="train_prefs",
        test_split="test_prefs",
        delta_star=2.0,
        output_name="qwen",
    ),
}


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


def _bool_arg(flag: bool) -> str:
    return "true" if flag else "false"


@app.function(
    image=image,
    gpu=GPU_CONFIG,
    volumes={VOLUME_ROOT: output_volume},
    secrets=_SECRETS,
    timeout=60 * 60 * 18,
)
def train_stage(
    model_family: str,
    stage_index: int,
    hf_repo_id: str,
    resume_adapter_repo: str = "",
    start_example: int = -1,
    examples_per_stage: int = EXAMPLES_PER_STAGE,
    batch_size: int = 32,
    grad_accum: int = 4,
    lr: str = "5e-7",
    weight_decay: float = 0.0,
    max_grad_norm: float = 10.0,
    optimizer: str = "RMSprop",
    scheduler: str = "cosine",
    warmup_ratio: float = 0.05,
    max_length: int = 2048,
    eval_batch_size: int = 16,
    n_eval_examples: int = 0,
    do_first_eval: bool = False,
    activation_checkpointing: bool = True,
    wandb_enabled: bool = False,
    hf_private: bool = True,
    hf_upload_adapter_only: bool = True,
    arc_t: float = 2.0,
    kappa: float = 2.0,
    sba_lambda: float = 1.0,
    sba_scale: float = 4.0,
    exp_clip: float = 30.0,
    use_advantage_shape: bool = False,
    fallback_to_uniform_shape: bool = True,
    min_tokens_per_chunk: int = 4,
    max_tokens_per_chunk: int = 64,
):
    import os
    import subprocess

    model_family = model_family.lower().strip()
    if model_family not in MODEL_PRESETS:
        raise ValueError(
            f"Unknown model_family={model_family!r}. "
            f"Choose one of: {', '.join(sorted(MODEL_PRESETS))}"
        )
    if stage_index < 0:
        raise ValueError("stage_index must be >= 0.")
    if start_example < -1:
        raise ValueError("start_example must be >= 0, or -1 to derive it from stage_index.")
    if examples_per_stage <= 0:
        raise ValueError("examples_per_stage must be positive.")
    if not hf_repo_id:
        raise ValueError("hf_repo_id is required so every stage is saved to Hugging Face.")
    effective_start_example = start_example if start_example >= 0 else stage_index * examples_per_stage
    if effective_start_example > 0 and not resume_adapter_repo:
        raise ValueError(
            "Continuing after example 0 requires resume_adapter_repo, "
            "e.g. the HF repo from the previous stage."
        )

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
    if examples_per_stage % batch_size != 0:
        raise ValueError(
            "examples_per_stage should be divisible by batch_size to keep each stage exact: "
            f"examples_per_stage={examples_per_stage}, batch_size={batch_size}."
        )

    preset = MODEL_PRESETS[model_family]
    skip_examples = effective_start_example
    stage_output_dir = (
        f"{OUTPUT_ROOT}/{preset.output_name}/"
        f"start_{skip_examples:08d}_count_{examples_per_stage:08d}"
    )

    os.chdir(REPO_DIR)
    os.makedirs(stage_output_dir, exist_ok=True)

    if not wandb_enabled:
        os.environ["WANDB_MODE"] = "disabled"
    os.environ["HYDRA_FULL_ERROR"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    print(f"[VERSION] {RUN_VERSION}")
    print(f"[GPU] {GPU_CONFIG}")
    print(f"[MODEL FAMILY] {model_family}")
    print(f"[STAGE] index={stage_index} examples=[{skip_examples}, {skip_examples + examples_per_stage})")
    print(f"[RESUME ADAPTER] {resume_adapter_repo or '<fresh LoRA>'}")
    print(f"[HF SAVE] {hf_repo_id}")
    print(f"[OUTPUT] {stage_output_dir}")

    cmd = [
        "python3",
        "train.py",
        f"model={preset.model_config}",
        "model.use_lora=true",
        "model.use_baseline_head=false",
        "loss=arc_bpo",
        "loss.beta=0.1",
        f"loss.delta_star={preset.delta_star}",
        f"loss.T={arc_t}",
        f"loss.kappa={kappa}",
        f"loss.sba_lambda={sba_lambda}",
        f"loss.sba_scale={sba_scale}",
        f"loss.exp_clip={exp_clip}",
        f"loss.use_advantage_shape={_bool_arg(use_advantage_shape)}",
        f"loss.fallback_to_uniform_shape={_bool_arg(fallback_to_uniform_shape)}",
        f"loss.min_tokens_per_chunk={min_tokens_per_chunk}",
        f"loss.max_tokens_per_chunk={max_tokens_per_chunk}",
        f"datasets={preset.datasets_raw}",
        f"dataset_train_split={preset.train_split}",
        f"dataset_test_split={preset.test_split}",
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
        f"n_examples={examples_per_stage}",
        "n_epochs=null",
        f"skip_examples={skip_examples}",
        f"eval_batch_size={eval_batch_size}",
        f"n_eval_examples={n_eval_examples}",
        "save_checkpoint=true",
        f"do_first_eval={_bool_arg(do_first_eval)}",
        f"activation_checkpointing={_bool_arg(activation_checkpointing)}",
        f"output_dir={stage_output_dir}",
        f"wandb.enabled={_bool_arg(wandb_enabled)}",
    ]
    if resume_adapter_repo:
        cmd.append(f"model.adapter_path={resume_adapter_repo}")

    print("[RUN] " + " ".join(cmd))
    subprocess.run(cmd, check=True)

    output_volume.commit()
    print("[DONE] Training finished; outputs committed to Modal volume.")

    from huggingface_hub import HfApi, upload_folder

    latest_dirs = []
    for root, dirs, _ in os.walk(stage_output_dir):
        if "LATEST" in dirs:
            latest_path = os.path.join(root, "LATEST")
            latest_dirs.append((os.path.getmtime(latest_path), latest_path))
    if not latest_dirs:
        raise RuntimeError(f"No LATEST checkpoint found under {stage_output_dir}")

    latest_path = max(latest_dirs)[1]
    upload_path = latest_path
    if hf_upload_adapter_only:
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
            f"Upload ARC-BPO {model_family} stage {stage_index} "
            f"({skip_examples}-{skip_examples + examples_per_stage})"
        ),
    )
    print(f"[HF UPLOAD] Done: https://huggingface.co/{hf_repo_id}")

    return {
        "stage_output_dir": stage_output_dir,
        "hf_repo_id": hf_repo_id,
        "resume_adapter_repo": resume_adapter_repo,
        "skip_examples": skip_examples,
        "examples_per_stage": examples_per_stage,
    }


@app.function(image=image, volumes={VOLUME_ROOT: output_volume}, timeout=60 * 5)
def ls_outputs():
    import os

    output_volume.reload()
    if not os.path.isdir(OUTPUT_ROOT):
        print(f"No outputs yet at {OUTPUT_ROOT}")
        return
    for root, dirs, files in os.walk(OUTPUT_ROOT):
        depth = root[len(OUTPUT_ROOT) :].count(os.sep)
        if depth > 3:
            continue
        indent = "  " * depth
        print(f"{indent}{os.path.basename(root) or OUTPUT_ROOT}/")
        for f in files:
            print(f"{indent}  {f}")


@app.local_entrypoint()
def main(
    model_family: str = "mistral",
    stage_index: int = 0,
    hf_repo_id: str = "",
    resume_adapter_repo: str = "",
    start_example: int = -1,
    examples_per_stage: int = EXAMPLES_PER_STAGE,
    batch_size: int = 32,
    grad_accum: int = 4,
    n_eval_examples: int = 0,
    do_first_eval: bool = False,
    hf_private: bool = True,
    wandb_enabled: bool = False,
):
    """Launch one resumable 20k ARC-BPO LoRA training stage."""
    print(f"[LAUNCH] {APP_NAME} ({RUN_VERSION})")
    print(
        f"[MODE] model={model_family} stage={stage_index} "
        f"examples_per_stage={examples_per_stage} batch_size={batch_size}"
    )
    train_stage.spawn(
        model_family=model_family,
        stage_index=stage_index,
        hf_repo_id=hf_repo_id,
        resume_adapter_repo=resume_adapter_repo,
        start_example=start_example,
        examples_per_stage=examples_per_stage,
        batch_size=batch_size,
        grad_accum=grad_accum,
        n_eval_examples=n_eval_examples,
        do_first_eval=do_first_eval,
        hf_private=hf_private,
        wandb_enabled=wandb_enabled,
    )
    print("[LAUNCHED] Resumable 20k training stage is running on Modal.")
    print("[RESULTS] Inspect outputs with: modal run modal_train_arc_bpo_20k_resume.py::ls_outputs")
