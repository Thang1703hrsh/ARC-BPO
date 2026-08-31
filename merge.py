"""
Script to merge adapter weights (e.g., LoRA) into a base model.
Supports loading from HuggingFace Hub and pushing back to Hub.
"""

import argparse
import os
import tempfile

import torch
from huggingface_hub import HfApi, login
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def merge_adapter(
    base_model_path: str,
    adapter_path: str,
    base_revision: str = None,
    output_path: str = None,
    device: str = "auto",
    torch_dtype: str = "auto",
    max_shard_size: str = "5GB",
    push_to_hub: bool = False,
    hub_repo_id: str = None,
    hub_folder: str = None,
    hub_token: str = None,
    hub_private: bool = False,
):
    """
    Merge adapter weights into base model and optionally push to HuggingFace Hub.

    Args:
        base_model_path: Path or HF repo ID of the base model
        adapter_path: Path or HF repo ID (with optional subfolder) of the adapter
        output_path: Local path to save merged model (optional if pushing to hub)
        device: Device to load model on ('auto', 'cpu', 'cuda')
        torch_dtype: Data type for model weights ('auto', 'float16', 'bfloat16', 'float32')
        max_shard_size: Maximum size of each shard when saving
        push_to_hub: Whether to push to HuggingFace Hub
        hub_repo_id: HuggingFace Hub repository ID (e.g., 'username/repo-name')
        hub_folder: Subfolder within the repo to push to (optional)
        hub_token: HuggingFace Hub token for authentication
        hub_private: Whether to create a private repo
    """

    # Login to HuggingFace Hub if pushing
    if push_to_hub:
        if hub_token:
            login(token=hub_token)
        print("Authenticated with HuggingFace Hub")

    print(f"Loading base model from: {base_model_path}")

    # Determine dtype
    dtype_map = {
        "auto": "auto",
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    dtype = dtype_map.get(torch_dtype, "auto")

    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        revision=base_revision,
        torch_dtype=dtype,
        device_map=device,
        trust_remote_code=True,
    )

    print(f"Loading adapter from: {adapter_path}")

    # Handle adapter path with subfolder (format: repo_id/subfolder)
    adapter_repo = adapter_path
    adapter_subfolder = None

    # Check if adapter_path contains a subfolder
    # If it's from Hub and has more than one '/', it might have a subfolder
    if "/" in adapter_path and not os.path.exists(adapter_path):
        parts = adapter_path.split("/")
        if len(parts) > 2:
            # Format: username/repo/subfolder or username/repo/sub/folder
            adapter_repo = f"{parts[0]}/{parts[1]}"
            adapter_subfolder = "/".join(parts[2:])
            print(f"  Repository: {adapter_repo}")
            print(f"  Subfolder: {adapter_subfolder}")

    # Load adapter and merge
    model = PeftModel.from_pretrained(
        base_model,
        adapter_repo,
        subfolder=adapter_subfolder,
    )

    print("Merging adapter weights into base model...")
    model = model.merge_and_unload()

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        revision=base_revision,
        trust_remote_code=True,
    )

    # Save locally if output_path is provided
    if output_path:
        os.makedirs(output_path, exist_ok=True)
        print(f"Saving merged model locally to: {output_path}")

        model.save_pretrained(
            output_path,
            max_shard_size=max_shard_size,
            safe_serialization=True,
        )
        tokenizer.save_pretrained(output_path)
        print(f"✓ Model saved locally to: {output_path}")

    # Push to HuggingFace Hub if requested
    if push_to_hub:
        if not hub_repo_id:
            raise ValueError("hub_repo_id is required when push_to_hub=True")

        print(f"\nPushing to HuggingFace Hub: {hub_repo_id}")
        if hub_folder:
            print(f"  Subfolder: {hub_folder}")

        # Save to a temporary directory first, then upload with proper path
        with tempfile.TemporaryDirectory() as tmp_dir:
            print("Saving model to temporary directory...")
            model.save_pretrained(
                tmp_dir,
                max_shard_size=max_shard_size,
                safe_serialization=True,
            )
            tokenizer.save_pretrained(tmp_dir)

            # Use HfApi to upload with proper subfolder support
            api = HfApi()

            # Create the repo if it doesn't exist
            api.create_repo(
                repo_id=hub_repo_id,
                private=hub_private,
                exist_ok=True,
            )

            # Upload the folder to the specified path
            api.upload_folder(
                folder_path=tmp_dir,
                repo_id=hub_repo_id,
                path_in_repo=hub_folder if hub_folder else "",
            )

        repo_url = f"https://huggingface.co/{hub_repo_id}"
        if hub_folder:
            repo_url += f"/tree/main/{hub_folder}"

        print(f"✓ Model pushed to Hub: {repo_url}")

    print("\n✓ Merge completed successfully!")


def main():
    parser = argparse.ArgumentParser(
        description="Merge adapter weights into base model and optionally push to HuggingFace Hub"
    )
    parser.add_argument(
        "--base_model",
        type=str,
        required=True,
        help="HuggingFace model ID or local path (e.g., 'meta-llama/Llama-2-7b-hf')",
    )
    parser.add_argument(
        "--adapter",
        type=str,
        required=True,
        help="Adapter path: local path or HF repo with subfolder (e.g., 'username/repo/checkpoints/final')",
    )
    parser.add_argument(
        "--base_revision",
        type=str,
        default=None,
        help="Optional immutable Hugging Face revision for the base model.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Local path to save merged model (optional if using --push_to_hub)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to load model on (default: auto)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
        help="Data type for model weights (default: auto)",
    )
    parser.add_argument(
        "--max_shard_size",
        type=str,
        default="5GB",
        help="Maximum size of each shard when saving (default: 5GB)",
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Push merged model to HuggingFace Hub",
    )
    parser.add_argument(
        "--hub_repo_id",
        type=str,
        default=None,
        help="HuggingFace Hub repository ID (e.g., 'username/merged-models')",
    )
    parser.add_argument(
        "--hub_folder",
        type=str,
        default=None,
        help="Subfolder in the Hub repo to push to (e.g., 'llama-7b-merged')",
    )
    parser.add_argument(
        "--hub_token",
        type=str,
        default="",
        help="HuggingFace Hub token (or set HF_TOKEN environment variable)",
    )
    parser.add_argument(
        "--hub_private",
        action="store_true",
        help="Make the Hub repository private",
    )

    args = parser.parse_args()

    # Validate arguments
    if args.push_to_hub and not args.hub_repo_id:
        parser.error("--hub_repo_id is required when --push_to_hub is set")

    if not args.output and not args.push_to_hub:
        parser.error("Either --output or --push_to_hub must be specified")

    merge_adapter(
        base_model_path=args.base_model,
        adapter_path=args.adapter,
        base_revision=args.base_revision,
        output_path=args.output,
        device=args.device,
        torch_dtype=args.dtype,
        max_shard_size=args.max_shard_size,
        push_to_hub=args.push_to_hub,
        hub_repo_id=args.hub_repo_id,
        hub_folder=args.hub_folder,
        hub_token=args.hub_token,
        hub_private=args.hub_private,
    )


if __name__ == "__main__":
    main()
