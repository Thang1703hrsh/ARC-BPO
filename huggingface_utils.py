from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional


DEFAULT_HF_NAMESPACE = "ducthang1703"


def resolve_hf_repo_id(repo_id: Optional[str], namespace: str = DEFAULT_HF_NAMESPACE) -> str:
    """Resolve a full Hugging Face model repository id.

    A short name is placed under ``namespace``.  Requiring a non-empty id when
    pushing prevents an accidental upload to a generated or ambiguous name.
    """
    value = str(repo_id or "").strip().strip("/")
    if not value:
        raise ValueError("huggingface.repo_id must be set when push_to_hub=true.")
    if "/" in value:
        return value
    owner = str(namespace or DEFAULT_HF_NAMESPACE).strip().strip("/")
    if not owner:
        raise ValueError("huggingface.namespace must not be empty.")
    return f"{owner}/{value}"


def checkpoint_upload_path(
    run_dir: str | os.PathLike[str],
    *,
    use_lora: bool,
    adapter_only: bool,
) -> Path:
    latest = Path(run_dir) / "LATEST"
    return latest / "adapter" if use_lora and adapter_only else latest


def push_final_checkpoint_to_hub(
    config: Any,
    run_dir: str | os.PathLike[str],
    *,
    use_lora: bool,
    api: Any = None,
) -> dict:
    """Upload the final checkpoint selected by the resolved training config."""
    hf_config = getattr(config, "huggingface", None)
    if hf_config is None:
        return {"status": "disabled"}
    if not bool(hf_config.push_to_hub):
        return {"status": "disabled"}

    repo_id = resolve_hf_repo_id(hf_config.repo_id, hf_config.namespace)
    adapter_only = bool(hf_config.adapter_only)
    upload_path = checkpoint_upload_path(
        run_dir,
        use_lora=use_lora,
        adapter_only=adapter_only,
    )
    if not upload_path.is_dir():
        raise FileNotFoundError(f"Hugging Face upload folder does not exist: {upload_path}")
    if use_lora and adapter_only:
        config_path = upload_path / "adapter_config.json"
        weights = list(upload_path.glob("adapter_model*.safetensors")) + list(
            upload_path.glob("adapter_model*.bin")
        )
        if not config_path.is_file() or not weights or any(path.stat().st_size <= 1024 for path in weights):
            raise RuntimeError(f"Incomplete LoRA adapter; refusing upload: {upload_path}")

    if api is None:
        from huggingface_hub import HfApi

        api = HfApi(token=os.environ.get("HF_TOKEN"))

    api.create_repo(
        repo_id=repo_id,
        repo_type="model",
        private=bool(hf_config.private),
        exist_ok=True,
    )
    api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=str(upload_path),
        commit_message=str(hf_config.commit_message),
    )
    return {
        "status": "uploaded",
        "repo_id": repo_id,
        "folder": str(upload_path),
        "url": f"https://huggingface.co/{repo_id}",
    }
