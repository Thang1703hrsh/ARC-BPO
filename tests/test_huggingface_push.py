import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from huggingface_utils import (
    checkpoint_upload_path,
    push_final_checkpoint_to_hub,
    resolve_hf_repo_id,
)


class FakeHfApi:
    def __init__(self):
        self.created = []
        self.uploaded = []

    def create_repo(self, **kwargs):
        self.created.append(kwargs)

    def upload_folder(self, **kwargs):
        self.uploaded.append(kwargs)


class HuggingFacePushTests(unittest.TestCase):
    def test_short_repo_name_uses_project_namespace(self):
        self.assertEqual(
            resolve_hf_repo_id("arc-bpo-test", "ducthang1703"),
            "ducthang1703/arc-bpo-test",
        )
        self.assertEqual(
            resolve_hf_repo_id("another/model", "ducthang1703"),
            "another/model",
        )

    def test_lora_upload_uses_exact_final_adapter(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            adapter = run_dir / "LATEST" / "adapter"
            adapter.mkdir(parents=True)
            (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
            (adapter / "adapter_model.safetensors").write_bytes(b"x" * 2048)
            config = SimpleNamespace(
                huggingface=SimpleNamespace(
                    push_to_hub=True,
                    namespace="ducthang1703",
                    repo_id="arc-bpo-test",
                    private=True,
                    adapter_only=True,
                    commit_message="test upload",
                )
            )
            api = FakeHfApi()
            result = push_final_checkpoint_to_hub(
                config,
                run_dir,
                use_lora=True,
                api=api,
            )
            self.assertEqual(result["repo_id"], "ducthang1703/arc-bpo-test")
            self.assertEqual(Path(api.uploaded[0]["folder_path"]), adapter)
            self.assertEqual(
                checkpoint_upload_path(run_dir, use_lora=True, adapter_only=True),
                adapter,
            )

    def test_push_is_opt_in(self):
        config = SimpleNamespace(
            huggingface=SimpleNamespace(push_to_hub=False)
        )
        self.assertEqual(
            push_final_checkpoint_to_hub(config, "missing", use_lora=True),
            {"status": "disabled"},
        )


if __name__ == "__main__":
    unittest.main()
