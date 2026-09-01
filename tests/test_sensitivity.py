import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

try:
    from omegaconf import OmegaConf
except ImportError:  # Minimal unit-test environments may omit training/Hydra deps.
    OmegaConf = None

from evaluate_sensitivity import build_lm_eval_command, extract_task_score
from run_sensitivity import execution_preflight, hf_checkpoint_path, upload_checkpoint_to_hf
try:
    from preference_datasets import get_batch_iterator
except ModuleNotFoundError as error:
    if error.name != "datasets":
        raise
    get_batch_iterator = None
from sensitivity.common import (
    RunSpec,
    audit_run_config,
    build_llama3_10k_bs64_base,
    build_run_specs,
    normalize_base_config,
    patch_run_config,
    validate_sensitivity_base,
)
from summarize_sensitivity import aggregate, make_latex, make_plots, write_markdown


def resolved_base(use_advantage_shape=True):
    return OmegaConf.create(
        {
            "seed": 42,
            "exp_name": "main",
            "local_run_dir": "output/main",
            "output_dir": "output",
            "fsdp_port": 12345,
            "datasets": "dataset/repo",
            "dataset_train_split": "train",
            "dataset_test_split": "test",
            "lr": 5e-7,
            "batch_size": 64,
            "gradient_accumulation_steps": 4,
            "label_noise_rate": 0.0,
            "label_noise_seed": 17,
            "label_noise_indices_path": None,
            "model": {"name_or_path": "base/model", "use_lora": True},
            "loss": {
                "name": "arc_bpo",
                "beta": 0.1,
                "delta_star": 2.0,
                "T": 2.0,
                "kappa": 2.0,
                "sba_lambda": 1.0,
                "min_tokens_per_chunk": 4,
                "max_tokens_per_chunk": 64,
                "use_advantage_shape": use_advantage_shape,
                "winsorize_advantages": True,
            },
        }
    )


@unittest.skipIf(OmegaConf is None, "omegaconf is an optional training dependency")
class SensitivityConfigTest(unittest.TestCase):
    def test_two_gpu_preset_keeps_global_and_per_gpu_batch_contract(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = build_llama3_10k_bs64_base(
                Path(__file__).resolve().parents[1],
                Path(temporary),
                seed=0,
                gradient_accumulation_steps=8,
            )
        self.assertEqual(base.batch_size, 64)
        self.assertEqual(base.gradient_accumulation_steps, 8)
        result = execution_preflight(
            base,
            visible_gpus=2,
            gpu_names=["NVIDIA A100-SXM4-80GB"] * 2,
            expected_gpus=2,
            expected_gpu_name="A100",
        )
        self.assertEqual(result["per_gpu_microbatch"], 4)
        self.assertEqual(result["optimizer_steps"], 157)

    def test_uniform_main_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "uniform allocation"):
            validate_sensitivity_base(resolved_base(use_advantage_shape=False))

    def test_grid_and_one_factor_audits(self):
        base = normalize_base_config(resolved_base())
        validate_sensitivity_base(base)
        specs = build_run_specs(base, ("T", "kappa", "delta0", "lambda"), (42,), 0.2)
        self.assertEqual(len(specs), 19)
        self.assertEqual({spec.seed for spec in specs}, {42})
        self.assertEqual(
            {spec.value_label for spec in specs if spec.sweep == "T"},
            {"4", "2", "1", "0.5"},
        )
        self.assertEqual(
            {spec.value_label for spec in specs if spec.sweep == "kappa"},
            {"3", "2", "1.5", "1"},
        )
        self.assertEqual(
            {spec.noise_rate for spec in specs if spec.sweep == "lambda"},
            {0.0},
        )
        missing_only = build_run_specs(
            base,
            ("T", "kappa", "delta0", "lambda"),
            (42,),
            0.2,
            include_default_points=False,
        )
        self.assertEqual(len(missing_only), 14)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for spec in specs:
                config = patch_run_config(base, spec, root, 2026, root / "noise.json")
                audit = audit_run_config(base, config, spec)
                self.assertTrue(audit["passed"])
                self.assertEqual(config.lr, base.lr)
                self.assertEqual(config.batch_size, base.batch_size)
                self.assertEqual(config.loss.beta, base.loss.beta)
                if spec.noise_rate == 0:
                    self.assertEqual(config.label_noise_seed, base.label_noise_seed)

            self.assertTrue(all(spec.value is not None for spec in specs))
            self.assertTrue(
                all(
                    spec.sweep == "kappa" or spec.noise_rate == 0.0
                    for spec in specs
                )
            )

    def test_audit_rejects_an_unrelated_change(self):
        base = normalize_base_config(resolved_base())
        spec = RunSpec("T", "T", "1", 1.0, 42, 0.0, 0)
        with tempfile.TemporaryDirectory() as temporary:
            config = patch_run_config(base, spec, Path(temporary), 2026, Path(temporary) / "n")
            config.lr = 1e-4
            with self.assertRaisesRegex(ValueError, "lr"):
                audit_run_config(base, config, spec)


@unittest.skipIf(get_batch_iterator is None, "datasets is an optional training dependency")
class LabelNoiseManifestTest(unittest.TestCase):
    @staticmethod
    def fake_dataset(*_args, **_kwargs):
        return {
            "p0": {
                "pairs": [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)],
                "prompt_dict": [],
                "responses": [],
                "sft_target": [],
                "response_scores": [],
            }
        }

    def test_manifest_is_saved_and_validated_for_replay(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest_path = Path(temporary) / "noise.json"
            kwargs = {
                "hf_dataset_repo_names": "dataset/repo",
                "tokenizer": None,
                "split": "train",
                "n_epochs": 0,
                "label_noise_rate": 0.2,
                "label_noise_seed": 2026,
                "label_noise_indices_path": str(manifest_path),
                "silent": True,
            }
            with mock.patch("preference_datasets.get_dataset_from_hf", self.fake_dataset):
                list(get_batch_iterator(**kwargs))
                first = json.loads(manifest_path.read_text(encoding="utf-8"))
                list(get_batch_iterator(**kwargs))
                second = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(first, second)
            self.assertEqual(first["total_pairs"], 5)

            kwargs["label_noise_rate"] = 0.3
            with mock.patch("preference_datasets.get_dataset_from_hf", self.fake_dataset):
                with self.assertRaisesRegex(ValueError, "rate mismatch"):
                    list(get_batch_iterator(**kwargs))


class EvaluationAndSummaryTest(unittest.TestCase):
    def test_two_a100_preflight_preserves_microbatch_four(self):
        config = SimpleNamespace(batch_size=64, gradient_accumulation_steps=8, n_examples=10000)
        result = execution_preflight(
            config,
            visible_gpus=2,
            gpu_names=["NVIDIA A100-SXM4-80GB"] * 2,
            expected_gpus=2,
            expected_gpu_name="A100",
        )
        self.assertEqual(result["global_batch_size"], 64)
        self.assertEqual(result["per_gpu_microbatch"], 4)
        self.assertEqual(result["optimizer_steps"], 157)

    def test_four_a100_preflight_reports_effective_batching(self):
        config = SimpleNamespace(batch_size=64, gradient_accumulation_steps=4, n_examples=10000)
        result = execution_preflight(
            config,
            visible_gpus=4,
            gpu_names=["NVIDIA A100-SXM4-80GB"] * 4,
            expected_gpus=4,
            expected_gpu_name="A100",
        )
        self.assertEqual(result["per_gpu_microbatch"], 4)
        self.assertEqual(result["optimizer_steps"], 157)
        self.assertEqual(result["full_batch_examples"], 10048)

    def test_preflight_rejects_wrong_gpu_or_indivisible_batch(self):
        config = SimpleNamespace(batch_size=64, gradient_accumulation_steps=4, n_examples=10000)
        with self.assertRaisesRegex(RuntimeError, "Expected 4"):
            execution_preflight(
                config,
                visible_gpus=2,
                gpu_names=["NVIDIA A100"] * 2,
                expected_gpus=4,
            )
        with self.assertRaisesRegex(RuntimeError, "mismatched devices"):
            execution_preflight(
                config,
                visible_gpus=4,
                gpu_names=["NVIDIA H100"] * 4,
                expected_gpu_name="A100",
            )
        config.batch_size = 63
        with self.assertRaisesRegex(ValueError, "must be divisible"):
            execution_preflight(
                config,
                visible_gpus=4,
                gpu_names=["NVIDIA A100"] * 4,
            )

    def test_hf_checkpoint_paths_are_stable_and_readable(self):
        self.assertEqual(
            hf_checkpoint_path("sens_T_0.5_clean_seed0"),
            "checkpoints/sens_T_0.5_clean_seed0",
        )
        self.assertEqual(
            hf_checkpoint_path("sens_kappa_3_noise20_seed0"),
            "checkpoints/sens_kappa_3_noise20_seed0",
        )

    def test_hf_upload_stages_adapter_and_audit_files(self):
        class FakeApi:
            def __init__(self):
                self.call = None

            def upload_folder(self, **kwargs):
                root = Path(kwargs["folder_path"])
                self.call = {
                    **kwargs,
                    "files": sorted(
                        str(path.relative_to(root)).replace("\\", "/")
                        for path in root.rglob("*")
                        if path.is_file()
                    ),
                }

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "LATEST"
            adapter = checkpoint / "adapter"
            adapter.mkdir(parents=True)
            (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
            (adapter / "adapter_model.safetensors").write_bytes(b"w" * 2048)
            run_dir = root / "run"
            run_dir.mkdir()
            config_path = run_dir / "resolved_config.yaml"
            config_path.write_text("seed: 0\n", encoding="utf-8")
            (run_dir / "config_diff.json").write_text("{}", encoding="utf-8")
            row = {
                "run_name": "sens_T_4_clean_seed0",
                "sweep": "T",
                "parameter": "T",
                "value": "4",
                "seed": "0",
                "noise_rate": "0.0",
                "scientific_hash": "abc123",
                "config_path": str(config_path),
            }
            api = FakeApi()
            destination, url = upload_checkpoint_to_hf(
                api=api,
                repo_id="owner/repo",
                row=row,
                checkpoint=checkpoint,
                adapter_only=True,
            )

        self.assertEqual(destination, "checkpoints/sens_T_4_clean_seed0")
        self.assertEqual(
            url,
            "https://huggingface.co/owner/repo/tree/main/checkpoints/sens_T_4_clean_seed0",
        )
        self.assertEqual(api.call["path_in_repo"], destination)
        self.assertEqual(
            api.call["files"],
            [
                "adapter/adapter_config.json",
                "adapter/adapter_model.safetensors",
                "checkpoint_metadata.json",
                "config_diff.json",
                "resolved_config.yaml",
            ],
        )

    def test_metric_extraction_uses_approved_keys(self):
        self.assertEqual(
            extract_task_score(
                {"results": {"arc_challenge": {"acc_norm,none": 0.625}}},
                "arc",
            ),
            0.625,
        )
        self.assertEqual(
            extract_task_score({"groups": {"mmlu": {"acc,none": 0.5}}}, "mmlu"),
            0.5,
        )

    def test_lm_eval_command_freezes_task_fewshot(self):
        args = SimpleNamespace(
            model_backend="vllm",
            tensor_parallel_size=2,
            dtype="bfloat16",
            gpu_memory_utilization=0.9,
            max_model_len=4096,
            lm_eval_command="lm_eval",
            batch_size="auto:4",
            evaluation_seed="0,1234,1234,1234",
            device="cuda:0",
            trust_remote_code=False,
            log_samples=True,
        )
        command = build_lm_eval_command(Path("model"), "hellaswag", Path("out"), args)
        self.assertEqual(command[command.index("--tasks") + 1], "hellaswag")
        self.assertEqual(command[command.index("--num_fewshot") + 1], "10")
        self.assertIn("--log_samples", command)

    def test_aggregation_computes_task_and_average_std(self):
        rows = []
        for seed, value in ((42, 60.0), (123, 64.0)):
            rows.append(
                {
                    "sweep": "T",
                    "parameter": "T",
                    "value": "2",
                    "numeric_value": "2.0",
                    "noise_rate": 0.0,
                    "seed": seed,
                    **{task: value for task in ("hellaswag", "arc", "mmlu", "truthfulqa", "winogrande", "gsm8k")},
                    "average": value,
                }
            )
        summary = aggregate(rows)
        self.assertEqual(summary[0]["average_mean"], 62.0)
        self.assertAlmostEqual(summary[0]["average_std"], 2.8284271247461903)
        self.assertEqual(summary[0]["n_seeds"], 2)

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            make_plots(summary, output)
            make_latex(summary, output)
            write_markdown(
                summary,
                missing=[],
                reproduction={"status": "not_checked", "reason": "test"},
                output_root=output,
            )
            self.assertTrue((output / "sensitivity_T.pdf").is_file())
            self.assertIn("\\begin{table*}", (output / "sensitivity_tables.tex").read_text())
            self.assertIn("not_checked", (output / "summary.md").read_text())


if __name__ == "__main__":
    unittest.main()
