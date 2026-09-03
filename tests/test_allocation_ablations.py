import tempfile
import unittest
from pathlib import Path

from arc_bpo_scores import extract_preference_scores
from run_allocation_ablations import (
    VARIANTS,
    adapter_checkpoint_complete,
    base_environment,
    config_diff,
    parse_variants,
    run_environment,
    scientific_environment,
)


class ScoreExtractionTests(unittest.TestCase):
    def test_direct_pair_scores(self):
        example = {"score_chosen": 1.25, "score_rejected": -0.5}
        self.assertEqual(extract_preference_scores(example, "c", "r"), (1.25, -0.5))

    def test_princeton_aligned_arrays_are_matched_by_text(self):
        example = {
            "all_generated_responses": ["middle", "rejected", "chosen"],
            "all_rm_scores": [0.2, -1.0, 2.5],
        }
        self.assertEqual(
            extract_preference_scores(example, "chosen", "rejected"),
            (2.5, -1.0),
        )

    def test_mismatched_response_is_not_assigned_global_extreme(self):
        example = {
            "all_generated_responses": ["a", "b"],
            "all_rm_scores": [3.0, -2.0],
        }
        self.assertEqual(extract_preference_scores(example, "missing", "b"), (None, -2.0))


class AllocationLauncherTests(unittest.TestCase):
    def test_parser_accepts_one_copy_of_known_variants(self):
        self.assertEqual(parse_variants("uniform,advantage"), ["uniform", "advantage"])
        with self.assertRaises(ValueError):
            parse_variants("uniform,uniform")

    def test_advantage_row_uses_quadratic_base_bregman(self):
        self.assertTrue(VARIANTS["advantage"].supported)
        self.assertEqual(VARIANTS["advantage"].env_patch["ARC_DIVERGENCE"], "quadratic")
        self.assertEqual(VARIANTS["advantage"].env_patch["USE_ADVANTAGE_SHAPE"], "true")

    def test_valid_variants_differ_only_in_allocation_flags(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base = base_environment(0, "0,1,2,3", 4, root)
            uniform = scientific_environment(
                run_environment(base, VARIANTS["uniform"], 0, root)
            )
            advantage = scientific_environment(
                run_environment(base, VARIANTS["advantage"], 0, root)
            )
            no_winsor = scientific_environment(
                run_environment(base, VARIANTS["advantage_sba_no_winsor"], 0, root)
            )
            self.assertEqual(
                config_diff(uniform, advantage),
                ["USE_ADVANTAGE_SHAPE: 'false' -> 'true'"],
            )
            self.assertEqual(
                config_diff(advantage, no_winsor),
                ["ARC_DIVERGENCE: 'quadratic' -> 'sba'"],
            )
            self.assertEqual(uniform["SEED"], "0")
            self.assertEqual(no_winsor["SEED"], "0")
            self.assertEqual(uniform["BATCH_SIZE"], "64")
            self.assertEqual(uniform["N_EXAMPLES"], "16000")
            self.assertEqual(uniform["SAVE_CHECKPOINT"], "false")
            self.assertEqual(uniform["SAVE_EVERY_EXAMPLES"], "0")
            self.assertEqual(uniform["MODEL_CONFIG"], "mistral_7b")
            self.assertEqual(
                uniform["DATASETS_RAW"],
                "HuggingFaceH4/ultrafeedback_binarized",
            )
            self.assertNotIn("MODEL_ADAPTER_PATH", uniform)

    def test_invalid_global_batch_decomposition_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(ValueError):
                base_environment(0, "0,1,2", 4, Path(temporary))

    def test_one_gpu_smoke_keeps_global_batch_64_with_microbatch_one(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = base_environment(
                0,
                "0",
                64,
                Path(temporary),
                n_examples=64,
                global_batch_size=64,
            )
            self.assertEqual(base["BATCH_SIZE"], "64")
            self.assertEqual(base["GRAD_ACCUM"], "64")
            self.assertEqual(base["N_EXAMPLES"], "64")

    def test_revisions_are_forwarded_to_the_public_launcher(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = base_environment(
                0,
                "0,1,2,3",
                4,
                Path(temporary),
                base_revision="base-sha",
                dataset_revision="dataset-sha",
            )
            self.assertEqual(base["MODEL_REVISION"], "base-sha")
            self.assertEqual(base["DATASET_REVISION"], "dataset-sha")

    def test_modal_launcher_uses_one_h100_with_microbatch_one(self):
        source = Path("modal_allocation_ablations.py").read_text(encoding="utf-8")
        self.assertIn('GPU_TYPE = "H100!"', source)
        self.assertIn("GPU_COUNT = 1", source)
        self.assertIn("CPU_COUNT = 4.0", source)
        self.assertIn("MEMORY_MIB = 64 * 1024", source)
        self.assertIn("cpu=CPU_COUNT", source)
        self.assertIn("memory=MEMORY_MIB", source)
        self.assertIn('gradient_accumulation_steps: int = 64', source)
        self.assertIn('"--gpu_ids",\n        "0",', source)
        self.assertIn('seed: int = 0', source)
        self.assertIn('global_batch_size: int = 64', source)
        self.assertIn('"uniform", "advantage", "advantage_sba_no_winsor"', source)
        self.assertIn("train_allocation_ablation.spawn(config)", source)
        self.assertIn('revisions["base_revision"]', source)
        self.assertIn('revisions["dataset_revision"]', source)
        self.assertIn(
            "ducthang1703/mistral7b-arc-bpo-uniform-lora-16k-bs64-seed0",
            source,
        )
        self.assertIn('BASE_MODEL = "HuggingFaceH4/mistral-7b-sft-alpha"', source)
        self.assertIn('DATASET_REPO = "HuggingFaceH4/ultrafeedback_binarized"', source)
        self.assertIn('arc_bpo_mistral.sh', source)

    def test_mistral_public_launcher_forwards_controlled_allocation_settings(self):
        source = Path("script/train/arc_bpo_mistral.sh").read_text(encoding="utf-8")
        self.assertIn('loss.divergence="${ARC_DIVERGENCE}"', source)
        self.assertIn('loss.winsorize_advantages="${WINSORIZE_ADVANTAGES}"', source)
        self.assertIn('loss.fallback_to_uniform_shape="${FALLBACK_TO_UNIFORM_SHAPE}"', source)
        self.assertIn('model.revision="${MODEL_REVISION}"', source)
        self.assertIn('dataset_revision="${DATASET_REVISION}"', source)
        self.assertIn('seed="${SEED}"', source)

    def test_adapter_completion_requires_config_and_nontrivial_weights(self):
        with tempfile.TemporaryDirectory() as temporary:
            adapter = Path(temporary)
            self.assertFalse(adapter_checkpoint_complete(adapter))
            (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
            (adapter / "adapter_model.safetensors").write_bytes(b"x" * 100)
            self.assertFalse(adapter_checkpoint_complete(adapter))
            (adapter / "adapter_model.safetensors").write_bytes(b"x" * 2048)
            self.assertTrue(adapter_checkpoint_complete(adapter))


if __name__ == "__main__":
    unittest.main()
