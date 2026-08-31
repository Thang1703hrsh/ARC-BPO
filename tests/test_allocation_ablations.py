import tempfile
import unittest
from pathlib import Path

from arc_bpo_scores import extract_preference_scores
from run_allocation_ablations import (
    ADVANTAGE_BLOCKER,
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

    def test_advantage_row_is_explicitly_blocked(self):
        self.assertFalse(VARIANTS["advantage"].supported)
        self.assertEqual(VARIANTS["advantage"].reason, ADVANTAGE_BLOCKER)

    def test_valid_variants_differ_only_in_allocation_flags(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base = base_environment(0, "0,1,2,3", 4, root)
            uniform = scientific_environment(
                run_environment(base, VARIANTS["uniform"], 0, root)
            )
            no_winsor = scientific_environment(
                run_environment(base, VARIANTS["advantage_sba_no_winsor"], 0, root)
            )
            self.assertEqual(
                config_diff(uniform, no_winsor),
                ["USE_ADVANTAGE_SHAPE: 'false' -> 'true'"],
            )
            self.assertEqual(uniform["SEED"], "0")
            self.assertEqual(no_winsor["SEED"], "0")
            self.assertEqual(uniform["BATCH_SIZE"], "64")
            self.assertEqual(uniform["N_EXAMPLES"], "10000")
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

    def test_modal_launcher_preserves_four_gpu_controlled_setting(self):
        source = Path("modal_allocation_ablations.py").read_text(encoding="utf-8")
        self.assertIn('GPU_TYPE = "A100-80GB:4"', source)
        self.assertIn("GPU_COUNT = 4", source)
        self.assertIn('gradient_accumulation_steps: int = 4', source)
        self.assertIn('seed: int = 0', source)
        self.assertIn('global_batch_size: int = 64', source)
        self.assertIn('"--reuse_uniform_checkpoint"', source)
        self.assertIn('revisions["base_revision"]', source)
        self.assertIn('revisions["dataset_revision"]', source)
        self.assertIn(
            "ducthang1703/llama3-arc-bpo-uniform-lora-10k-bs64",
            source,
        )

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
