import unittest
import tempfile
import pathlib
from pathlib import Path
from types import SimpleNamespace

import torch

from analyze_credit_drift import (
    assign_tertile_groups,
    correlation_payload,
    bootstrap_group_kl,
    build_credit_targets,
    grouped_credit_summary,
    observed_logps_and_forward_kl,
    _plot_outputs,
    _write_summary,
)
from diversity_metrics.prepare_prompts import extract_prompt, prepare_prompt_records


class CreditDriftAnalysisTest(unittest.TestCase):
    def test_modal_launcher_is_pinned_to_one_a100_and_requested_checkpoint(self):
        source = pathlib.Path("modal_generation_diversity.py").read_text(encoding="utf-8")
        self.assertIn('gpu="A100"', source)
        self.assertNotIn('gpu="A100:', source)
        self.assertIn("ducthang1703/llama3-arc-bpo-uniform-lora-10k-bs64", source)
        self.assertIn('"--tensor_parallel_size",\n                    1,', source)

    def test_heldout_diversity_prompts_preserve_user_messages(self):
        valid = {
            "id": "source-7",
            "chosen": [
                {"role": "user", "content": "Explain robust optimization."},
                {"role": "assistant", "content": "Chosen answer"},
            ],
            "rejected": [
                {"role": "user", "content": "Explain robust optimization."},
                {"role": "assistant", "content": "Rejected answer"},
            ],
        }
        invalid = {"chosen": [], "rejected": []}

        extracted = extract_prompt(valid, source_index=7)
        self.assertEqual(extracted["source_id"], "source-7")
        self.assertEqual(extracted["prompt_messages"], valid["chosen"][:-1])
        records = prepare_prompt_records([invalid, valid, valid], 1, False, seed=42)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["id"], 0)

    def test_logratio_credit_targets_are_calibrated(self):
        chosen = torch.tensor([0.2, 0.7, -0.1])
        rejected = torch.tensor([-0.4, 0.3])
        tau_w, tau_l, pi_w, rho_l = build_credit_targets(
            chosen,
            rejected,
            allocation_mode="logratio",
            delta0=2.5,
            temperature=2.0,
            kappa=2.0,
        )

        self.assertTrue(torch.allclose(pi_w.sum(), torch.tensor(1.0)))
        self.assertTrue(torch.allclose(rho_l.sum(), torch.tensor(1.0)))
        self.assertTrue(torch.all(tau_w >= 0))
        self.assertTrue(torch.all(tau_l <= 0))
        self.assertTrue(torch.allclose(tau_w.sum() - tau_l.sum(), torch.tensor(2.5)))
        self.assertFalse(tau_w.requires_grad)
        self.assertFalse(tau_l.requires_grad)

    def test_uniform_mode_matches_public_target_shape(self):
        chosen = torch.tensor([1.0, 2.0])
        rejected = torch.tensor([-1.0, -2.0, -3.0, -4.0])
        tau_w, tau_l, pi_w, rho_l = build_credit_targets(
            chosen,
            rejected,
            allocation_mode="uniform",
            delta0=2.0,
            temperature=2.0,
            kappa=2.0,
        )

        self.assertTrue(torch.allclose(pi_w, torch.full((2,), 0.5)))
        self.assertTrue(torch.allclose(rho_l, torch.full((4,), 0.25)))
        self.assertTrue(torch.allclose(tau_w, torch.full((2,), 0.5)))
        self.assertTrue(torch.allclose(tau_l, torch.full((4,), -0.25)))

    def test_full_vocabulary_forward_kl_and_observed_logps(self):
        policy_logits = torch.tensor([[2.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
        reference_logits = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        observed = torch.tensor([0, 1])

        policy_logps, reference_logps, token_kl = observed_logps_and_forward_kl(
            policy_logits,
            reference_logits,
            observed,
            kl_device=torch.device("cpu"),
            token_batch_size=1,
        )

        expected_policy = torch.log_softmax(policy_logits, dim=-1)[range(2), observed]
        expected_reference = torch.log_softmax(reference_logits, dim=-1)[range(2), observed]
        policy_probs = torch.softmax(policy_logits, dim=-1)
        expected_kl = (
            policy_probs
            * (torch.log_softmax(policy_logits, dim=-1) - torch.log_softmax(reference_logits, dim=-1))
        ).sum(-1)

        self.assertTrue(torch.allclose(policy_logps, expected_policy))
        self.assertTrue(torch.allclose(reference_logps, expected_reference))
        self.assertTrue(torch.allclose(token_kl, expected_kl, atol=1e-6))
        self.assertGreater(token_kl[0].item(), 0.0)
        self.assertAlmostEqual(token_kl[1].item(), 0.0, places=7)

    def test_tertiles_use_fixed_value_thresholds_and_keep_ties_together(self):
        rows = [{"credit": value} for value in (0.1, 0.2, 0.2, 0.8, 1.0, 1.2)]
        thresholds = assign_tertile_groups(rows, "credit", "group")

        self.assertLessEqual(thresholds["lower_tertile"], thresholds["upper_tertile"])
        tied_groups = {row["group"] for row in rows if row["credit"] == 0.2}
        self.assertEqual(tied_groups, {"Low"})
        self.assertEqual(rows[-1]["group"], "High")

    def test_bootstrap_resamples_preference_pairs_not_individual_chunks(self):
        rows = [
            {"example_index": 0, "credit_group": "Low", "policy_reference_kl": 1.0},
            {"example_index": 0, "credit_group": "Low", "policy_reference_kl": 3.0},
            {"example_index": 1, "credit_group": "High", "policy_reference_kl": 9.0},
            {"example_index": 2, "credit_group": "Medium", "policy_reference_kl": 5.0},
        ]
        first = bootstrap_group_kl(rows, 100, 0.95, seed=42)
        second = bootstrap_group_kl(rows, 100, 0.95, seed=42)

        self.assertEqual(first, second)
        self.assertGreater(first["Low"]["valid_replicates"], 0)
        self.assertGreater(first["Medium"]["valid_replicates"], 0)
        self.assertGreater(first["High"]["valid_replicates"], 0)

    def test_grouped_summary_handles_an_empty_tied_group_honestly(self):
        rows = []
        for example_index, credit in enumerate((0.5, 0.5, 1.0)):
            rows.append(
                {
                    "example_index": example_index,
                    "credit_magnitude": credit,
                    "allocation_weight": credit,
                    "policy_reference_kl": credit * 2,
                }
            )
        assign_tertile_groups(rows, "credit_magnitude", "credit_group")
        grouped = grouped_credit_summary(rows, 20, 0.95, seed=7)

        self.assertEqual([row["credit_group"] for row in grouped], ["Low", "Medium", "High"])
        self.assertEqual(sum(row["num_chunks"] for row in grouped), len(rows))
        empty = [row for row in grouped if row["num_chunks"] == 0]
        for row in empty:
            self.assertIsNone(row["mean_policy_reference_kl"])

    def test_reviewer_plots_and_markdown_summary_are_created(self):
        rows = []
        for index, credit in enumerate((0.1, 0.2, 0.4, 0.6, 0.8, 1.0)):
            rows.append(
                {
                    "example_index": index // 2,
                    "side": "winner" if index % 2 == 0 else "loser",
                    "credit_magnitude": credit,
                    "allocation_weight": credit / 1.5,
                    "num_tokens": index + 2,
                    "policy_reference_kl": credit * 0.5,
                }
            )
        threshold = assign_tertile_groups(rows, "credit_magnitude", "credit_group")
        assign_tertile_groups(rows, "allocation_weight", "allocation_group")
        grouped = grouped_credit_summary(rows, 20, 0.95, seed=3)
        correlations = correlation_payload(rows)
        args = SimpleNamespace(
            policy_model="policy",
            reference_model="reference",
            dataset="dataset",
            split="test",
            allocation_mode="logratio",
        )
        diagnostics = {
            "processed_examples": 3,
            "num_chunks": 6,
            "max_calibration_error": 0.0,
            "mean_calibration_error": 0.0,
            "min_raw_token_kl": 0.0,
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            output = Path(tmp_dir)
            _plot_outputs(output, rows, grouped, correlations, 42, 100)
            _write_summary(
                output / "summary.md",
                args,
                grouped,
                correlations,
                diagnostics,
                {"credit_magnitude": threshold},
            )
            self.assertGreater((output / "credit_group_kl.pdf").stat().st_size, 0)
            self.assertGreater((output / "credit_vs_kl.pdf").stat().st_size, 0)
            summary = (output / "summary.md").read_text(encoding="utf-8")
            self.assertIn("Credit-Conditioned Policy Drift", summary)
            self.assertIn("Spearman rho", summary)


if __name__ == "__main__":
    unittest.main()
