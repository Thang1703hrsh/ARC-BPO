import inspect
import pathlib
import unittest

import torch

from loss.loss import arc_bpo_pair_loss
from loss.loss_utils import (
    compute_exact_chunk_log_ratios,
    construct_arc_bpo_one_sided_targets,
)


class ArcBPOTest(unittest.TestCase):
    def test_chunk_log_ratios_telescope(self):
        policy = torch.tensor([-1.0, -2.0, -0.5, -3.0])
        reference = torch.tensor([-1.5, -1.0, -0.25, -2.0])
        chunks, seq = compute_exact_chunk_log_ratios(
            policy,
            reference,
            [(0, 2), (2, 4)],
            beta=0.1,
        )

        expected = 0.1 * (policy - reference).sum()
        self.assertTrue(torch.allclose(chunks.sum(), seq, atol=1e-6))
        self.assertTrue(torch.allclose(seq, expected, atol=1e-6))

    def test_target_calibration(self):
        tau_w, tau_l, _, _ = construct_arc_bpo_one_sided_targets(
            n_chosen_chunks=2,
            n_rejected_chunks=3,
            delta_star=2.0,
        )

        self.assertTrue(torch.allclose(tau_w.sum() - tau_l.sum(), torch.tensor(2.0)))

    def test_detached_advantage_targets(self):
        chosen_adv = torch.tensor([0.0, 2.0], requires_grad=True)
        rejected_adv = torch.tensor([3.0, -1.0, 0.5], requires_grad=True)
        delta_star = torch.tensor(1.5, requires_grad=True)
        tau_w, tau_l, pi_w, rho_l = construct_arc_bpo_one_sided_targets(
            2,
            3,
            delta_star,
            chosen_advantages=chosen_adv,
            rejected_advantages=rejected_adv,
            use_advantage_shape=True,
        )

        self.assertFalse(tau_w.requires_grad)
        self.assertFalse(tau_l.requires_grad)
        self.assertFalse(pi_w.requires_grad)
        self.assertFalse(rho_l.requires_grad)
        self.assertTrue(torch.allclose(tau_w.sum() - tau_l.sum(), torch.tensor(1.5)))

    def test_gradient_sign_pulls_chunk_ratio_toward_target(self):
        chosen_a = torch.tensor([0.0], requires_grad=True)
        rejected_a = torch.tensor([-1.0], requires_grad=True)
        loss, _ = arc_bpo_pair_loss(chosen_a, rejected_a, delta_star=2.0)
        loss.backward()

        self.assertLess(chosen_a.grad.item(), 0.0)
        self.assertEqual(rejected_a.grad.item(), 0.0)

        chosen_a = torch.tensor([2.0], requires_grad=True)
        rejected_a = torch.tensor([-1.0], requires_grad=True)
        loss, _ = arc_bpo_pair_loss(chosen_a, rejected_a, delta_star=2.0)
        loss.backward()

        self.assertGreater(chosen_a.grad.item(), 0.0)

    def test_zero_loss_at_target(self):
        chosen_a = torch.tensor([0.5, 0.5], requires_grad=True)
        rejected_a = torch.full((4,), -0.25, requires_grad=True)

        loss, out = arc_bpo_pair_loss(chosen_a, rejected_a, delta_star=2.0)

        self.assertLess(loss.item(), 1e-10)
        self.assertTrue(torch.allclose(out["model_margin"], out["target_margin"]))

    def test_different_chunk_counts_are_allowed(self):
        chosen_a = torch.tensor([0.2, 0.4, 0.6], requires_grad=True)
        rejected_a = torch.tensor([-0.2], requires_grad=True)

        loss, out = arc_bpo_pair_loss(chosen_a, rejected_a, delta_star=2.0)
        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(out["tau_w"].numel(), 3)
        self.assertEqual(out["tau_l"].numel(), 1)
        self.assertIsNotNone(chosen_a.grad)
        self.assertIsNotNone(rejected_a.grad)

    def test_uniform_shape(self):
        tau_w, tau_l, pi_w, rho_l = construct_arc_bpo_one_sided_targets(
            n_chosen_chunks=4,
            n_rejected_chunks=2,
            delta_star=2.0,
            use_advantage_shape=False,
        )

        self.assertTrue(torch.allclose(pi_w, torch.full((4,), 0.25)))
        self.assertTrue(torch.allclose(rho_l, torch.full((2,), 0.5)))
        self.assertTrue(torch.allclose(tau_w, torch.full((4,), 0.25)))
        self.assertTrue(torch.allclose(tau_l, torch.full((2,), -0.5)))

    def test_arc_bpo_does_not_use_tbpo_w_value_head_or_token_matching(self):
        loss_source = inspect.getsource(arc_bpo_pair_loss)

        trainer_source = pathlib.Path("trainers.py").read_text(encoding="utf-8")
        method_start = trainer_source.index("def arc_bpo_concatenated_forward")
        method_end = trainer_source.index("def get_batch_metrics", method_start)
        method_source = trainer_source[method_start:method_end]

        forbidden_symbols = (
            "log_w",
            "w_t",
            "w_ij",
            "baseline_head",
            "value_head",
            "compute_tbpo_loss_mask",
            "Q_tbpo_get_batch_logps",
            "A_tbpo_get_batch_logps",
            "loss_mask",
        )
        for source in (loss_source, method_source):
            for symbol in forbidden_symbols:
                self.assertNotIn(symbol, source)


if __name__ == "__main__":
    unittest.main()
