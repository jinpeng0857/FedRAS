"""Contract tests for the RGA (DeYO + ReCAP) adaptation path."""

import unittest
from types import SimpleNamespace

import torch
from torch import nn

from main_test import (
    RGAAdapter,
    apply_fedras_primary_config,
    rga_joint_reliability_weight,
    rga_sample_reliability,
    resolve_tta_method,
)


class _TinyRGAClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 2, 1, bias=False)
        self.bn = nn.BatchNorm2d(2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(2, 2)

    def forward(self, images):
        features = self.pool(self.bn(self.conv(images))).flatten(1)
        return self.fc(features)


class _CountingRestorer:
    def __init__(self):
        self.calls = 0

    def step(self):
        self.calls += 1
        return 0

    @property
    def stats(self):
        return {"restore_events": 0}


class RGAContractTests(unittest.TestCase):
    def test_fedras_primary_config_is_normalized_product_with_reliability_gate(self):
        args = SimpleNamespace(
            rga_normalize_by_weight=False,
            rga_weight_mode="sample_only",
            disable_dsr_reliability_gate=False,
        )
        apply_fedras_primary_config(args, "fedras")
        self.assertTrue(args.rga_normalize_by_weight)
        self.assertEqual(args.rga_weight_mode, "product")
        self.assertTrue(args.disable_dsr_reliability_gate)
        self.assertAlmostEqual(args.adapt_lr, 1.0e-3)
        self.assertEqual(args.fedras_bn_scope, "excluding_layer4")
        self.assertAlmostEqual(args.rga_hybrid_threshold, 0.2)
        self.assertAlmostEqual(args.rga_anchor_rectify_beta, 0.3)

    def _adapter(
        self, plpd_threshold=-1.0, entropy_margin_factor=2.0,
        rga_cue_ablation="full",
    ):
        model = _TinyRGAClassifier()
        model.train()
        params = list(model.parameters())
        optimizer = torch.optim.SGD(params, lr=1e-3)
        restorer = _CountingRestorer()
        adapter = RGAAdapter(
            model=model,
            optimizer=optimizer,
            feature_variance=torch.zeros(2),
            restorer=restorer,
            fishers={},
            tta_method="rga",
            deyo_entropy_margin_factor=entropy_margin_factor,
            deyo_plpd_threshold=plpd_threshold,
            deyo_patch_grid=2,
            view_consistency=True,
            rga_cue_ablation=rga_cue_ablation,
            dsr_grad_clip_norm=None,
        )
        return adapter, restorer

    def test_rga_is_resolvable_and_combines_soft_reliability_with_recap_loss(self):
        self.assertEqual(resolve_tta_method("rga"), "rga")
        adapter, restorer = self._adapter()
        logits, stats = adapter.adapt(torch.randn(4, 3, 8, 8))

        self.assertEqual(tuple(logits.shape), (4, 2))
        self.assertEqual(stats["deyo_entropy_passed"], 4)
        self.assertEqual(stats["plpd_passed"], 4)
        self.assertEqual(stats["selected"], 4)
        self.assertGreater(stats["recap_loss"], 0.0)
        self.assertTrue(stats["updated"])
        self.assertEqual(restorer.calls, 1)
        self.assertGreater(stats["recap_soft_weight_min"], 0.0)
        self.assertLessEqual(stats["recap_soft_weight_max"], 2.0)
        self.assertGreaterEqual(stats["rga_joint_weight_min"], 0.0)
        self.assertLessEqual(
            stats["rga_joint_weight_max"],
            stats["recap_soft_weight_max"],
        )

    def test_rga_does_not_hard_reject_samples_with_legacy_thresholds(self):
        adapter, restorer = self._adapter(
            plpd_threshold=2.0,
            entropy_margin_factor=-1.0,
        )
        _logits, stats = adapter.adapt(torch.randn(4, 3, 8, 8))

        self.assertEqual(stats["deyo_entropy_passed"], 0)
        self.assertEqual(stats["plpd_passed"], 0)
        self.assertEqual(stats["selected"], 4)
        self.assertTrue(stats["updated"])
        self.assertEqual(restorer.calls, 1)

    def test_rga_joint_weight_multiplies_sample_and_region_reliability(self):
        sample_reliability = torch.tensor([0.25, 0.80])
        regional_reliability = torch.tensor([2.00, 0.50])

        joint_weight = rga_joint_reliability_weight(
            sample_reliability,
            regional_reliability,
        )

        torch.testing.assert_close(joint_weight, torch.tensor([0.50, 0.40]))

    def test_rga_dsr_name_is_registered(self):
        self.assertEqual(resolve_tta_method("rga_dsr"), "rga_dsr")

    def test_integrated_reduced_variant_disables_weak_view_cue(self):
        adapter, _restorer = self._adapter(rga_cue_ablation="reduced")
        self.assertEqual(adapter.rga_cue_ablation, "reduced")
        self.assertFalse(adapter.view_consistency)

    def test_reduced_sample_reliability_uses_only_patch_shuffle_signal(self):
        entropy = torch.tensor([0.0, 4.0])
        plpd = torch.tensor([0.2, 0.8])
        probability = torch.tensor([[0.99, 0.01], [0.51, 0.49]])
        reliability, _margin = rga_sample_reliability(
            prediction_entropy=entropy,
            plpd=plpd,
            probability=probability,
            plpd_threshold=0.1,
            reliability_alpha=0.7,
            hybrid_threshold=None,
            cue_ablation="reduced",
        )
        expected = ((plpd - 0.1) / 0.9).clamp(0.0, 1.0)
        torch.testing.assert_close(reliability, expected)


if __name__ == "__main__":
    unittest.main()
