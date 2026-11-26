"""Lightweight contract tests for the newly integrated TTA baselines."""

import unittest

import torch
from torch import nn

from main_test import _counter_delta
from tests.tta_baselines import (
    CoTTAAdapter,
    DeYOAdapter,
    configure_cotta_model,
    configure_deyo_model,
)


class _TinyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 1, bias=False)
        self.bn = nn.BatchNorm2d(4)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(4, 2)

    def forward(self, images):
        features = self.avgpool(self.bn(self.conv(images))).flatten(1)
        return self.fc(features)


class _NoopRestorer:
    def step(self):
        return 0


class ExternalBaselineTests(unittest.TestCase):
    def test_shared_counter_delta_allows_optional_baseline_metrics(self):
        self.assertEqual(_counter_delta({"samples": 8}, {"samples": 3}, "samples"), 5)
        self.assertEqual(_counter_delta({}, {}, "recap_only_metric"), 0)

    def test_cotta_uses_all_weight_and_bias_tensors(self):
        model = _TinyCNN()
        params, names = configure_cotta_model(model)
        self.assertIn("conv.weight", names)
        self.assertIn("fc.weight", names)
        self.assertEqual(len(params), len(names))

    def test_deyo_uses_bn_affine_only(self):
        model = _TinyCNN()
        params, names = configure_deyo_model(model)
        self.assertEqual(names, ["bn.weight", "bn.bias"])
        self.assertEqual(len(params), 2)

    def test_cotta_returns_pre_update_teacher_prediction(self):
        torch.manual_seed(1)
        model = _TinyCNN()
        params, _ = configure_cotta_model(model)
        optimizer = torch.optim.SGD(params, lr=1e-3)
        adapter = CoTTAAdapter(
            model,
            optimizer,
            _NoopRestorer(),
            augmentation_count=0,
            restore_probability=0.0,
        )
        images = torch.randn(4, 3, 8, 8)
        with torch.no_grad():
            expected = adapter.model_ema(images)
        logits, stats = adapter.adapt(images)
        self.assertTrue(torch.allclose(logits, expected))
        self.assertEqual(stats["selected"], 4)
        self.assertTrue(stats["updated"])

    def test_cotta_official_cifar_augmentation_preserves_shape(self):
        model = _TinyCNN()
        params, _ = configure_cotta_model(model)
        adapter = CoTTAAdapter(
            model,
            torch.optim.SGD(params, lr=1e-3),
            _NoopRestorer(),
            augmentation_count=1,
            input_mean=(0.5, 0.5, 0.5),
            input_std=(0.25, 0.25, 0.25),
        )
        images = torch.randn(2, 3, 32, 32)
        augmented = adapter._augment(images)
        self.assertEqual(augmented.shape, images.shape)
        self.assertTrue(torch.isfinite(augmented).all())

    def test_deyo_executes_patch_plpd_contract(self):
        torch.manual_seed(2)
        model = _TinyCNN()
        params, _ = configure_deyo_model(model)
        optimizer = torch.optim.SGD(params, lr=1e-3)
        adapter = DeYOAdapter(
            model,
            optimizer,
            _NoopRestorer(),
            num_classes=2,
            entropy_margin_factor=2.0,
            plpd_threshold=-1.0,
            patch_grid=2,
        )
        logits, stats = adapter.adapt(torch.randn(4, 3, 8, 8))
        self.assertEqual(tuple(logits.shape), (4, 2))
        self.assertEqual(stats["entropy_passed"], 4)
        self.assertEqual(stats["plpd_passed"], 4)
        self.assertTrue(stats["updated"])


if __name__ == "__main__":
    unittest.main()
