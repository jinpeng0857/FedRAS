"""Tests for BN configuration, ReCAP region proxy, DLRS, and DualReliabilityAdapter."""

import math
import unittest

import torch
from torch import nn

from adapters import (
    ASRAdapter,
    DLRSAdapter,
    DisabledSparseRestorer,
    DualReliabilityAdapter,
    RegionConfidenceProxy,
    dlrs_joint_reliability_weight,
    dlrs_sample_reliability,
    recap_region_soft_weight,
    top1_top2_probability_margin,
)
from evaluator.engine import (
    configure_bn_adaptation,
    configure_feddr_bn_adaptation,
    configure_official_bn_adaptation,
)
from sparse_restore import PeriodicSparseRestorer


class _TinyBNStages(nn.Module):
    def __init__(self):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(2)
        self.layer1 = nn.Sequential(nn.BatchNorm2d(2))
        self.layer2 = nn.Sequential(nn.BatchNorm2d(2))
        self.layer3 = nn.Sequential(nn.BatchNorm2d(2))
        self.layer4 = nn.Sequential(nn.BatchNorm2d(2))
        self.fc = nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            self.fc.weight.copy_(torch.eye(2))

    def forward(self, inputs):
        outputs = self.bn1(inputs)
        outputs = self.layer1(outputs)
        outputs = self.layer2(outputs)
        outputs = self.layer3(outputs)
        outputs = self.layer4(outputs)
        features = outputs.mean(dim=(2, 3))
        return self.fc(features)


class _NoopRestorer:
    def __init__(self):
        self.reliability_threshold = 0.55
        self.stats = {
            "restore_events": 0,
            "scheduled_restore_events": 0,
            "skipped_low_drift_events": 0,
            "skipped_low_reliability_events": 0,
            "budget_event_counts": {},
            "restored_assignments_total": 0,
        }

    def step(self, reliability=None, entropy=None, anchor_disagreement=None, gradient_clipped=False):
        return 0


class BNConfigurationTests(unittest.TestCase):
    def test_configure_bn_adaptation_excludes_layer4(self):
        model = _TinyBNStages()
        params, names = configure_bn_adaptation(model)
        self.assertTrue(params)
        self.assertTrue(names)
        self.assertFalse(any("layer4" in name for name in names))
        self.assertFalse(model.layer4[0].weight.requires_grad)

    def test_configure_official_bn_adaptation_includes_all(self):
        model = _TinyBNStages()
        params, names = configure_official_bn_adaptation(model)
        self.assertTrue(params)
        self.assertTrue(names)
        self.assertTrue(any("layer4" in name for name in names))
        self.assertTrue(model.layer4[0].weight.requires_grad)

    def test_configure_feddr_bn_adaptation_excludes_layer4(self):
        model = _TinyBNStages()
        params, names = configure_feddr_bn_adaptation(model)
        self.assertTrue(params)
        self.assertTrue(names)
        self.assertFalse(any("layer4" in name for name in names))
        self.assertFalse(model.layer4[0].weight.requires_grad)


class ProxyAndDLRSMathTests(unittest.TestCase):
    def test_region_confidence_proxy_shapes_and_values(self):
        weight = torch.eye(2)
        variance = torch.tensor([1.0, 1.0])
        proxy = RegionConfidenceProxy(weight, variance)
        logits = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
        re, ri = proxy(logits)
        self.assertEqual(re.shape, (2,))
        self.assertEqual(ri.shape, (2,))
        self.assertTrue(torch.isfinite(re).all())
        self.assertTrue(torch.isfinite(ri).all())

    def test_dlrs_sample_reliability_bounds(self):
        prob = torch.tensor([[0.9, 0.1], [0.55, 0.45]])
        entropy = -(prob * prob.log()).sum(dim=1)
        plpd = torch.tensor([0.4, 0.1])
        rel, margin = dlrs_sample_reliability(
            prediction_entropy=entropy,
            plpd=plpd,
            probability=prob,
            plpd_threshold=0.2,
            reliability_alpha=0.7,
        )
        self.assertEqual(rel.shape, (2,))
        self.assertEqual(margin.shape, (2,))
        self.assertTrue((rel >= 0.0).all() and (rel <= 1.0).all())
        self.assertGreater(rel[0], rel[1])

    def test_dlrs_sample_reliability_hybrid_thresholding(self):
        prob = torch.tensor([[0.9, 0.1], [0.55, 0.45]])
        entropy = -(prob * prob.log()).sum(dim=1)
        plpd = torch.tensor([0.4, 0.0])
        rel, _ = dlrs_sample_reliability(
            prediction_entropy=entropy,
            plpd=plpd,
            probability=prob,
            plpd_threshold=0.2,
            reliability_alpha=0.7,
            hybrid_threshold=0.3,
        )
        self.assertGreater(rel[0], 0.0)
        self.assertEqual(rel[1], 0.0)

    def test_dlrs_joint_reliability_weight_product(self):
        sample_rel = torch.tensor([0.8, 0.2])
        reg_rel = torch.tensor([1.5, 0.5])
        joint = dlrs_joint_reliability_weight(sample_rel, reg_rel)
        expected = torch.tensor([1.2, 0.1])
        self.assertTrue(torch.allclose(joint, expected, atol=1e-5))


class AdapterExecutionTests(unittest.TestCase):
    def test_dlrs_adapter_adapt_step(self):
        model = _TinyBNStages()
        params, _ = configure_feddr_bn_adaptation(model)
        optimizer = torch.optim.SGD(params, lr=1e-3)
        adapter = DLRSAdapter(
            model=model,
            optimizer=optimizer,
            feature_variance=torch.ones(2),
            restorer=_NoopRestorer(),
            fishers={},
            tta_method="dlrs",
            early_exit_enabled=True,
        )
        inputs = torch.randn(4, 2, 8, 8)
        logits, stats = adapter.adapt(inputs)
        self.assertEqual(logits.shape, (4, 2))
        self.assertTrue(stats["updated"])
        self.assertEqual(stats["samples"], 4)
        self.assertIn("batch_entropy", stats)
        self.assertIn("batch_mean_reliability", stats)
        self.assertIn("anchor_disagreement", stats)
        self.assertIn("sb_ratio", stats)
        self.assertIn("ess_ratio", stats)

    def test_dlrs_adapter_bounded_scale_normalization(self):
        model = _TinyBNStages()
        params, _ = configure_feddr_bn_adaptation(model)
        optimizer = torch.optim.SGD(params, lr=1e-3)
        adapter = DLRSAdapter(
            model=model,
            optimizer=optimizer,
            feature_variance=torch.ones(2),
            restorer=_NoopRestorer(),
            fishers={},
            tta_method="dlrs",
            normalize_by_weight=True,
            c_max=2.0,
        )
        inputs = torch.randn(4, 2, 8, 8)
        logits, stats = adapter.adapt(inputs)
        self.assertTrue(stats["updated"])
        self.assertTrue(math.isfinite(stats["loss"]))

    def test_dlrs_adapter_skip_update_on_low_quality(self):
        model = _TinyBNStages()
        params, _ = configure_feddr_bn_adaptation(model)
        optimizer = torch.optim.SGD(params, lr=1e-3)
        # Set high skip threshold to force skip
        adapter = DLRSAdapter(
            model=model,
            optimizer=optimizer,
            feature_variance=torch.ones(2),
            restorer=_NoopRestorer(),
            fishers={},
            tta_method="dlrs",
            skip_update_tau_s=0.99,
        )
        inputs = torch.randn(4, 2, 8, 8)
        logits, stats = adapter.adapt(inputs)
        self.assertFalse(stats["updated"])
        self.assertTrue(stats["skipped_low_quality"])

    def test_asr_adapter_adapt_step(self):
        model = _TinyBNStages()
        params, names = configure_official_bn_adaptation(model)
        optimizer = torch.optim.SGD(params, lr=1e-3)
        adapter = ASRAdapter(
            model=model,
            optimizer=optimizer,
            feature_variance=torch.ones(2),
            restorer=_NoopRestorer(),
            fishers={},
            param_names=names,
        )
        inputs = torch.randn(4, 2, 8, 8)
        logits, stats = adapter.adapt(inputs)
        self.assertEqual(logits.shape, (4, 2))
        self.assertTrue(stats["updated"])
        self.assertIn("asr_entropy", stats)

    def test_dlrs_adapter_online_fisher_gradient_accumulation(self):
        model = _TinyBNStages()
        params, names = configure_feddr_bn_adaptation(model)
        optimizer = torch.optim.SGD(params, lr=1e-3)
        fishers = {
            name: (torch.ones_like(param), torch.zeros_like(param))
            for name, param in model.named_parameters()
            if param.requires_grad
        }
        restorer = PeriodicSparseRestorer(
            model=model,
            optimizer=optimizer,
            fishers=fishers,
            interval=1,
            online_fisher_beta=0.5,
            event_tau_r=0.0,
        )
        adapter = DLRSAdapter(
            model=model,
            optimizer=optimizer,
            feature_variance=torch.ones(2),
            restorer=restorer,
            fishers=fishers,
            tta_method="dlrs_fslr",
            deyo_plpd_threshold=-1.0,
        )
        inputs = torch.randn(4, 2, 8, 8)
        _logits, stats = adapter.adapt(inputs)
        self.assertTrue(stats["updated"])
        # Verify that online gradient accumulators accumulated non-zero gradients in real adaptation
        total_online_acc = sum(
            float(acc.sum().item())
            for acc in restorer._online_fisher_accumulators.values()
        )
        self.assertGreater(total_online_acc, 0.0)

    def test_dlrs_hybrid_sample_filter_naming(self):
        model = _TinyBNStages()
        params, names = configure_feddr_bn_adaptation(model)
        optimizer = torch.optim.SGD(params, lr=1e-3)
        fishers = {
            name: (torch.ones_like(p), torch.zeros_like(p))
            for name, p in model.named_parameters()
            if p.requires_grad
        }
        restorer = PeriodicSparseRestorer(
            model=model, optimizer=optimizer, fishers=fishers, interval=10
        )
        adapter_hybrid = DLRSAdapter(
            model=model,
            optimizer=optimizer,
            feature_variance=torch.ones(2),
            restorer=restorer,
            fishers=fishers,
            tta_method="dlrs_fslr",
            hybrid_threshold=0.2,
        )
        self.assertEqual(adapter_hybrid.sample_filter, "hybrid_soft_hard_tau0.2")

        adapter_soft = DLRSAdapter(
            model=model,
            optimizer=optimizer,
            feature_variance=torch.ones(2),
            restorer=restorer,
            fishers=fishers,
            tta_method="dlrs_fslr",
            hybrid_threshold=None,
        )
        self.assertEqual(adapter_soft.sample_filter, "soft_reliability_all_samples")

    def test_dlrs_diagnostic_counters(self):
        model = _TinyBNStages()
        params, names = configure_feddr_bn_adaptation(model)
        optimizer = torch.optim.SGD(params, lr=1e-3)
        fishers = {
            name: (torch.ones_like(p), torch.zeros_like(p))
            for name, p in model.named_parameters()
            if p.requires_grad
        }
        restorer = PeriodicSparseRestorer(
            model=model, optimizer=optimizer, fishers=fishers, interval=10
        )
        adapter = DLRSAdapter(
            model=model,
            optimizer=optimizer,
            feature_variance=torch.ones(2),
            restorer=restorer,
            fishers=fishers,
            tta_method="dlrs_fslr",
            deyo_entropy_margin_factor=10.0,
            deyo_plpd_threshold=-1.0,
        )
        inputs = torch.randn(4, 2, 8, 8)
        _logits, stats = adapter.adapt(inputs)
        self.assertTrue(stats["updated"])
        self.assertIn("dlrs_effective_sample_size", stats)
        self.assertGreater(adapter.counters["effective_sample_size_sum"], 0.0)
        self.assertEqual(adapter.counters["effective_sample_size_batches"], 1)

    def test_dlrs_anchor_rectification(self):
        model = _TinyBNStages()
        params, names = configure_feddr_bn_adaptation(model)
        optimizer = torch.optim.SGD(params, lr=1e-3)
        fishers = {
            name: (torch.ones_like(p), torch.zeros_like(p))
            for name, p in model.named_parameters()
            if p.requires_grad
        }
        restorer = PeriodicSparseRestorer(
            model=model, optimizer=optimizer, fishers=fishers, interval=10
        )
        adapter = DLRSAdapter(
            model=model,
            optimizer=optimizer,
            feature_variance=torch.ones(2),
            restorer=restorer,
            fishers=fishers,
            tta_method="dlrs_fslr",
            anchor_rectify_beta=0.3,
            deyo_entropy_margin_factor=10.0,
            deyo_plpd_threshold=-1.0,
        )
        self.assertTrue(adapter.needs_anchor)
        self.assertIsNotNone(adapter.anchor_model)

        inputs = torch.randn(4, 2, 8, 8)
        _logits, stats = adapter.adapt(inputs)
        self.assertIn("dlrs_effective_sample_size", stats)

    def test_fslr_continuous_risk_budget(self):
        model = _TinyBNStages()
        params, names = configure_feddr_bn_adaptation(model)
        optimizer = torch.optim.SGD(params, lr=1e-3)
        fishers = {
            name: (torch.ones_like(p), torch.zeros_like(p))
            for name, p in model.named_parameters()
            if p.requires_grad
        }
        restorer = PeriodicSparseRestorer(
            model=model,
            optimizer=optimizer,
            fishers=fishers,
            interval=1,
            budget_mode="continuous_risk",
            risk_rho_min=0.0,
            risk_rho_max=0.10,
        )
        # Test zero-drift step (e.g. early adaptation / single domain)
        restorer.step(reliability=0.2, entropy=0.1, anchor_disagreement=0.80, ess_ratio=0.2)
        self.assertIsNotNone(restorer.last_risk_score)
        self.assertEqual(restorer.last_risk_score, 0.0)
        self.assertEqual(restorer.last_selected_ratio, 0.0)

        # Induce parameter drift
        for p in model.parameters():
            p.data.add_(0.2)

        # Test high risk step with drift
        restorer.step(reliability=0.2, entropy=0.1, anchor_disagreement=0.80, ess_ratio=0.2)
        high_risk = restorer.last_risk_score
        high_ratio = restorer.last_selected_ratio
        self.assertGreater(high_risk, 0.0)
        self.assertGreater(high_ratio, 0.0)
        self.assertLessEqual(high_ratio, 0.10)

    def test_dlrs_composite_weak_perturbation_and_view_consistency(self):
        from adapters.dlrs import create_composite_weak_perturbation
        images = torch.randn(4, 3, 32, 32)
        weak_images = create_composite_weak_perturbation(images)
        self.assertEqual(weak_images.shape, images.shape)
        self.assertFalse(torch.allclose(weak_images, images))

        # Test DLRSAdapter with view_consistency enabled
        model = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(16, 10),
        )
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        adapter = DLRSAdapter(
            model=model,
            optimizer=optimizer,
            feature_variance=torch.ones(16),
            restorer=DisabledSparseRestorer(model=model),
            fishers={},
            tta_method="dlrs",
            view_consistency=True,
            view_alpha=0.70,
            view_js_scale=1.0,
        )
        outputs, metrics = adapter.adapt(images)
        self.assertEqual(outputs.shape, (4, 10))
        self.assertIsInstance(metrics, dict)
        self.assertGreater(adapter.counters["r_view_count"], 0)
        self.assertGreater(adapter.counters["forward_passes_extra"], 0)

    def test_fslr_hierarchical_layer_budget(self):
        model = nn.Sequential(
            nn.BatchNorm2d(16),
            nn.BatchNorm2d(32),
        )
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        fishers = {
            name: (torch.ones_like(p) * (1.0 if "0" in name else 10.0), torch.zeros_like(p))
            for name, p in model.named_parameters()
            if p.requires_grad
        }
        restorer = PeriodicSparseRestorer(
            model=model,
            optimizer=optimizer,
            fishers=fishers,
            interval=1,
            budget_mode="continuous_risk",
            hierarchical_budget=True,
            layer_quota_cap=0.35,
            risk_rho_min=0.0,
            risk_rho_max=0.10,
        )
        # Induce drift across both layers
        for p in model.parameters():
            p.data.add_(0.15)

        # Trigger restore step
        recovered = restorer.step(reliability=0.5, entropy=0.1, anchor_disagreement=0.5, ess_ratio=0.5)
        self.assertGreater(recovered, 0)
        # Verify both layers got allocated restoration quota
        for name, p in model.named_parameters():
            self.assertIn(name, restorer.selected)

    def test_fslr_direction_aware_momentum_discounting(self):
        model = nn.Sequential(nn.BatchNorm2d(16))
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        # Initialize momentum buffers
        for p in model.parameters():
            loss = p.sum()
            loss.backward()
        optimizer.step()

        fishers = {
            name: (torch.ones_like(p), torch.zeros_like(p))
            for name, p in model.named_parameters()
            if p.requires_grad
        }
        restorer = PeriodicSparseRestorer(
            model=model,
            optimizer=optimizer,
            fishers=fishers,
            interval=1,
            budget_mode="continuous_risk",
            momentum_discounting=True,
            momentum_away_factor=0.20,
            risk_rho_min=0.0,
            risk_rho_max=0.10,
        )
        # Induce drift
        for p in model.parameters():
            p.data.add_(0.10)

        # Trigger restore step
        recovered = restorer.step(reliability=0.5, entropy=0.1, anchor_disagreement=0.5, ess_ratio=0.5)
        self.assertGreater(recovered, 0)
        self.assertGreater(restorer.momentum_away_damped_count + restorer.momentum_towards_kept_count, 0)


if __name__ == "__main__":
    unittest.main()




