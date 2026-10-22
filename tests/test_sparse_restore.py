"""Targeted tests for drift-tiered, reliability-gated, layer-aware, event-triggered, and hybrid FSLR."""

import math
import unittest

import torch
from torch import nn

from sparse_restore import PeriodicSparseRestorer


class _BNModel(nn.Module):
    def __init__(self, channels=100):
        super().__init__()
        self.bn = nn.BatchNorm2d(channels)


class _LNModel(nn.Module):
    def __init__(self, normalized_shape=64):
        super().__init__()
        self.ln = nn.LayerNorm(normalized_shape)


def _build_restorer(
    drift,
    budget_mode="drift_tiered",
    reliability_threshold=0.55,
    disable_reliability_gate=False,
    event_trigger=False,
    event_k=3,
    event_rho_max=0.20,
    event_cooldown=20,
    online_fisher_beta=0.0,
):
    model = _BNModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    fishers = {
        "bn.weight": (
            torch.linspace(2.0, 3.0, model.bn.weight.numel()),
            torch.ones_like(model.bn.weight),
        ),
        "bn.bias": (
            torch.linspace(0.0, 1.0, model.bn.bias.numel()),
            torch.zeros_like(model.bn.bias),
        ),
    }
    restorer = PeriodicSparseRestorer(
        model=model,
        optimizer=optimizer,
        fishers=fishers,
        interval=1,
        sparse_ratio=0.01,
        restore_strength=0.5,
        budget_mode=budget_mode,
        drift_thresholds=(0.10, 0.20, 0.35),
        drift_ratios=(0.0, 0.02, 0.05, 0.10),
        reliability_threshold=reliability_threshold,
        disable_reliability_gate=disable_reliability_gate,
        event_trigger=event_trigger,
        event_k=event_k,
        event_rho_max=event_rho_max,
        event_cooldown=event_cooldown,
        online_fisher_beta=online_fisher_beta,
    )
    with torch.no_grad():
        model.bn.weight.add_(drift)
    return model, restorer


class DriftTieredBudgetTests(unittest.TestCase):
    def test_drift_boundaries_map_to_expected_ratios(self):
        _model, restorer = _build_restorer(0.0)
        cases = [
            (0.00, 0.00),
            (0.099, 0.00),
            (0.10, 0.02),
            (0.199, 0.02),
            (0.20, 0.05),
            (0.349, 0.05),
            (0.35, 0.10),
            (1.00, 0.10),
        ]
        for drift, expected in cases:
            with self.subTest(drift=drift):
                self.assertEqual(restorer._ratio_for_drift(drift), expected)

    def test_low_drift_skips_restore(self):
        _model, restorer = _build_restorer(0.05)
        self.assertEqual(restorer.step(), 0)
        stats = restorer.stats
        self.assertEqual(stats["scheduled_restore_events"], 1)
        self.assertEqual(stats["restore_events"], 0)
        self.assertEqual(stats["skipped_low_drift_events"], 1)
        self.assertEqual(stats["selected_elements"], 0)

    def test_medium_drift_restores_five_percent(self):
        _model, restorer = _build_restorer(0.25)
        before = restorer.drift_stats()["total"]
        recovered = restorer.step()
        after = restorer.drift_stats()["total"]
        self.assertEqual(recovered, math.ceil(0.05 * 200))
        self.assertEqual(restorer.stats["last_selected_ratio"], 0.05)
        self.assertLess(after, before)

    def test_high_drift_restores_ten_percent(self):
        _model, restorer = _build_restorer(0.40)
        recovered = restorer.step()
        self.assertEqual(recovered, math.ceil(0.10 * 200))
        self.assertEqual(restorer.stats["maximum_selected_elements"], 20)
        self.assertEqual(restorer.stats["budget_event_counts"]["0.1"], 1)

    def test_dynamic_selection_uses_fisher_times_current_drift(self):
        model, restorer = _build_restorer(0.0)
        with torch.no_grad():
            model.bn.weight[0] += 0.40
            model.bn.weight[-1] += 0.40
            model.bn.bias[-19:] += 1.30

        recovered = restorer.step()
        selected_weight = restorer.selected["bn.weight"].tolist()
        self.assertEqual(recovered, math.ceil(0.10 * 200))
        self.assertIn(model.bn.weight.numel() - 1, selected_weight)
        self.assertGreater(
            restorer.stats["last_selected_drift_energy_ratio"], 0.0
        )

    def test_fixed_mode_preserves_legacy_budget(self):
        _model, restorer = _build_restorer(0.0, budget_mode="fixed")
        recovered = restorer.step()
        self.assertEqual(recovered, math.ceil(0.01 * 200))
        self.assertEqual(restorer.stats["budget_mode"], "fixed")


class ReliabilityGatingTests(unittest.TestCase):
    def test_reliability_below_threshold_skips_restore(self):
        _model, restorer = _build_restorer(0.40, reliability_threshold=0.60)
        recovered = restorer.step(reliability=0.30)
        self.assertEqual(recovered, 0)
        self.assertEqual(restorer.stats["skipped_low_reliability_events"], 1)

    def test_reliability_above_threshold_scales_budget(self):
        _model, restorer = _build_restorer(0.40, reliability_threshold=0.50)
        recovered = restorer.step(reliability=0.80)
        # Base ratio for high drift is 0.10, scaled by 0.80 -> 0.08 * 200 = 16
        self.assertEqual(recovered, math.ceil(round(0.08 * 200, 6)))
        self.assertEqual(restorer.stats["restore_events"], 1)
        self.assertAlmostEqual(restorer.stats["last_selected_ratio"], 0.08)

    def test_disable_reliability_gate_disables_skip_and_scaling(self):
        _model, restorer = _build_restorer(
            0.40, reliability_threshold=0.60, disable_reliability_gate=True
        )
        # Reliability 0.20 is below 0.60, but gate is disabled -> does not skip, ratio remains 0.10
        recovered = restorer.step(reliability=0.20)
        self.assertEqual(recovered, math.ceil(0.10 * 200))
        self.assertEqual(restorer.stats["skipped_low_reliability_events"], 0)
        self.assertEqual(restorer.stats["last_selected_ratio"], 0.10)
        self.assertTrue(restorer.stats["reliability_gate_disabled"])

    def test_event_rho_max_updates_maximum_selected_elements(self):
        _model, restorer = _build_restorer(
            0.0, event_trigger=True, event_rho_max=0.20
        )
        self.assertEqual(restorer.stats["maximum_selected_elements"], 40)


class TripleConditionEventTests(unittest.TestCase):
    def test_clean_domain_does_not_trigger_event(self):
        # Clean domain: low entropy (1.0), high reliability (0.9), low drift (0.02), low disagreement (0.05)
        _model, restorer = _build_restorer(
            drift=0.02, event_trigger=True, event_k=3
        )
        for _ in range(5):
            recovered = restorer.step(reliability=0.90, entropy=1.0, anchor_disagreement=0.05)
            self.assertFalse(restorer.last_event_triggered)
            self.assertEqual(recovered, 0)  # low drift skips restore

    def test_severe_drift_anomaly_triggers_sustained_event(self):
        # Corrupted domain with pathological overconfidence:
        # High drift (>0.20), Low reliability (0.25), Abnormally low entropy relative to history
        model, restorer = _build_restorer(
            drift=0.0, event_trigger=True, event_k=3, event_cooldown=20
        )
        # Establish baseline entropy around 3.5 on low-drift clean phase
        for _ in range(10):
            restorer.step(reliability=0.80, entropy=3.5, anchor_disagreement=0.10)

        # Induce high parameter drift (0.35) and sudden anomaly (entropy drops to 0.5, reliability to 0.20)
        with torch.no_grad():
            model.bn.weight.add_(0.35)

        for step_idx in range(1, 4):
            recovered = restorer.step(reliability=0.20, entropy=0.5, anchor_disagreement=0.30)
            if step_idx < 3:
                self.assertFalse(restorer.last_event_triggered)
            else:
                self.assertTrue(restorer.last_event_triggered)
                self.assertGreater(recovered, 0)
                self.assertGreaterEqual(restorer.last_elastic_strength, 0.50)
                self.assertEqual(restorer.cooldown_remaining, 20)

    def test_overconfident_error_captured_by_anchor_disagreement(self):
        # "Confident but wrong" failure mode:
        # High sample score / reliability (0.85), but high Lookahead Anchor Disagreement (0.80 > 0.50)
        model, restorer = _build_restorer(
            drift=0.0, event_trigger=True, event_k=3
        )
        for _ in range(10):
            restorer.step(reliability=0.85, entropy=3.5, anchor_disagreement=0.05)

        with torch.no_grad():
            model.bn.weight.add_(0.35)

        # 3 consecutive batches with high anchor disagreement and abnormal entropy drop
        for step_idx in range(1, 4):
            recovered = restorer.step(reliability=0.85, entropy=0.5, anchor_disagreement=0.80)
            if step_idx < 3:
                self.assertFalse(restorer.last_event_triggered)
            else:
                self.assertTrue(restorer.last_event_triggered)
                self.assertGreater(recovered, 0)

    def test_tiered_momentum_handling(self):
        # 1. Light restore (2% ratio): momentum is decayed, not zeroed
        model, restorer = _build_restorer(drift=0.15)  # 0.15 drift -> 2% ratio
        restorer.optimizer.state[model.bn.weight] = {
            "momentum_buffer": torch.ones_like(model.bn.weight)
        }
        restorer.step(reliability=1.0)
        self.assertGreater(restorer.stats["momentum_decay_count"], 0)
        self.assertEqual(restorer.stats["momentum_zero_count"], 0)

        # 2. Heavy restore (5% ratio): momentum is zeroed
        model2, restorer2 = _build_restorer(drift=0.25)  # 0.25 drift -> 5% ratio
        restorer2.optimizer.state[model2.bn.weight] = {
            "momentum_buffer": torch.ones_like(model2.bn.weight)
        }
        restorer2.step(reliability=1.0)
        self.assertGreater(restorer2.stats["momentum_zero_count"], 0)

    def test_adam_state_clearing_on_heavy_restore(self):
        model, restorer = _build_restorer(drift=0.25)
        restorer.optimizer.state[model.bn.weight] = {
            "exp_avg": torch.ones_like(model.bn.weight),
            "exp_avg_sq": torch.ones_like(model.bn.weight),
        }
        restorer.step(reliability=1.0)
        selected_idx = restorer.selected["bn.weight"].cpu()
        self.assertGreater(selected_idx.numel(), 0)
        exp_avg = restorer.optimizer.state[model.bn.weight]["exp_avg"].view(-1)
        exp_avg_sq = restorer.optimizer.state[model.bn.weight]["exp_avg_sq"].view(-1)
        self.assertTrue((exp_avg[selected_idx] == 0.0).all())
        self.assertTrue((exp_avg_sq[selected_idx] == 0.0).all())


class HybridOnlineFisherTests(unittest.TestCase):
    def test_online_fisher_blends_gradient_sensitivity_on_healthy_batches(self):
        model, restorer = _build_restorer(
            drift=0.30, online_fisher_beta=0.50
        )
        model.bn.weight.grad = torch.zeros_like(model.bn.weight)
        model.bn.weight.grad[0] = 10.0  # High gradient on index 0
        restorer.update_online_gradients(healthy=True)
        effective = restorer._get_hybrid_fishers()
        self.assertGreater(effective[0], effective[1])

    def test_online_fisher_frozen_during_unhealthy_batches(self):
        model, restorer = _build_restorer(
            drift=0.30, online_fisher_beta=0.50
        )
        model.bn.weight.grad = torch.ones_like(model.bn.weight)
        restorer.update_online_gradients(healthy=False)
        effective = restorer._get_hybrid_fishers()
        # Accumulator remained zero
        self.assertEqual(restorer._online_fisher_accumulators["bn.weight"].sum().item(), 0.0)


class MultiNormAndLayerAwareTests(unittest.TestCase):
    def test_layernorm_collection_and_restoration(self):
        model = _LNModel(normalized_shape=64)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        fishers = {
            "ln.weight": torch.ones_like(model.ln.weight),
            "ln.bias": torch.zeros_like(model.ln.bias),
        }
        restorer = PeriodicSparseRestorer(
            model=model,
            optimizer=optimizer,
            fishers=fishers,
            interval=1,
            sparse_ratio=0.05,
            budget_mode="fixed",
        )
        self.assertEqual(restorer.total_trainable_elements, 128)
        with torch.no_grad():
            model.ln.weight.add_(0.5)
        recovered = restorer.step()
        self.assertGreater(recovered, 0)

    def test_layer_aware_score_weighting(self):
        model = _BNModel(channels=10)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        fishers = {
            "bn.weight": torch.ones_like(model.bn.weight),
            "bn.bias": torch.ones_like(model.bn.bias),
        }
        restorer = PeriodicSparseRestorer(
            model=model,
            optimizer=optimizer,
            fishers=fishers,
            interval=1,
            budget_mode="drift_tiered",
            layer_aware=True,
            layer_weights={"bn.weight": 2.0, "bn.bias": 0.5},
        )
        with torch.no_grad():
            model.bn.weight.add_(0.4)
            model.bn.bias.add_(0.4)
        recovered = restorer.step()
        self.assertGreater(recovered, 0)


if __name__ == "__main__":
    unittest.main()
