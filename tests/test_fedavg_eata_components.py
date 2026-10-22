"""Targeted checks for federated initialization and TTA comparisons."""

import math
import unittest
from types import SimpleNamespace

import torch
from torch import nn

from train.fedavg.server import FedAvgServer
from train.fedprox.client import FedProxClient
from train.fedprox.server import FedProxServer
from main_test import (
    DUAL_RELIABILITY_METHODS,
    DualReliabilityAdapter,
    configure_bn_adaptation,
    configure_official_bn_adaptation,
    resolve_tta_method,
    select_anchor_state,
)


class _TwoClassModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            self.fc.weight.copy_(torch.eye(2))

    def forward(self, inputs):
        return self.fc(inputs)


class _NoopRestorer:
    def step(self):
        return 0


class _LayeredBNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(2)
        self.layer4 = nn.Sequential(nn.BatchNorm2d(2))

    def forward(self, inputs):
        return self.layer4(self.bn1(inputs))


class FedAvgCheckpointTests(unittest.TestCase):
    def test_checkpoint_uses_explicit_adaptation_anchor(self):
        configs = SimpleNamespace(
            join_ratio=1.0,
            global_rounds=1,
            checkpoint_path="unused",
            save_freq=1,
            eval_freq=1,
            debug=False,
            num_class=2,
            backbone="tiny",
            dataset="synthetic",
            alpha=0.3,
        )
        server = FedAvgServer(torch.device("cpu"), nn.Linear(2, 2), configs)
        checkpoint = server._build_checkpoint(0)
        self.assertIn("adaptation_anchor", checkpoint)
        self.assertNotIn("lookahead_anchor", checkpoint)
        self.assertEqual(checkpoint["anchor_type"], "fedavg_global_model")

    def test_weighted_average_and_bn_counter(self):
        server = FedAvgServer.__new__(FedAvgServer)
        first = {
            "weight": torch.tensor([2.0]),
            "bn.num_batches_tracked": torch.tensor(3, dtype=torch.long),
        }
        second = {
            "weight": torch.tensor([4.0]),
            "bn.num_batches_tracked": torch.tensor(7, dtype=torch.long),
        }
        averaged = server.aggregate(
            [first, second],
            local_deltas_list=[],
            client_ids=[0, 1],
            client_weights=[1, 3],
        )
        self.assertTrue(torch.allclose(averaged["weight"], torch.tensor([3.5])))
        self.assertEqual(int(averaged["bn.num_batches_tracked"]), 7)

    def test_anchor_selector_keeps_fedavg_distinct(self):
        state = {"fc.weight": torch.ones(2, 2)}
        checkpoint = {"adaptation_anchor": state}
        self.assertIs(
            select_anchor_state(checkpoint, "adaptation_anchor"),
            state,
        )
        with self.assertRaises(KeyError):
            select_anchor_state(checkpoint, "lookahead_anchor")


class FedProxComponentTests(unittest.TestCase):
    @staticmethod
    def _configs(mu=0.01):
        return SimpleNamespace(
            local_epochs=1,
            learning_rate=0.1,
            momentum=0.0,
            weight_decay=0.0,
            batch_size=2,
            debug=False,
            num_workers=0,
            ce_weight=1.0,
            prox_weight=0.0,
            fedprox_mu=mu,
            label_smoothing=0.0,
            join_ratio=1.0,
            global_rounds=1,
            checkpoint_path="unused",
            save_freq=1,
            eval_freq=1,
            num_class=2,
            backbone="tiny",
            dataset="synthetic",
            alpha=0.3,
        )

    def test_client_uses_explicit_fedprox_mu(self):
        client = FedProxClient(
            0, torch.device("cpu"), nn.Linear(2, 2), self._configs(0.25)
        )
        self.assertEqual(client.ce_weight, 1.0)
        self.assertEqual(client.prox_weight, 0.25)

    def test_nonpositive_mu_is_rejected(self):
        with self.assertRaises(ValueError):
            FedProxClient(
                0, torch.device("cpu"), nn.Linear(2, 2), self._configs(0.0)
            )

    def test_checkpoint_uses_fedprox_adaptation_anchor(self):
        server = FedProxServer(
            torch.device("cpu"), nn.Linear(2, 2), self._configs(0.01)
        )
        checkpoint = server._build_checkpoint(0)
        self.assertEqual(checkpoint["method"], "fedprox")
        self.assertIn("adaptation_anchor", checkpoint)
        self.assertNotIn("lookahead_anchor", checkpoint)
        self.assertEqual(checkpoint["config_info"]["fedprox_mu"], 0.01)


class EATASelectionTests(unittest.TestCase):
    def test_official_tent_eata_scope_includes_layer4_bn(self):
        model = _LayeredBNModel()
        _params, names = configure_official_bn_adaptation(model)
        self.assertIn("bn1.weight", names)
        self.assertIn("layer4.0.weight", names)

    def test_main_recap_scope_still_excludes_layer4_bn(self):
        model = _LayeredBNModel()
        _params, names = configure_bn_adaptation(model)
        self.assertIn("bn1.weight", names)
        self.assertNotIn("layer4.0.weight", names)
    def test_method_resolution_requires_explicit_method_and_removes_legacy_paths(self):
        with self.assertRaises(ValueError):
            resolve_tta_method()
        self.assertEqual(resolve_tta_method("feddr"), "dlrs_fslr")
        self.assertEqual(resolve_tta_method("recap"), "recap")
        with self.assertRaises(ValueError):
            resolve_tta_method("recap_fslr")
        with self.assertRaises(ValueError):
            resolve_tta_method("eata_recap_fslr")
        self.assertEqual(resolve_tta_method("tent"), "tent")

    def test_tent_uses_all_samples_and_entropy_objective(self):
        model = _TwoClassModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
        adapter = DualReliabilityAdapter(
            model=model,
            optimizer=optimizer,
            feature_variance=torch.zeros(2),
            restorer=_NoopRestorer(),
            fishers={},
            tta_method="tent",
            eata_fisher_alpha=0.0,
        )
        _logits, stats = adapter.adapt(
            torch.tensor([[5.0, 0.0], [0.0, 5.0]])
        )
        self.assertEqual(stats["entropy_passed"], 2)
        self.assertEqual(stats["nonredundant"], 2)
        self.assertEqual(stats["region_reliable"], 2)
        self.assertEqual(stats["selected"], 2)
        self.assertEqual(stats["recap_loss"], 0.0)
        self.assertTrue(stats["updated"])

    def test_all_comparison_methods_execute_on_the_same_batch_contract(self):
        for method in DUAL_RELIABILITY_METHODS:
            with self.subTest(method=method):
                model = _TwoClassModel()
                optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
                adapter = DualReliabilityAdapter(
                    model=model,
                    optimizer=optimizer,
                    feature_variance=torch.zeros(2),
                    restorer=_NoopRestorer(),
                    fishers={},
                    tta_method=method,
                    eata_e_margin_factor=0.4,
                    eata_fisher_alpha=0.0,
                )
                logits, stats = adapter.adapt(torch.tensor([[5.0, 0.0]]))
                self.assertEqual(tuple(logits.shape), (1, 2))
                self.assertEqual(stats["samples"], 1)
                self.assertEqual(stats["selected"], 1)
                self.assertTrue(stats["updated"])

    def test_pure_eata_does_not_use_recap_objective(self):
        model = _TwoClassModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
        adapter = DualReliabilityAdapter(
            model=model,
            optimizer=optimizer,
            feature_variance=torch.zeros(2),
            restorer=_NoopRestorer(),
            fishers={},
            tta_method="eata",
            eata_e_margin_factor=0.4,
            eata_fisher_alpha=0.0,
        )
        _logits, stats = adapter.adapt(torch.tensor([[5.0, 0.0]]))
        self.assertEqual(stats["selected"], 1)
        self.assertEqual(stats["recap_loss"], 0.0)
        self.assertGreater(stats["loss"], 0.0)
        self.assertTrue(stats["updated"])

    def test_repeated_confident_probability_becomes_redundant(self):
        model = _TwoClassModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
        adapter = DualReliabilityAdapter(
            model=model,
            optimizer=optimizer,
            feature_variance=torch.zeros(2),
            restorer=_NoopRestorer(),
            fishers={},
            tta_method="eata",
            eata_e_margin_factor=0.4,
            eata_d_margin=0.05,
            eata_fisher_alpha=0.0,
        )
        self.assertAlmostEqual(adapter.eata_e_margin, 0.4 * math.log(2))

        inputs = torch.tensor([[5.0, 0.0]])
        _logits, first = adapter.adapt(inputs)
        _logits, second = adapter.adapt(inputs)

        self.assertEqual(first["entropy_passed"], 1)
        self.assertEqual(first["nonredundant"], 1)
        self.assertEqual(first["selected"], 1)
        self.assertTrue(first["updated"])
        self.assertEqual(second["entropy_passed"], 1)
        self.assertEqual(second["nonredundant"], 0)
        self.assertEqual(second["selected"], 0)
        self.assertFalse(second["updated"])


if __name__ == "__main__":
    unittest.main()
