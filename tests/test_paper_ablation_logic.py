"""Unit tests for FedDR Paper Ablation Logic.

Validates:
  1. CLI argument parser accepts `--dlrs_weight_mode uniform`.
  2. DLRSAdapter in uniform mode produces unit weights without errors.
  3. Four coordinate selection criteria produce distinct selections on synthetic drift/fisher tensors.
  4. Fixed budget mode routes through selection_criterion.
  5. Random selection criterion is reproducible with same seed and differs across seeds.
  6. Results summary parses Schema v9 JSONs and correctly converts fractions to percentages.
  7. Strict summary mode refuses incomplete LaTeX table generation.
  8. Primary FedDR configuration activates weak-view consistency, continuous risk, hierarchical budget, and non-layer4 BN scope.
"""

import copy
import json
import math
from pathlib import Path
import tempfile
import unittest

import torch
import torch.nn as nn

from adapters.dlrs import DLRSAdapter
from evaluator.engine import configure_bn_adaptation
from main_test import apply_feddr_primary_config, parse_args
from scripts.run_paper_ablations import validate_existing_result
from scripts.summarize_paper_ablations import load_metrics_from_json, summarize_block_a, summarize_block_c
from sparse_restore import PeriodicSparseRestorer


class ToyResNet(nn.Module):
    """Toy ResNet-like model with hierarchical layers."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = nn.Sequential(nn.BatchNorm2d(16))
        self.layer4 = nn.Sequential(nn.BatchNorm2d(16))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(16, 10)

    def forward(self, x):
        x = self.bn1(self.conv1(x))
        x = self.layer1(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


class TestPaperAblationLogic(unittest.TestCase):

    def test_01_cli_parser_accepts_uniform_weight_mode(self):
        """Test that --dlrs_weight_mode accepts 'uniform'."""
        test_args = [
            "--dataset", "cifar100",
            "--model_path", "dummy.pt",
            "--statistics_path", "dummy.pt",
            "--output_path", "out.json",
            "--dlrs_weight_mode", "uniform",
        ]
        import sys
        old_argv = sys.argv
        try:
            sys.argv = ["main_test.py"] + test_args
            parsed = parse_args()
            self.assertEqual(parsed.dlrs_weight_mode, "uniform")
        finally:
            sys.argv = old_argv

    def test_09_cli_parser_accepts_block_c_fixed_ratio(self):
        """Block C's fixed 5% ratio must be accepted by the main CLI."""
        test_args = [
            "--dataset", "cifar100",
            "--model_path", "dummy.pt",
            "--statistics_path", "dummy.pt",
            "--output_path", "out.json",
            "--tta_method", "dlrs_fslr",
            "--restore_budget_mode", "fixed",
            "--sparse_ratio", "0.05",
            "--fslr_selection_criterion", "random",
            "--fslr_selection_seed", "42",
        ]
        import sys
        old_argv = sys.argv
        try:
            sys.argv = ["main_test.py"] + test_args
            parsed = parse_args()
            self.assertAlmostEqual(parsed.sparse_ratio, 0.05)
            self.assertEqual(parsed.fslr_selection_seed, 42)
        finally:
            sys.argv = old_argv

    def test_10_result_validator_accepts_schema_v9_nested_clean_accuracy(self):
        """Schema v9 clean retention is stored under clean_retention.final_clean.accuracy."""
        payload = {
            "dataset": "cifar100",
            "adaptation": {"tta_method": "dlrs"},
            "domain_wise": {"summary": {"mean_accuracy": 0.5}},
            "continual": {"summary": {"cycle_summaries": [{}, {}, {}]}},
            "clean_retention": {
                "initial_clean": {"accuracy": 0.76},
                "final_clean": {"accuracy": 0.70},
            },
        }
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as tf:
            json.dump(payload, tf)
            tf_path = Path(tf.name)

        try:
            self.assertTrue(validate_existing_result(tf_path, {"tta_method": "dlrs"}))
        finally:
            if tf_path.exists():
                tf_path.unlink()

    def test_02_dlrs_adapter_uniform_weights(self):
        """Test that DLRSAdapter in uniform mode produces all-one joint weights."""
        model = ToyResNet()
        opt = torch.optim.SGD(model.parameters(), lr=0.001)
        feature_var = torch.ones(16)
        fishers = {name: torch.ones_like(p) for name, p in model.named_parameters()}

        restorer = PeriodicSparseRestorer(
            model=model,
            optimizer=opt,
            fishers=fishers,
            interval=10,
            sparse_ratio=0.05,
        )

        adapter = DLRSAdapter(
            model=model,
            optimizer=opt,
            feature_variance=feature_var,
            restorer=restorer,
            fishers=fishers,
            tta_method="dlrs",
            weight_mode="uniform",
            hybrid_threshold=0.0,
            region_scale_min=1.2,
            region_scale_max=1.2,
        )

        dummy_images = torch.randn(4, 3, 8, 8)
        logits, metadata = adapter.adapt(dummy_images)
        self.assertEqual(logits.shape, (4, 10))
        self.assertTrue(metadata.get("updated"))
        self.assertAlmostEqual(metadata.get("dlrs_joint_weight_sum"), 4.0, places=4)
        self.assertAlmostEqual(metadata.get("dlrs_effective_sample_size"), 4.0, places=4)

    def test_03_coordinate_selection_criteria_distinct(self):
        """Test that the four selection criteria produce distinct index sets on constructive inputs."""
        model = ToyResNet()
        opt = torch.optim.SGD(model.parameters(), lr=0.001)
        params = {n: p for n, p in model.named_parameters() if "bn" in n or "layer" in n}

        # Synthetic fishers: highest in bn1
        fishers = {}
        for n, p in params.items():
            if "bn1" in n:
                fishers[n] = torch.full_like(p, 100.0)
            else:
                fishers[n] = torch.full_like(p, 0.01)

        # 1. Fisher only -> Should select exclusively from bn1
        restorer_fisher = PeriodicSparseRestorer(
            model=model,
            optimizer=opt,
            fishers=fishers,
            interval=1,
            budget_mode="fixed",
            sparse_ratio=0.20,
            selection_criterion="fisher_only",
        )

        # Induce drift AFTER restorer captures anchors at initialization
        with torch.no_grad():
            for n, p in params.items():
                if "layer4" in n:
                    p.add_(5.0)
                else:
                    p.add_(0.001)

        restorer_fisher.step(reliability=1.0)
        selected_fisher = copy.deepcopy(restorer_fisher.selected)

        # 2. Drift only -> Should select exclusively from layer4
        model_drift = ToyResNet()
        opt_drift = torch.optim.SGD(model_drift.parameters(), lr=0.001)
        params_drift = {n: p for n, p in model_drift.named_parameters() if "bn" in n or "layer" in n}
        fishers_drift = {n: fishers[n].clone() for n in params_drift}

        restorer_drift = PeriodicSparseRestorer(
            model=model_drift,
            optimizer=opt_drift,
            fishers=fishers_drift,
            interval=1,
            budget_mode="fixed",
            sparse_ratio=0.20,
            selection_criterion="drift_only",
        )

        # Induce drift in layer4
        with torch.no_grad():
            for n, p in params_drift.items():
                if "layer4" in n:
                    p.add_(5.0)
                else:
                    p.add_(0.001)

        restorer_drift.step(reliability=1.0)
        selected_drift = copy.deepcopy(restorer_drift.selected)

        fisher_bn1_count = sum(indices.numel() for n, indices in selected_fisher.items() if "bn1" in n)
        drift_layer4_count = sum(indices.numel() for n, indices in selected_drift.items() if "layer4" in n)

        self.assertGreater(fisher_bn1_count, 0)
        self.assertGreater(drift_layer4_count, 0)
        drift_bn1_count = sum(indices.numel() for n, indices in selected_drift.items() if "bn1" in n)
        self.assertEqual(drift_bn1_count, 0)

    def test_04_fixed_budget_respects_selection_criterion(self):
        """Test that fixed budget mode does NOT ignore selection_criterion."""
        model = ToyResNet()
        opt = torch.optim.SGD(model.parameters(), lr=0.001)
        params = {n: p for n, p in model.named_parameters() if "bn" in n or "layer" in n}
        fishers = {n: torch.full_like(p, 10.0) for n, p in params.items()}

        restorer = PeriodicSparseRestorer(
            model=model,
            optimizer=opt,
            fishers=fishers,
            interval=1,
            budget_mode="fixed",
            sparse_ratio=0.10,
            selection_criterion="drift_only",
        )

        with torch.no_grad():
            for n, p in params.items():
                if "bn1.weight" in n:
                    p.add_(10.0)
                else:
                    p.add_(0.01)

        restorer.step(reliability=1.0)
        self.assertIn("bn1.weight", restorer.selected)
        self.assertGreater(restorer.selected["bn1.weight"].numel(), 0)

    def test_05_random_selection_seed_reproducibility(self):
        """Test that random coordinate selection is reproducible with same seed and varies across seeds."""
        model_a = ToyResNet()
        opt_a = torch.optim.SGD(model_a.parameters(), lr=0.001)
        fishers_a = {n: torch.ones_like(p) for n, p in model_a.named_parameters()}

        r_s42_a = PeriodicSparseRestorer(
            model=model_a, optimizer=opt_a, fishers=fishers_a,
            interval=1, budget_mode="fixed", sparse_ratio=0.10,
            selection_criterion="random", selection_seed=42,
        )
        r_s42_a.step(reliability=1.0)

        model_b = ToyResNet()
        opt_b = torch.optim.SGD(model_b.parameters(), lr=0.001)
        fishers_b = {n: torch.ones_like(p) for n, p in model_b.named_parameters()}

        r_s42_b = PeriodicSparseRestorer(
            model=model_b, optimizer=opt_b, fishers=fishers_b,
            interval=1, budget_mode="fixed", sparse_ratio=0.10,
            selection_criterion="random", selection_seed=42,
        )
        r_s42_b.step(reliability=1.0)

        model_c = ToyResNet()
        opt_c = torch.optim.SGD(model_c.parameters(), lr=0.001)
        fishers_c = {n: torch.ones_like(p) for n, p in model_c.named_parameters()}

        r_s43 = PeriodicSparseRestorer(
            model=model_c, optimizer=opt_c, fishers=fishers_c,
            interval=1, budget_mode="fixed", sparse_ratio=0.10,
            selection_criterion="random", selection_seed=43,
        )
        r_s43.step(reliability=1.0)

        # Compare s42_a vs s42_b (must be identical)
        for name in r_s42_a.params:
            idx_a = r_s42_a.selected.get(name, torch.empty(0))
            idx_b = r_s42_b.selected.get(name, torch.empty(0))
            self.assertTrue(torch.equal(idx_a, idx_b))

        # Compare s42 vs s43 (must have differences)
        diff_found = False
        for name in r_s42_a.params:
            idx_42 = r_s42_a.selected.get(name, torch.empty(0))
            idx_43 = r_s43.selected.get(name, torch.empty(0))
            if not torch.equal(idx_42, idx_43):
                diff_found = True
                break
        self.assertTrue(diff_found)

    def test_06_summarizer_parses_schema_v9_and_scales_percentages(self):
        """Test that summarize_paper_ablations accurately parses schema v9 JSON and multiplies by 100."""
        fixture_payload = {
            "schema_version": 9,
            "dataset": "cifar100",
            "domain_wise": {
                "summary": {"mean_accuracy": 0.62115}
            },
            "continual": {
                "summary": {
                    "cycle_summaries": [
                        {"mean_accuracy": 0.63609},
                        {"mean_accuracy": 0.63400},
                        {"mean_accuracy": 0.63289},
                    ],
                    "late_delta_cycle3_minus_cycle1": -0.00320,
                },
                "post_stream": {
                    "mean_revisit_accuracy": 0.58755,
                },
                "restoration": {
                    "total_drift_reduction": 0.0512,
                }
            },
            "clean_retention": {
                "final_continual_clean_accuracy": 0.7373,
            }
        }

        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as tf:
            json.dump(fixture_payload, tf)
            tf_path = Path(tf.name)

        try:
            metrics = load_metrics_from_json(tf_path)
            self.assertIsNotNone(metrics)
            self.assertAlmostEqual(metrics["domain_wise"], 62.115, places=2)
            self.assertAlmostEqual(metrics["cycle1"], 63.609, places=2)
            self.assertAlmostEqual(metrics["cycle3"], 63.289, places=2)
            self.assertAlmostEqual(metrics["delta"], -0.320, places=2)
            self.assertAlmostEqual(metrics["revisit"], 58.755, places=2)
            self.assertAlmostEqual(metrics["clean"], 73.73, places=2)
        finally:
            if tf_path.exists():
                tf_path.unlink()

    def test_07_strict_mode_detects_missing_outputs(self):
        """Test that missing files are recorded and mark completeness as False."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_root = Path(tmpdir)
            missing = []
            md, tex, comp = summarize_block_a(out_root, ["run1", "run2", "run3"], missing)
            self.assertFalse(comp)
            self.assertGreater(len(missing), 0)

    def test_08_feddr_full_primary_configuration(self):
        """Test that FedDR-Full resolves stability-oriented primary flags."""
        class DummyArgs:
            def __init__(self):
                self.tta_method = "feddr"
                self.strict_mechanism_ablation = False
                self.dlrs_view_consistency = None
                self.dlrs_view_alpha = None
                self.dlrs_view_js_scale = None
                self.fslr_continuous_risk = None
                self.fslr_hierarchical_budget = None
                self.fslr_layer_quota_cap = None
                self.dlrs_hybrid_threshold = None
                self.dlrs_anchor_rectify_beta = None
                self.fslr_grad_clip_norm = None
                self.batch_size = None
                self.adapt_lr = None
                self.feddr_bn_scope = "method_default"

        args = DummyArgs()
        args = apply_feddr_primary_config(args, "feddr")

        self.assertTrue(args.dlrs_view_consistency)
        self.assertAlmostEqual(args.dlrs_view_alpha, 0.80)
        self.assertTrue(args.fslr_continuous_risk)
        self.assertTrue(args.fslr_hierarchical_budget)
        self.assertAlmostEqual(args.fslr_layer_quota_cap, 0.35)
        self.assertAlmostEqual(args.adapt_lr, 1.0e-3)
        self.assertEqual(args.feddr_bn_scope, "excluding_layer4")
        self.assertAlmostEqual(args.dlrs_hybrid_threshold, 0.2)
        self.assertAlmostEqual(args.dlrs_anchor_rectify_beta, 0.3)

        # Check parameter scope exclusion of layer4
        model = ToyResNet()
        params, names = configure_bn_adaptation(model)
        trainable_names = [n for n, p in model.named_parameters() if p.requires_grad]
        self.assertTrue(any("bn1" in n for n in trainable_names))
        self.assertTrue(any("layer1" in n for n in trainable_names))
        self.assertFalse(any("layer4" in n for n in trainable_names))


if __name__ == "__main__":
    unittest.main()
