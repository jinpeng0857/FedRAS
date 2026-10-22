import unittest
from types import SimpleNamespace

from supplement_experiments.common import official_protocol_flags
from supplement_experiments.exp4_component_ablation import VARIANTS, ablation_flags
from supplement_experiments.exp_rga_reliability import rga_flags


def _flag_values(flags):
    values = {}
    boolean_flags = {
        "--dlrs_normalize_by_weight",
        "--dlrs_view_consistency",
        "--fslr_continuous_risk",
        "--fslr_event_trigger",
        "--fslr_hierarchical_budget",
        "--disable_fslr_reliability_gate",
    }
    value_counts = {
        "--restore_drift_thresholds": 3,
        "--restore_drift_ratios": 4,
    }
    i = 0
    while i < len(flags):
        flag = flags[i]
        if flag in boolean_flags:
            values[flag] = True
            i += 1
        elif flag in value_counts:
            count = value_counts[flag]
            values[flag] = flags[i + 1 : i + 1 + count]
            i += 1 + count
        else:
            values[flag] = flags[i + 1]
            i += 2
    return values


class OfficialSupplementProtocolTests(unittest.TestCase):
    def test_protocol_flags_match_current_main_feddr_protocol(self):
        flags = official_protocol_flags(
            SimpleNamespace(num_workers=2, seed=0, max_batches=-1)
        )
        values = _flag_values(flags)

        self.assertEqual(values["--adapt_lr"], "0.001")
        self.assertEqual(values["--batch_size"], "128")
        self.assertEqual(values["--feddr_bn_scope"], "excluding_layer4")
        self.assertEqual(values["--restore_strength"], "0.7")
        self.assertEqual(values["--dlrs_region_scale_min"], "0.6")
        self.assertEqual(values["--dlrs_region_scale_max"], "1.2")
        self.assertEqual(values["--dlrs_hybrid_threshold"], "0.2")
        self.assertEqual(values["--fslr_grad_clip_norm"], "15.0")
        self.assertTrue(values["--dlrs_normalize_by_weight"])
        self.assertTrue(values["--dlrs_view_consistency"])
        self.assertTrue(values["--fslr_continuous_risk"])
        self.assertTrue(values["--fslr_event_trigger"])
        self.assertTrue(values["--fslr_hierarchical_budget"])

    def test_component_ablation_uses_public_feddr_entry_and_official_controls(self):
        self.assertEqual(VARIANTS["full"][0], "feddr")
        flags = ablation_flags(
            SimpleNamespace(num_workers=2, seed=0, max_batches=-1)
        )
        self.assertNotIn("--strict_mechanism_ablation", flags)
        self.assertNotIn("--disable_fslr_event_trigger", flags)
        self.assertNotIn("--disable_fslr_continuous_risk", flags)
        self.assertNotIn("--disable_fslr_hierarchical_budget", flags)
        values = _flag_values(flags)
        self.assertEqual(values["--adapt_lr"], "0.001")
        self.assertEqual(values["--batch_size"], "128")
        self.assertEqual(values["--fslr_grad_clip_norm"], "15.0")

    def test_rga_validity_rows_keep_official_controls(self):
        flags = rga_flags(
            SimpleNamespace(num_workers=2, seed=0, max_batches=-1),
            weight_mode="sample_only",
            sample_mode="entropy",
        )
        self.assertNotIn("--strict_mechanism_ablation", flags)
        values = _flag_values(flags)
        self.assertEqual(values["--adapt_lr"], "0.001")
        self.assertEqual(values["--batch_size"], "128")
        self.assertEqual(values["--dlrs_region_scale_min"], "0.6")
        self.assertEqual(values["--dlrs_region_scale_max"], "1.2")
        self.assertEqual(values["--dlrs_hybrid_threshold"], "0.2")
        self.assertEqual(values["--dlrs_weight_mode"], "sample_only")
        self.assertEqual(values["--dlrs_sample_reliability_mode"], "entropy")
        self.assertTrue(values["--dlrs_view_consistency"])
        self.assertTrue(values["--fslr_continuous_risk"])


if __name__ == "__main__":
    unittest.main()
