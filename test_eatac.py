"""
EATA-C TTA Evaluation Entry Point.

Replaces ACG-C test.py Hydra entry.
Evaluates a trained FL model under distribution shifts using EATA-C.

Usage:
  # Test all 15 corruptions
  python test_eatac.py --dataset cifar100 --backbone eatac_resnet18 \
      --model_path results/.../model_round100.pt \
      --fisher_path results/.../fisher.pt --adapt_steps 3

  # Test single corruption
  python test_eatac.py --dataset cifar100 --corruption gaussian_noise --severity 5 \
      --model_path .../model_round100.pt --fisher_path .../fisher.pt
"""

import logging
import os
import sys

import torch
from torch.utils.data import DataLoader

from src.algorithm.EATA_C.eata_c_test import EATACTester
from src.model.eatac_resnet import eatac_resnet18, eatac_resnet34
from src.model.eatac_imagenet_resnet import eatac_resnet50, eatac_resnet101
from dataset.dataset import (
    load_clean_test,
    load_corrupted_test,
    STANDARD_CORRUPTIONS,
)
from utils.config import parser
from utils.logger import logger
from utils.utils import set_seed, load_checkpoint_weights


def test_single_corruption(configs, corruption_type, tester):
    """Evaluate on a single corruption type."""
    configs.corruption = corruption_type
    test_dataset = load_corrupted_test(
        configs, corruption_type, configs.severity
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=configs.test_batch_size,
        shuffle=False,
        num_workers=0,
    )

    # Source-only baseline (no adaptation)
    source_result = tester.evaluate_source_only(test_loader)
    source_acc = source_result["accuracy"]

    # Reset model and run EATA-C with adaptation
    tester._reset_model()
    result = tester.evaluate(test_loader)

    eata_acc = result["accuracy"]
    ent_pass = result.get("avg_entropy_pass_rate", 0.0)
    sel_ratio = result.get("avg_selected_ratio", 0.0)
    gap = eata_acc - source_acc

    result["source_acc"] = source_acc

    logger.info(
        f"  {corruption_type:25s}  "
        f"Source={source_acc:.2%}  "
        f"EATA-C={eata_acc:.2%}  "
        f"Gap={gap:+.2%}  "
        f"EntPass={ent_pass:.1%}  "
        f"Selected={sel_ratio:.1%}  "
        f"ECE={result['ece']:.2%}"
    )
    return result


def run():
    configs = parser.parse_args()
    set_seed(configs.seed)
    device = torch.device(configs.device if torch.cuda.is_available() else "cpu")

    # ---- Auto-detect num_class from checkpoint ----
    if configs.model_path and os.path.exists(configs.model_path):
        ckpt = torch.load(configs.model_path, map_location="cpu", weights_only=False)
        sd = ckpt
        if isinstance(ckpt, dict):
            for key in ("model_state_dict", "state_dict", "model"):
                if key in ckpt:
                    sd = ckpt[key]
                    break
        if isinstance(sd, dict):
            for fc_key in ("fc.weight", "module.fc.weight"):
                if fc_key in sd:
                    inferred = sd[fc_key].shape[0]
                    if inferred != configs.num_class:
                        logger.info(
                            f"  Auto-detected num_class={inferred} from checkpoint, "
                            f"was {configs.num_class}"
                        )
                        configs.num_class = inferred
                    break

    # ---- Load model ----
    logger.info(f"Loading model: {configs.backbone}")
    if configs.backbone in ("eatac_resnet18", "resnet18"):
        model = eatac_resnet18(
            num_classes=configs.num_class,
            subnet_drop_rate=configs.subnet_drop_rate,
        )
    elif configs.backbone == "eatac_resnet34":
        model = eatac_resnet34(
            num_classes=configs.num_class,
            subnet_drop_rate=configs.subnet_drop_rate,
        )
    elif configs.backbone == "eatac_resnet50":
        model = eatac_resnet50(
            num_classes=configs.num_class,
            subnet_drop_rate=configs.subnet_drop_rate,
        )
    elif configs.backbone == "eatac_resnet101":
        model = eatac_resnet101(
            num_classes=configs.num_class,
            subnet_drop_rate=configs.subnet_drop_rate,
        )
    else:
        raise ValueError(f"Unknown backbone: {configs.backbone}")

    if configs.model_path and os.path.exists(configs.model_path):
        load_checkpoint_weights(model, configs.model_path, device="cpu")
        logger.info(f"  已加载权重: {configs.model_path}")
    else:
        logger.warning("  No model_path provided, using random init!")

    # ---- Create EATA-C tester ----
    tester = EATACTester(
        model=model,
        configs=configs,
        device=device,
        fisher_path=configs.fisher_path,
    )

    # ---- Test ----
    all_results = []

    if configs.corruption is not None:
        # Single corruption
        result = test_single_corruption(configs, configs.corruption, tester)
        all_results.append(result)
    else:
        # All 15 corruptions
        for corruption in STANDARD_CORRUPTIONS:
            logger.info(f"Testing {corruption}...")
            result = test_single_corruption(configs, corruption, tester)
            all_results.append(result)
            # Reset model for each corruption type
            tester._reset_model()

    # ---- Summary ----
    avg_acc = sum(r["accuracy"] for r in all_results) / len(all_results)
    avg_source = sum(r.get("source_acc", 0) for r in all_results) / len(all_results)
    avg_ece = sum(r["ece"] for r in all_results) / len(all_results)
    avg_gap = avg_acc - avg_source
    total_fw = sum(r["forwards"] for r in all_results)
    total_bw = sum(r["backwards"] for r in all_results)

    logger.info("=" * 60)
    logger.info(f"Source-Only Mean Acc: {avg_source:.2%}")
    logger.info(f"EATA-C    Mean Acc:   {avg_acc:.2%}")
    logger.info(f"TTA Gain:             {avg_gap:+.2%}")
    logger.info(f"Mean ECE:             {avg_ece:.2%}")
    logger.info(f"Total Forwards:       {total_fw}")
    logger.info(f"Total Backwards:      {total_bw}")
    logger.info("=" * 60)


if __name__ == "__main__":
    run()
