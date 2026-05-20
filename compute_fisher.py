"""
Fisher Information Pre-computation Entry Point.

Replaces ACG-C compute_fisher.py Hydra entry.
Computes Fisher values for BN layers on a small set of ID samples.

Usage:
  python compute_fisher.py --dataset cifar100 --backbone eatac_resnet18 \
      --model_path results/.../model_round100.pt \
      --fisher_path results/.../fisher.pt --num_fisher_samples 2000
"""

import logging
import os
import sys

import torch
from torch.utils.data import DataLoader, Subset

from src.algorithm.EATA_C.fisher import compute_fisher_information
from src.model.eatac_resnet import eatac_resnet18, eatac_resnet34
from src.model.eatac_imagenet_resnet import eatac_resnet50, eatac_resnet101
from dataset.dataset import load_clean_test
from utils.config import parser
from utils.logger import logger
from utils.utils import set_seed, load_checkpoint_weights


def run():
    configs = parser.parse_args()
    configs.step = "fisher"
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
    logger.info(f"Loading model: {configs.backbone} (num_class={configs.num_class})")
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

    model.to(device)
    model.eval()

    # ---- Load ID samples ----
    logger.info("Loading clean test data for Fisher computation...")
    clean_dataset, num_class = load_clean_test(configs)
    num_samples = min(configs.num_fisher_samples, len(clean_dataset))
    indices = list(range(num_samples))
    subset = Subset(clean_dataset, indices)
    loader = DataLoader(subset, batch_size=configs.test_batch_size, shuffle=False)

    id_samples = []
    for x, _ in loader:
        id_samples.append(x)
    id_samples = torch.cat(id_samples, dim=0)[:num_samples]

    # ---- Compute Fisher ----
    logger.info(f"Computing Fisher on {id_samples.size(0)} samples...")
    fisher = compute_fisher_information(
        model, id_samples, batch_size=configs.test_batch_size, device=device
    )

    # ---- Save ----
    fisher_path = configs.fisher_path
    if fisher_path is None:
        fisher_path = os.path.join(
            configs.checkpoint_path,
            f"fisher_{configs.dataset}_{num_samples}.pt",
        )
    elif not fisher_path.endswith(".pt"):
        # Directory path given → append default filename
        fisher_path = os.path.join(
            fisher_path,
            f"fisher_{configs.dataset}_{num_samples}.pt",
        )
    os.makedirs(os.path.dirname(fisher_path) or ".", exist_ok=True)
    torch.save(fisher, fisher_path)
    logger.info(f"Fisher saved to: {fisher_path}")
    logger.info(f"  Parameter count: {len(fisher)}")
    logger.info(f"  Example keys: {list(fisher.keys())[:5]}...")


if __name__ == "__main__":
    run()
