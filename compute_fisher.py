"""
Fisher Information Pre-computation Entry Point.

Replaces ACG-C compute_fisher.py Hydra entry.
Computes Fisher values for BN layers on a small set of ID samples.

Default data source: train set (calibration split). Use --fisher_data_split test
to revert to test-set behavior (with a warning).

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
import torchvision

from src.algorithm.EATA_C.fisher import compute_fisher_information
from src.model.eatac_resnet import eatac_resnet18, eatac_resnet34
from src.model.eatac_imagenet_resnet import eatac_resnet50, eatac_resnet101
from dataset.dataset import (
    load_clean_test, load_cifar,
    DATASET_CONFIGS, DATASET_PATHS, get_transform,
)
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

    # ---- Load data for Fisher computation ----
    split = configs.fisher_data_split
    num_samples = configs.num_fisher_samples
    logger.info(f"Fisher split = {split}")
    logger.info(f"Fisher num samples = {num_samples}")

    if split == "train":
        # Load train set and take first N samples as calibration split
        name = configs.dataset
        cfg = DATASET_CONFIGS[name]
        data_path = configs.dataset_path or DATASET_PATHS[name]
        transform = get_transform(name, train=True)
        if name == "cifar10":
            dataset = torchvision.datasets.CIFAR10(
                root=data_path, train=True, transform=transform, download=True,
            )
        elif name == "cifar100":
            dataset = torchvision.datasets.CIFAR100(
                root=data_path, train=True, transform=transform, download=True,
            )
        else:
            raise ValueError(f"Train split Fisher not supported for {name}")
        total_avail = len(dataset)
        n_use = min(num_samples, total_avail)
        indices = list(range(n_use))
        subset = Subset(dataset, indices)
        loader = DataLoader(subset, batch_size=configs.test_batch_size, shuffle=False)
        id_samples = []
        for x, _ in loader:
            id_samples.append(x)
        id_samples = torch.cat(id_samples, dim=0)[:n_use]
        logger.info(f"  Loaded {n_use} train samples (calibration split)")
    else:
        # Test split — emit warning
        logger.warning("WARNING: using test split for Fisher may cause evaluation leakage.")
        logger.warning("  Recommend using --fisher_data_split train (default).")
        clean_dataset, _ = load_clean_test(configs)
        total_avail = len(clean_dataset)
        n_use = min(num_samples, total_avail)
        indices = list(range(n_use))
        subset = Subset(clean_dataset, indices)
        loader = DataLoader(subset, batch_size=configs.test_batch_size, shuffle=False)
        id_samples = []
        for x, _ in loader:
            id_samples.append(x)
        id_samples = torch.cat(id_samples, dim=0)[:n_use]

    # ---- Compute Fisher ----
    logger.info(f"Computing Fisher on {id_samples.size(0)} samples...")
    fisher = compute_fisher_information(
        model, id_samples, batch_size=configs.test_batch_size, device=device
    )

    # ---- Fisher diagnostics ----
    logger.info(f"  Fisher param count: {len(fisher)}")
    nonzero_counts = {}
    for k, v in fisher.items():
        nz = v.count_nonzero().item()
        nonzero_counts[k] = nz
    total_nz = sum(nonzero_counts.values())
    total_params = sum(v.numel() for v in fisher.values())
    logger.info(f"  Fisher nonzero entries: {total_nz} / {total_params} "
                f"({total_nz / max(total_params, 1) * 100:.1f}%)")

    # ---- Verify keys match model.named_parameters() ----
    model_param_names = {n for n, _ in model.named_parameters()}
    fisher_keys = set(fisher.keys())
    matched = fisher_keys & model_param_names
    missing_in_model = fisher_keys - model_param_names
    extra_in_fisher = model_param_names - fisher_keys
    logger.info(f"  Fisher keys matched to model: {len(matched)} / {len(fisher_keys)}")
    if missing_in_model:
        logger.warning(f"  Fisher keys NOT in model: {sorted(missing_in_model)[:10]}")
    logger.info(f"  Example keys: {sorted(fisher.keys())[:5]}")

    # ---- Save ----
    fisher_path = configs.fisher_path
    if fisher_path is None:
        fisher_path = os.path.join(
            configs.checkpoint_path,
            f"fisher_{configs.dataset}_{n_use}.pt",
        )
    elif not fisher_path.endswith(".pt"):
        fisher_path = os.path.join(
            fisher_path,
            f"fisher_{configs.dataset}_{n_use}.pt",
        )
    os.makedirs(os.path.dirname(fisher_path) or ".", exist_ok=True)
    torch.save(fisher, fisher_path)
    logger.info(f"Fisher saved to: {fisher_path}")


if __name__ == "__main__":
    run()
