"""
FEDEATA Federated Training Entry Point.

Replaces ACG-C federated_train.py Hydra entry.
Uses simple argparse config (project framework style).

Usage:
  python train_fedacg.py --method fedacg --fedacg_mode fedprox \
      --dataset cifar100 --alpha 0.3 --num_client 50
"""

import numpy as np
import torch
from dataset.dataset import load_cifar, load_cinic, load_imagenet
from src.model.resnet import resnet18
from src.model.eatac_resnet import eatac_resnet18, eatac_resnet34
from utils.config import parser
from utils.logger import logger, add_file_handler
from utils.utils import set_seed, make_save_path, prepare_server_and_clients


def run():
    configs = parser.parse_args()
    configs.step = "train"

    set_seed(configs.seed)
    save_path = make_save_path(configs)

    if configs.checkpoint_path == "default":
        configs.checkpoint_path = save_path

    # File logging
    add_file_handler(logger, save_path)

    logger.info("=" * 50)
    logger.info("FEDEATA Training Configuration")
    logger.info("=" * 50)
    for key, value in sorted(vars(configs).items()):
        logger.info(f"  {key}: {value}")

    # ---- Load data ----
    logger.info("Loading dataset...")
    if configs.dataset in ("cifar10", "cifar100"):
        train_datasets, num_class = load_cifar(configs)
        data_size = 32
    elif configs.dataset == "cinic":
        train_datasets, num_class = load_cinic(configs)
        data_size = 32
    elif configs.dataset == "imagenet":
        train_datasets, num_class = load_imagenet(configs)
        data_size = 224
    else:
        raise ValueError(f"Unknown dataset: {configs.dataset}")

    configs.num_class = num_class
    logger.info(f"  num_class={num_class}, clients={len(train_datasets)}")

    # ---- Build model ----
    logger.info("Building model...")
    if configs.backbone == "resnet18":
        backbone = resnet18(num_classes=num_class)
    elif configs.backbone == "eatac_resnet18":
        backbone = eatac_resnet18(
            num_classes=num_class,
            subnet_drop_rate=configs.subnet_drop_rate,
        )
    elif configs.backbone == "eatac_resnet34":
        backbone = eatac_resnet34(
            num_classes=num_class,
            subnet_drop_rate=configs.subnet_drop_rate,
        )
    else:
        raise ValueError(f"Unknown backbone: {configs.backbone}")

    logger.info(f"  backbone={configs.backbone}")

    # ---- Setup server and clients ----
    logger.info("Setting up server and clients...")
    server_class, client_class = prepare_server_and_clients(configs)
    device = torch.device(configs.device if torch.cuda.is_available() else "cpu")

    server = server_class(device, backbone, configs)
    clients = [
        client_class(cid, device, backbone, configs)
        for cid in range(configs.num_client)
    ]
    for cid, client in enumerate(clients):
        client.set_train_set(train_datasets[cid])
    server.clients.extend(clients)

    # ---- Diagnostics ----
    logger.info("=" * 50)
    logger.info("Client Data Diagnostics")
    logger.info("=" * 50)

    # 1. Model output layer
    fc_weight = backbone.fc.weight if hasattr(backbone, 'fc') else None
    if fc_weight is not None:
        logger.info(f"  Model fc.weight shape: {fc_weight.shape}  →  {fc_weight.shape[0]} classes")
    elif hasattr(backbone, 'module') and hasattr(backbone.module, 'fc'):
        logger.info(f"  Model fc.weight shape: {backbone.module.fc.weight.shape}  →  {backbone.module.fc.weight.shape[0]} classes")
    else:
        logger.info(f"  Model output classes: {num_class}")

    # 2. Label range across all training data (via Subset.dataset.targets)
    orig_targets = np.array(train_datasets[0].dataset.targets)
    all_labels = np.concatenate([orig_targets[ds.indices] for ds in train_datasets])
    logger.info(f"  Label range: [{all_labels.min()}, {all_labels.max()}]  (num_class={num_class}), "
                f"total samples={len(all_labels)}")

    # 3. Per-client sample counts
    client_sizes = [len(ds) for ds in train_datasets]
    logger.info(f"  Client sample counts: min={min(client_sizes)}, max={max(client_sizes)}, "
                f"mean={np.mean(client_sizes):.1f}")

    # 4. Per-client class distribution (first and last 5 clients)
    logger.info("  Per-client class distribution (first 5 clients):")
    for cid in range(min(5, len(clients))):
        client_labels = orig_targets[train_datasets[cid].indices]
        unique, counts = np.unique(client_labels, return_counts=True)
        dist_str = ", ".join([f"class {u}:{c}" for u, c in zip(unique, counts)])
        logger.info(f"    client {cid:3d}  ({client_sizes[cid]:4d} samples): {dist_str}")

    if len(clients) > 5:
        logger.info("  Per-client class distribution (last 5 clients):")
        for cid in range(max(5, len(clients) - 5), len(clients)):
            client_labels = orig_targets[train_datasets[cid].indices]
            unique, counts = np.unique(client_labels, return_counts=True)
            dist_str = ", ".join([f"class {u}:{c}" for u, c in zip(unique, counts)])
            logger.info(f"    client {cid:3d}  ({client_sizes[cid]:4d} samples): {dist_str}")

    logger.info("=" * 50)

    # ---- Train ----
    logger.info("Starting federated training...")
    server.fit()

    logger.info("Training complete!")


if __name__ == "__main__":
    run()
