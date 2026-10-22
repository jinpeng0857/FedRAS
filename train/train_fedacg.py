"""FedACG/FedAvg federated training entry point.

Usage:
  python -m train.train_fedacg --method fedacg --fedacg_mode fedacg \
      --dataset cifar100 --alpha 0.3 --num_client 100
"""

import numpy as np
import torch
from dataset.dataset import load_cifar, load_cinic, load_imagenet
from dataset.tiny_imagenet import load_tiny_imagenet
from utils.config import parser
from utils.logger import logger, add_file_handler
from utils.model_utils import SUPPORTED_BACKBONES, build_model
from utils.helpers import set_seed, make_save_path, prepare_server_and_clients


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
    logger.info(f"{configs.method.upper()} Training Configuration")
    logger.info("=" * 50)
    for key, value in sorted(vars(configs).items()):
        logger.info(f"  {key}: {value}")

    # ---- Load data ----
    logger.info("Loading dataset...")
    if configs.dataset in ("cifar10", "cifar100"):
        train_datasets, num_class = load_cifar(configs)
    elif configs.dataset == "cinic":
        train_datasets, num_class = load_cinic(configs)
    elif configs.dataset == "imagenet":
        train_datasets, num_class = load_imagenet(configs)
    elif configs.dataset == "tiny-imagenet":
        train_datasets, num_class = load_tiny_imagenet(configs)
    else:
        raise ValueError(f"Unknown dataset: {configs.dataset}")

    configs.num_class = num_class
    logger.info(f"  num_class={num_class}, clients={len(train_datasets)}")

    # ---- Build model ----
    logger.info("Building model...")
    if configs.backbone in SUPPORTED_BACKBONES:
        norm_type = "bn" if configs.use_bn_layer else "gn"
        backbone = build_model(
            configs.backbone,
            num_classes=num_class,
            norm_type=norm_type,
            num_groups=configs.num_groups,
            imagenet_stem=configs.dataset == "imagenet",
        )
    else:
        raise ValueError(f"Unknown backbone: {configs.backbone}")

    logger.info(f"  backbone={configs.backbone}, norm_type={norm_type}")

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
    base_dataset = train_datasets[0].dataset
    if hasattr(base_dataset, "targets"):
        orig_targets = np.array(base_dataset.targets)
    elif hasattr(base_dataset, "samples"):
        orig_targets = np.array([sample[1] for sample in base_dataset.samples])
    else:
        raise AttributeError(
            "Base dataset must expose either .targets or .samples for diagnostics"
        )
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
