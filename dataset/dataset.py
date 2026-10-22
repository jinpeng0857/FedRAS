"""
FEDEATA dataset loading, preprocessing, and partitioning (training only).

Public API:
  Loading:          load_cifar, load_cinic, load_imagenet
  Partitioning:     dirichlet_partition, dirichlet_partition_equal_size
  Transforms:       get_transform
  Constants:        DATASET_PATHS, DATASET_CONFIGS
"""

import os
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Subset


# ========== Constants ==========

_DATA_ROOT = os.environ.get("FEDEATA_DATA_ROOT", "./data")

DATASET_PATHS = {
    "cifar10": os.path.join(_DATA_ROOT, "cifar10"),
    "cifar100": os.path.join(_DATA_ROOT, "cifar100"),
    "cinic": os.path.join(_DATA_ROOT, "cinic"),
    "imagenet": os.path.join(_DATA_ROOT, "imagenet"),
}

DATASET_CONFIGS = {
    "cifar10": {"num_classes": 10, "img_size": 32, "mean": [0.4914, 0.4822, 0.4465], "std": [0.2023, 0.1994, 0.2010]},
    "cifar100": {"num_classes": 100, "img_size": 32, "mean": [0.5071, 0.4867, 0.4408], "std": [0.2675, 0.2565, 0.2761]},
    "cinic": {"num_classes": 10, "img_size": 32, "mean": [0.47889522, 0.47227842, 0.43047404], "std": [0.24205776, 0.23828046, 0.25874835]},
    "imagenet": {"num_classes": 1000, "img_size": 224, "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
}


# ========== Transforms ==========

def get_transform(dataset_name, train=True, configs=None):
    cfg = DATASET_CONFIGS[dataset_name]
    img_size = cfg["img_size"]
    if dataset_name == "imagenet":
        if train:
            return transforms.Compose([
                transforms.RandomResizedCrop(img_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(cfg["mean"], cfg["std"]),
            ])
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(cfg["mean"], cfg["std"]),
        ])
    if train:
        transform_ops = [
            transforms.RandomCrop(img_size, padding=4),
            transforms.RandomHorizontalFlip(),
        ]
        if (
            dataset_name == "cifar100"
            and configs is not None
            and getattr(configs, "use_randaugment", False)
        ):
            transform_ops.append(
                transforms.RandAugment(
                    num_ops=getattr(configs, "randaugment_n", 2),
                    magnitude=getattr(configs, "randaugment_m", 9),
                )
            )
        transform_ops.extend([
            transforms.ToTensor(),
            transforms.Normalize(cfg["mean"], cfg["std"]),
        ])
        return transforms.Compose(transform_ops)
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(cfg["mean"], cfg["std"]),
        ])


# ========== Dirichlet Partition ==========

def dirichlet_partition(labels, num_clients, alpha, seed=42):
    """
    Partition data using Dirichlet distribution.

    Each client gets samples from each class proportional to Dirichlet(alpha).
    alpha=0 means extreme non-IID, alpha=inf means IID.
    """
    if alpha <= 0:
        raise ValueError(f"Dirichlet alpha must be > 0, got {alpha}")
    rng = np.random.RandomState(seed)
    num_classes = len(np.unique(labels))
    class_indices = [np.where(labels == c)[0] for c in range(num_classes)]

    client_indices = [[] for _ in range(num_clients)]

    for c in range(num_classes):
        # Dirichlet proportions for this class across clients
        proportions = rng.dirichlet(np.repeat(alpha, num_clients))
        proportions = (proportions * len(class_indices[c])).astype(int)

        # Adjust for rounding
        diff = len(class_indices[c]) - proportions.sum()
        proportions[0] += diff

        # Shuffle and assign
        perm = rng.permutation(class_indices[c])
        start = 0
        for k in range(num_clients):
            client_indices[k].extend(
                perm[start : start + proportions[k]].tolist()
            )
            start += proportions[k]

    return client_indices


def dirichlet_partition_equal_size(labels, num_clients, alpha, seed=42):
    """
    Dirichlet non-IID partition with exactly equal sample count per client.

    Each client receives exactly len(labels) // num_clients samples.
    Within each client, class proportions follow Dirichlet(alpha) non-IID.

    Algorithm (single-pass, no trim-and-fill dilution):
      1) For each class c, draw Dirichlet proportions p_c ~ Dir(alpha) across clients.
      2) Base allocation: client k gets floor(p_c[k] * size(class_c)) samples of class c.
      3) Priority fill: remaining samples of class c are allocated one-by-one
         to the client with the highest p_c[k] that is still below capacity.

    This guarantees both exact equal sizes AND strict Dirichlet non-IID
    (no random pool fill that would dilute the Dirichlet distribution).

    alpha=0.1 (extreme non-IID), alpha=1.0 (moderate), alpha=1000 (near-IID).
    """
    if alpha <= 0:
        raise ValueError(f"Dirichlet alpha must be > 0, got {alpha}")

    rng = np.random.RandomState(seed)
    num_classes = len(np.unique(labels))
    class_indices = [np.where(labels == c)[0] for c in range(num_classes)]
    class_sizes = np.array([len(idx) for idx in class_indices])

    total = len(labels)
    samples_per_client = total // num_clients

    # Step 1: For each class, draw Dirichlet proportions across clients
    class_props = rng.dirichlet(np.repeat(alpha, num_clients), size=num_classes)

    # Step 2: Base allocation (floor)
    allocation = np.floor(class_props * class_sizes.reshape(-1, 1)).astype(int)
    client_totals = allocation.sum(axis=0)

    # Step 3: Priority fill — one-by-one for remaining samples per class
    remaining_class = class_sizes - allocation.sum(axis=1)
    for c in range(num_classes):
        n_remain = int(remaining_class[c])
        if n_remain <= 0:
            continue
        # Sort clients by Dirichlet proportion descending (priority = deserves more)
        order = np.argsort(class_props[c])[::-1]
        idx = 0
        while n_remain > 0 and idx < len(order):
            k = order[idx]
            if client_totals[k] < samples_per_client:
                allocation[c, k] += 1
                client_totals[k] += 1
                n_remain -= 1
            else:
                idx += 1

    # Convert allocation matrix to per-client index lists
    client_indices = [[] for _ in range(num_clients)]
    for c in range(num_classes):
        perm = rng.permutation(class_indices[c])
        start = 0
        for k in range(num_clients):
            n = int(allocation[c, k])
            if n > 0:
                client_indices[k].extend(perm[start:start + n].tolist())
                start += n

    return client_indices


# ========== FL Dataset Loaders ==========

def load_cifar(configs):
    """Load CIFAR-10/100 with equal-size Dirichlet non-IID partition."""
    name = configs.dataset
    cfg = DATASET_CONFIGS[name]
    data_path = configs.dataset_path or DATASET_PATHS[name]

    if name == "cifar10":
        dataset_cls = torchvision.datasets.CIFAR10
    else:
        dataset_cls = torchvision.datasets.CIFAR100

    train_dataset = dataset_cls(
        root=data_path, train=True,
        transform=get_transform(name, train=True, configs=configs),
        download=True,
    )

    # Equal-size Dirichlet non-IID partition
    # Each client gets exactly len(train_dataset) // num_client samples,
    # with Dirichlet(alpha) non-IID class distribution within each client.
    client_idxs = dirichlet_partition_equal_size(
        labels=np.array(train_dataset.targets),
        num_clients=configs.num_client,
        alpha=configs.alpha,
        seed=configs.dataset_seed,
    )

    client_datasets = []
    for idxs in client_idxs:
        client_datasets.append(Subset(train_dataset, idxs))

    return client_datasets, cfg["num_classes"]


def load_cinic(configs):
    """Load CINIC-10 with Dirichlet partition."""
    name = "cinic"
    data_path = configs.dataset_path or DATASET_PATHS[name]
    cfg = DATASET_CONFIGS[name]

    # CINIC-10: each class is a subdirectory under train/valid/test
    train_dir = os.path.join(data_path, "train")
    train_dataset = torchvision.datasets.ImageFolder(
        root=train_dir,
        transform=get_transform(name, train=True),
    )

    client_idxs = dirichlet_partition(
        labels=np.array([s[1] for s in train_dataset.samples]),
        num_clients=configs.num_client,
        alpha=configs.alpha,
        seed=configs.dataset_seed,
    )

    client_datasets = [Subset(train_dataset, idxs) for idxs in client_idxs]
    return client_datasets, cfg["num_classes"]


def load_imagenet(configs):
    """Load ImageNet with Dirichlet partition (class subdirectories)."""
    name = "imagenet"
    data_path = configs.dataset_path or DATASET_PATHS[name]
    cfg = DATASET_CONFIGS[name]

    train_dir = os.path.join(data_path, "train")
    train_dataset = torchvision.datasets.ImageFolder(
        root=train_dir,
        transform=get_transform(name, train=True),
    )

    client_idxs = dirichlet_partition(
        labels=np.array([s[1] for s in train_dataset.samples]),
        num_clients=configs.num_client,
        alpha=configs.alpha,
        seed=configs.dataset_seed,
    )

    client_datasets = [Subset(train_dataset, idxs) for idxs in client_idxs]
    return client_datasets, cfg["num_classes"]

