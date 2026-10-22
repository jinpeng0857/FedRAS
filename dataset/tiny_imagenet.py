"""Tiny-ImageNet loaders that preserve the official validation-file layout."""

import os

import numpy as np
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, Subset
from torchvision.datasets.folder import default_loader

from dataset.dataset import (
    DATASET_CONFIGS,
    DATASET_PATHS,
    dirichlet_partition,
)


def tiny_imagenet_transform(train, configs=None):
    """64-pixel augmentation for the train split and deterministic clean eval."""
    cfg = DATASET_CONFIGS["tiny-imagenet"]
    normalize = transforms.Normalize(cfg["mean"], cfg["std"])
    if train:
        operations = [
            transforms.RandomCrop(cfg["img_size"], padding=4),
            transforms.RandomHorizontalFlip(),
        ]
        if configs is not None and getattr(configs, "use_randaugment", False):
            operations.append(
                transforms.RandAugment(
                    num_ops=getattr(configs, "randaugment_n", 2),
                    magnitude=getattr(configs, "randaugment_m", 9),
                )
            )
        operations.extend([
            transforms.ToTensor(),
            normalize,
        ])
        return transforms.Compose(operations)
    return transforms.Compose([transforms.ToTensor(), normalize])


class TinyImageNetValidationDataset(Dataset):
    """Read the official flat ``val/images`` split via ``val_annotations.txt``."""

    def __init__(self, data_path, class_to_idx, transform=None):
        val_dir = os.path.join(data_path, "val")
        images_dir = os.path.join(val_dir, "images")
        annotations_path = os.path.join(val_dir, "val_annotations.txt")
        if not os.path.isdir(images_dir):
            raise FileNotFoundError(f"Tiny-ImageNet validation images not found: {images_dir}")
        if not os.path.isfile(annotations_path):
            raise FileNotFoundError(
                f"Tiny-ImageNet validation annotations not found: {annotations_path}"
            )

        self.transform = transform
        self.samples = []
        with open(annotations_path, "r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                fields = line.rstrip("\n").split("\t")
                if len(fields) < 2:
                    raise ValueError(
                        f"invalid Tiny-ImageNet annotation at {annotations_path}:{line_number}"
                    )
                image_name, wnid = fields[:2]
                if wnid not in class_to_idx:
                    raise ValueError(
                        f"unknown Tiny-ImageNet class {wnid!r} in "
                        f"{annotations_path}:{line_number}"
                    )
                image_path = os.path.join(images_dir, image_name)
                if not os.path.isfile(image_path):
                    raise FileNotFoundError(
                        f"Tiny-ImageNet validation image not found: {image_path}"
                    )
                self.samples.append((image_path, class_to_idx[wnid]))
        if not self.samples:
            raise ValueError(f"no Tiny-ImageNet validation samples found in {annotations_path}")
        self.targets = [target for _, target in self.samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, target = self.samples[index]
        image = default_loader(path)
        if self.transform is not None:
            image = self.transform(image)
        return image, target


def load_tiny_imagenet_validation(data_path, transform):
    """Build a clean validation dataset without moving any images on disk."""
    train_dir = os.path.join(data_path, "train")
    train_dataset = torchvision.datasets.ImageFolder(root=train_dir)
    val_dir = os.path.join(data_path, "val")
    annotations_path = os.path.join(val_dir, "val_annotations.txt")
    if os.path.isfile(annotations_path):
        return TinyImageNetValidationDataset(
            data_path, train_dataset.class_to_idx, transform=transform
        )

    # Some mirrors reorganize val/images into WordNet-ID directories.
    val_dataset = torchvision.datasets.ImageFolder(root=val_dir, transform=transform)
    if val_dataset.class_to_idx != train_dataset.class_to_idx:
        raise ValueError(
            "Tiny-ImageNet train/val class mappings differ; use the official "
            "val_annotations.txt layout or matching WordNet-ID directories."
        )
    return val_dataset


def equal_size_dirichlet_partition(labels, num_clients, alpha, seed=42):
    """Assign every sample once while keeping every client at exact capacity.

    Each class has an independent Dirichlet preference over clients.  Samples
    are drawn according to that preference only from clients with remaining
    capacity, so no post-hoc random fill can dilute the intended class skew.
    """
    if alpha <= 0:
        raise ValueError(f"Dirichlet alpha must be > 0, got {alpha}")
    total = len(labels)
    if total % num_clients:
        raise ValueError(
            f"{total} samples cannot be split equally among {num_clients} clients"
        )
    rng = np.random.RandomState(seed)
    capacity = total // num_clients
    remaining_capacity = np.full(num_clients, capacity, dtype=np.int64)
    client_indices = [[] for _ in range(num_clients)]
    for class_id in np.unique(labels):
        preference = rng.dirichlet(np.full(num_clients, alpha))
        for sample_index in rng.permutation(np.flatnonzero(labels == class_id)):
            eligible = np.flatnonzero(remaining_capacity > 0)
            if not len(eligible):
                raise RuntimeError("all client capacities were exhausted too early")
            probabilities = preference[eligible]
            probabilities /= probabilities.sum()
            client_id = int(rng.choice(eligible, p=probabilities))
            client_indices[client_id].append(int(sample_index))
            remaining_capacity[client_id] -= 1
    if np.any(remaining_capacity):
        raise RuntimeError("equal-size Tiny-ImageNet partition left unused capacity")
    return client_indices


def load_tiny_imagenet(configs):
    """Load and Dirichlet-partition Tiny-ImageNet's 200-class train split."""
    name = "tiny-imagenet"
    data_path = configs.dataset_path or DATASET_PATHS[name]
    cfg = DATASET_CONFIGS[name]
    train_dir = os.path.join(data_path, "train")
    if not os.path.isdir(train_dir):
        raise FileNotFoundError(
            f"Tiny-ImageNet training directory not found: {train_dir}. "
            "Pass --dataset_path to the directory containing train/ and val/."
        )
    train_dataset = torchvision.datasets.ImageFolder(
        root=train_dir,
        transform=tiny_imagenet_transform(train=True, configs=configs),
    )
    if len(train_dataset.classes) != cfg["num_classes"]:
        raise ValueError(
            f"Tiny-ImageNet must contain {cfg['num_classes']} classes, found "
            f"{len(train_dataset.classes)} under {train_dir}"
        )
    partition = (
        equal_size_dirichlet_partition
        if getattr(configs, "equal_size", False)
        else dirichlet_partition
    )
    client_idxs = partition(
        labels=np.asarray(train_dataset.targets),
        num_clients=configs.num_client,
        alpha=configs.alpha,
        seed=configs.dataset_seed,
    )
    return [Subset(train_dataset, idxs) for idxs in client_idxs], cfg["num_classes"]
