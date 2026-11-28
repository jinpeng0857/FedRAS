"""Dataset, transform, and loader constructors for TTA evaluation."""

from pathlib import Path
from functools import lru_cache
from typing import Dict, Tuple
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision.datasets.folder import default_loader
import torchvision.transforms as transforms

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CORRUPTIONS = [
    "gaussian_noise",
    "shot_noise",
    "impulse_noise",
    "defocus_blur",
    "glass_blur",
    "motion_blur",
    "zoom_blur",
    "snow",
    "frost",
    "fog",
    "brightness",
    "contrast",
    "elastic_transform",
    "pixelate",
    "jpeg_compression",
]
REVISIT_CORRUPTIONS = CORRUPTIONS[:4]
CONTINUAL_CYCLES = 3
SEVERITY = 5

DATASET_STATS = {
    "cifar10": {
        "num_classes": 10,
        "mean": [0.4914, 0.4822, 0.4465],
        "std": [0.2023, 0.1994, 0.2010],
        "corruption_dir": PROJECT_ROOT / "data" / "CIFAR-10-C",
    },
    "cifar100": {
        "num_classes": 100,
        "mean": [0.5071, 0.4867, 0.4408],
        "std": [0.2675, 0.2565, 0.2761],
        "corruption_dir": PROJECT_ROOT / "data" / "CIFAR-100-C",
    },
    "tiny-imagenet": {
        "num_classes": 200,
        "mean": [0.4802, 0.4481, 0.3975],
        "std": [0.2302, 0.2265, 0.2262],
        "clean_dir": PROJECT_ROOT / "data" / "tiny-imagenet-200",
        "corruption_dir": PROJECT_ROOT / "data" / "Tiny-ImageNet-C",
    },
}


def _extract_tensor_normalize_stats(transform):
    """Return (mean, std) when transform is standard ToTensor + Normalize, else None."""
    if not isinstance(transform, transforms.Compose) or len(transform.transforms) != 2:
        return None
    first, second = transform.transforms
    if isinstance(first, transforms.ToTensor) and isinstance(second, transforms.Normalize):
        return second.mean, second.std
    return None


class CIFARCDataset(Dataset):
    """One corruption and severity block from the official CIFAR-C arrays."""

    def __init__(self, root: Path, corruption: str, transform):
        data_path = root / f"{corruption}.npy"
        labels_path = root / "labels.npy"
        if not data_path.is_file():
            raise FileNotFoundError(f"CIFAR-C data file not found: {data_path}")
        if not labels_path.is_file():
            raise FileNotFoundError(f"CIFAR-C labels file not found: {labels_path}")

        full_data = np.load(data_path, mmap_mode="r")
        labels = np.load(labels_path, mmap_mode="r")
        if len(full_data) % 5 != 0:
            raise ValueError(f"invalid CIFAR-C severity layout: {data_path}")

        per_severity = len(full_data) // 5
        start = (SEVERITY - 1) * per_severity
        end = SEVERITY * per_severity
        self.data = full_data[start:end]
        if len(labels) == per_severity:
            self.targets = labels
        elif len(labels) == len(full_data):
            self.targets = labels[start:end]
        else:
            raise ValueError(
                f"labels length must be {per_severity} or {len(full_data)}, "
                f"got {len(labels)}"
            )
        self.transform = transform
        self._tensor_data = None
        self._targets_tensor = None
        norm_stats = _extract_tensor_normalize_stats(transform)
        if norm_stats is not None:
            mean, std = norm_stats
            raw_uint8 = np.array(self.data, copy=True)
            tensor = torch.from_numpy(raw_uint8).permute(0, 3, 1, 2).float().div_(255.0)
            mean_t = torch.as_tensor(mean, dtype=torch.float32).view(1, -1, 1, 1)
            std_t = torch.as_tensor(std, dtype=torch.float32).view(1, -1, 1, 1)
            tensor.sub_(mean_t).div_(std_t)
            self._tensor_data = tensor
            self._targets_tensor = np.asarray(self.targets, dtype=np.int64).tolist()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if self._tensor_data is not None:
            return self._tensor_data[index], self._targets_tensor[index]
        image = Image.fromarray(self.data[index])
        return self.transform(image), int(self.targets[index])

    def close(self):
        self._tensor_data = None
        self._targets_tensor = None
        if hasattr(self.data, "_mmap") and self.data._mmap is not None:
            self.data._mmap.close()


@lru_cache(maxsize=4)
def _tiny_class_to_idx(clean_root: str):
    """Read the training ImageFolder mapping used by the Tiny-ImageNet model."""
    train_root = Path(clean_root) / "train"
    if not train_root.is_dir():
        raise FileNotFoundError(
            f"Tiny-ImageNet training directory not found: {train_root}"
        )
    dataset = datasets.ImageFolder(root=str(train_root))
    if len(dataset.class_to_idx) != DATASET_STATS["tiny-imagenet"]["num_classes"]:
        raise ValueError(
            "Tiny-ImageNet class mapping has "
            f"{len(dataset.class_to_idx)} classes, expected 200"
        )
    return dataset.class_to_idx


class TinyImageNetCDataset(Dataset):
    """One corruption/severity block from the directory-form Tiny-ImageNet-C."""

    _IMAGE_SUFFIXES = {".jpeg", ".jpg", ".png"}

    def __init__(self, root: Path, corruption: str, class_to_idx, transform):
        severity_root = Path(root) / corruption / str(SEVERITY)
        if not severity_root.is_dir():
            raise FileNotFoundError(
                f"Tiny-ImageNet-C severity directory not found: {severity_root}"
            )
        samples = []
        for class_name in sorted(class_to_idx):
            class_root = severity_root / class_name
            if not class_root.is_dir():
                raise FileNotFoundError(
                    f"Tiny-ImageNet-C class directory not found: {class_root}"
                )
            for image_path in sorted(class_root.iterdir()):
                if image_path.is_file() and image_path.suffix.lower() in self._IMAGE_SUFFIXES:
                    samples.append((str(image_path), int(class_to_idx[class_name])))
        if not samples:
            raise ValueError(f"no Tiny-ImageNet-C images found under {severity_root}")
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, target = self.samples[index]
        image = default_loader(path)
        if self.transform is not None:
            image = self.transform(image)
        return image, target

    def close(self):
        """Keep the same cleanup contract as CIFARCDataset."""


def build_clean_dataset(dataset_name: str, root: Path, transform):
    """Build the clean evaluation split without downloading or moving files."""
    if dataset_name == "tiny-imagenet":
        from dataset.tiny_imagenet import load_tiny_imagenet_validation

        return load_tiny_imagenet_validation(str(root), transform=transform)
    dataset_class = datasets.CIFAR10 if dataset_name == "cifar10" else datasets.CIFAR100
    return dataset_class(
        root=str(root),
        train=False,
        transform=transform,
        download=False,
    )


def build_transform(dataset_name: str):
    stats = DATASET_STATS[dataset_name]
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=stats["mean"], std=stats["std"]),
        ]
    )


def make_corruption_loader(args, corruption: str) -> Tuple[Dataset, DataLoader]:
    root = (
        Path(args.cifar_c_path)
        if args.cifar_c_path
        else DATASET_STATS[args.dataset]["corruption_dir"]
    )
    if args.dataset == "tiny-imagenet":
        clean_root = (
            Path(args.clean_data_root)
            if args.clean_data_root
            else DATASET_STATS["tiny-imagenet"]["clean_dir"]
        )
        dataset = TinyImageNetCDataset(
            root=root,
            corruption=corruption,
            class_to_idx=_tiny_class_to_idx(str(clean_root)),
            transform=build_transform(args.dataset),
        )
    else:
        dataset = CIFARCDataset(
            root=root,
            corruption=corruption,
            transform=build_transform(args.dataset),
        )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda") and torch.cuda.is_available(),
    )
    return dataset, loader
