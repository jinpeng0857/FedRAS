"""
FEDEATA dataset loading, preprocessing, partitioning, and corruption.

Consolidated from:
  - src/data/data_partition.py (loading, partitioning, transforms)
  - src/data/corruptions.py (CIFAR-C online corruptions)

Public API:
  Dataset classes:  CorruptedDataset, PrecomputedCDataset
  Corruption:       CIFARCorruption
  Loading:          load_cifar, load_cinic, load_tiny_imagenet,
                    load_clean_test, load_corrupted_test
  Partitioning:     dirichlet_partition
  Transforms:       get_transform
  Constants:        DATASET_PATHS, DATASET_CONFIGS, CIFAR_CORRUPTIONS
"""

import io
import math
import os
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset, Subset


# ========== Constants ==========

_DATA_ROOT = os.environ.get("FEDEATA_DATA_ROOT", "./data")

DATASET_PATHS = {
    "cifar10": os.path.join(_DATA_ROOT, "cifar10"),
    "cifar100": os.path.join(_DATA_ROOT, "cifar100"),
    "cinic": os.path.join(_DATA_ROOT, "cinic"),
    "tinyimagenet": os.path.join(_DATA_ROOT, "tiny-imagenet-200"),
    "imagenet": os.path.join(_DATA_ROOT, "ImageNet-C"),
}

DATASET_CONFIGS = {
    "cifar10": {"num_classes": 10, "img_size": 32, "mean": [0.4914, 0.4822, 0.4465], "std": [0.2023, 0.1994, 0.2010]},
    "cifar100": {"num_classes": 100, "img_size": 32, "mean": [0.5071, 0.4867, 0.4408], "std": [0.2675, 0.2565, 0.2761]},
    "cinic": {"num_classes": 10, "img_size": 32, "mean": [0.47889522, 0.47227842, 0.43047404], "std": [0.24205776, 0.23828046, 0.25874835]},
    "tinyimagenet": {"num_classes": 200, "img_size": 64, "mean": [0.4802, 0.4481, 0.3975], "std": [0.2302, 0.2265, 0.2262]},
    "imagenet": {"num_classes": 1000, "img_size": 224, "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
}

CIFAR_CORRUPTIONS = [
    "gaussian_noise", "shot_noise", "impulse_noise",
    "defocus_blur", "glass_blur", "motion_blur", "zoom_blur",
    "snow", "frost", "fog", "brightness", "contrast",
    "elastic_transform", "pixelate", "jpeg_compression",
]


# ========== Transforms ==========

def get_transform(dataset_name, train=True):
    cfg = DATASET_CONFIGS[dataset_name]
    img_size = cfg["img_size"]
    if train:
        return transforms.Compose([
            transforms.RandomCrop(img_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(cfg["mean"], cfg["std"]),
        ])
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
    Dirichlet non-IID partition with equal sample count per client.

    Each client receives exactly len(labels) // num_clients samples.
    Preserves Dirichlet non-IID character by:
      1) Running standard Dirichlet partition (non-IID, unequal sizes)
      2) Trimming over-filled clients + filling under-filled clients from pool

    The class distribution from step 1 is largely preserved in step 2
    because trimming keeps the Dirichlet ratio, and filling adds only a
    small fraction (< samples_per_client方差/总) of random samples.

    alpha=0.1 (extreme non-IID), alpha=1.0 (moderate), alpha=1000 (near-IID).
    """
    if alpha <= 0:
        raise ValueError(f"Dirichlet alpha must be > 0, got {alpha}")

    total = len(labels)
    samples_per_client = total // num_clients

    # Step 1: Standard Dirichlet partition (non-IID, unequal sizes)
    client_indices = dirichlet_partition(labels, num_clients, alpha, seed)

    # Step 2: Balance to equal sizes
    rng = np.random.RandomState(seed + 999)

    pool = []
    for k in range(num_clients):
        if len(client_indices[k]) > samples_per_client:
            # Randomly trim to samples_per_client, move excess to pool
            kept = rng.choice(
                np.array(client_indices[k]), samples_per_client, replace=False
            ).tolist()
            kept_set = set(kept)
            pool.extend([x for x in client_indices[k] if x not in kept_set])
            client_indices[k] = kept

    rng.shuffle(pool)
    pool_ptr = 0

    for k in range(num_clients):
        deficit = samples_per_client - len(client_indices[k])
        if deficit > 0:
            client_indices[k].extend(pool[pool_ptr:pool_ptr + deficit])
            pool_ptr += deficit

    return client_indices


# ========== FL Dataset Loaders ==========

def load_cifar(configs):
    """Load CIFAR-10/100 with Dirichlet partition."""
    name = configs.dataset
    cfg = DATASET_CONFIGS[name]
    data_path = configs.dataset_path or DATASET_PATHS[name]

    if name == "cifar10":
        dataset_cls = torchvision.datasets.CIFAR10
    else:
        dataset_cls = torchvision.datasets.CIFAR100

    train_dataset = dataset_cls(
        root=data_path, train=True,
        transform=get_transform(name, train=True),
        download=True,
    )

    # Dirichlet partition (optional equal-size mode)
    equal_size = getattr(configs, "equal_size", False)
    if equal_size:
        client_idxs = dirichlet_partition_equal_size(
            labels=np.array(train_dataset.targets),
            num_clients=configs.num_client,
            alpha=configs.alpha,
            seed=configs.dataset_seed,
        )
    else:
        client_idxs = dirichlet_partition(
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


def load_tiny_imagenet(configs):
    """Load Tiny-ImageNet with Dirichlet partition."""
    name = "tinyimagenet"
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


# ========== Clean Test Data Loader ==========

def load_clean_test(configs):
    """Load clean (ID) test dataset."""
    name = configs.dataset
    cfg = DATASET_CONFIGS[name]
    data_path = configs.dataset_path or DATASET_PATHS[name]
    transform = get_transform(name, train=False)

    if name == "cifar10":
        dataset = torchvision.datasets.CIFAR10(
            root=data_path, train=False,
            transform=transform, download=True,
        )
    elif name == "cifar100":
        dataset = torchvision.datasets.CIFAR100(
            root=data_path, train=False,
            transform=transform, download=True,
        )
    else:
        # For image-folder datasets
        val_dir = os.path.join(data_path, "val")
        if not os.path.isdir(val_dir):
            raise FileNotFoundError(
                f"Validation directory not found: {val_dir}\n"
                f"Expected for dataset '{name}' at: {val_dir}"
            )
        dataset = torchvision.datasets.ImageFolder(root=val_dir, transform=transform)

    return dataset, cfg["num_classes"]


# ========== CIFAR-C Corruption Functions ==========

def _clipped_zoom(img, factor):
    """Zoom into or out of an image, then resize back."""
    h, w = img.shape[-2], img.shape[-1]
    if factor < 1:
        # Zoom in: crop center
        new_h, new_w = int(h * factor), int(w * factor)
        top = (h - new_h) // 2
        left = (w - new_w) // 2
        cropped = img[..., top:top + new_h, left:left + new_w]
        return F.interpolate(cropped.unsqueeze(0), size=(h, w), mode='bilinear',
                             align_corners=False).squeeze(0)
    elif factor > 1:
        # Zoom out: pad and resize
        new_h, new_w = int(h / factor), int(w / factor)
        resized = F.interpolate(img.unsqueeze(0), size=(new_h, new_w),
                                mode='bilinear', align_corners=False).squeeze(0)
        pad_h, pad_w = (h - new_h) // 2, (w - new_w) // 2
        padded = F.pad(resized, [pad_w, w - new_w - pad_w, pad_h, h - new_h - pad_h])
        return padded
    return img


def _disk_kernel(radius):
    """Create a disk-shaped kernel for motion/glass blur."""
    ax = torch.arange(-radius, radius + 1, dtype=torch.float32)
    xx, yy = torch.meshgrid(ax, ax, indexing='ij')
    kernel = (xx ** 2 + yy ** 2) <= radius ** 2
    return kernel.float() / kernel.sum()


def _gaussian_kernel(size, sigma):
    ax = torch.arange(-size // 2 + 1, size // 2 + 1, dtype=torch.float32)
    xx, yy = torch.meshgrid(ax, ax, indexing='ij')
    kernel = torch.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    return kernel / kernel.sum()


class CIFARCorruption:
    """Apply CIFAR-C style corruptions to CIFAR images (tensors [0,1])."""

    @staticmethod
    def get_fn(name):
        fn_map = {
            "gaussian_noise": CIFARCorruption.gaussian_noise,
            "shot_noise": CIFARCorruption.shot_noise,
            "impulse_noise": CIFARCorruption.impulse_noise,
            "defocus_blur": CIFARCorruption.defocus_blur,
            "glass_blur": CIFARCorruption.glass_blur,
            "motion_blur": CIFARCorruption.motion_blur,
            "zoom_blur": CIFARCorruption.zoom_blur,
            "snow": CIFARCorruption.snow,
            "frost": CIFARCorruption.frost,
            "fog": CIFARCorruption.fog,
            "brightness": CIFARCorruption.brightness,
            "contrast": CIFARCorruption.contrast,
            "elastic_transform": CIFARCorruption.elastic_transform,
            "pixelate": CIFARCorruption.pixelate,
            "jpeg_compression": CIFARCorruption.jpeg_compression,
        }
        if name not in fn_map:
            raise KeyError(f"Unknown corruption type: '{name}'. Valid: {list(fn_map.keys())}")
        return fn_map[name]

    @staticmethod
    def gaussian_noise(x, severity=1):
        std = [0.04, 0.06, 0.08, 0.09, 0.10][min(severity, 5) - 1]
        noise = torch.randn_like(x) * std
        return torch.clamp(x + noise, 0, 1)

    @staticmethod
    def shot_noise(x, severity=1):
        scale = [60, 25, 12, 5, 3][min(severity, 5) - 1]
        x_scaled = x * scale
        noisy = torch.poisson(x_scaled) / scale
        return torch.clamp(noisy, 0, 1)

    @staticmethod
    def impulse_noise(x, severity=1):
        amount = [0.01, 0.02, 0.03, 0.05, 0.07][min(severity, 5) - 1]
        mask = torch.rand_like(x)
        noisy = x.clone()
        noisy[mask < amount / 2] = 1.0
        noisy[(mask >= amount / 2) & (mask < amount)] = 0.0
        return noisy

    @staticmethod
    def defocus_blur(x, severity=1):
        radius = [1, 2, 3, 4, 6][min(severity, 5) - 1]
        kernel_size = radius * 2 + 1
        kernel = _disk_kernel(radius)
        kernel = kernel.view(1, 1, kernel_size, kernel_size).repeat(3, 1, 1, 1)
        x_blurred = F.conv2d(
            x.unsqueeze(0), kernel.to(x.device), groups=3, padding=radius
        ).squeeze(0)
        return torch.clamp(x_blurred, 0, 1)

    @staticmethod
    def glass_blur(x, severity=1):
        """Glass blur: defocus blur + random pixel offsets (vectorized)."""
        radius = [0.5, 0.5, 0.5, 0.5, 0.5][min(severity, 5) - 1]
        sigma = [0.7, 0.9, 1.1, 1.3, 1.5][min(severity, 5) - 1]
        max_displace = [1, 2, 3, 4, 5][min(severity, 5) - 1]
        repeats = [1, 1, 1, 2, 2][min(severity, 5) - 1]

        # Defocus blur first
        kernel_size = int(2 * radius + 1)
        if kernel_size < 3:
            kernel_size = 3
        kernel = _gaussian_kernel(kernel_size, sigma)
        kernel = kernel.view(1, 1, kernel_size, kernel_size).repeat(3, 1, 1, 1)
        x_blurred = F.conv2d(
            x.unsqueeze(0), kernel.to(x.device), groups=3,
            padding=kernel_size // 2
        ).squeeze(0)

        # Vectorized random pixel swaps
        c, h, w = x_blurred.shape
        device = x_blurred.device
        for _ in range(repeats):
            # Generate all displacement vectors at once (h, w)
            di = torch.randint(-max_displace, max_displace + 1, (h, w), device=device)
            dj = torch.randint(-max_displace, max_displace + 1, (h, w), device=device)
            # Source coordinate grids
            i_src = torch.arange(h, device=device).view(-1, 1)  # (h, 1)
            j_src = torch.arange(w, device=device).view(1, -1)  # (1, w)
            # Target coordinates with boundary clamping
            ni = (i_src + di).clamp(0, h - 1)
            nj = (j_src + dj).clamp(0, w - 1)
            # Vectorized gather: each pixel takes value from a nearby displaced position
            x_blurred = x_blurred[:, ni, nj]

        return torch.clamp(x_blurred, 0, 1)

    @staticmethod
    def motion_blur(x, severity=1):
        """Motion blur using a directional line kernel."""
        kernel_size = [7, 9, 11, 13, 15][min(severity, 5) - 1]
        angle = float(torch.empty(1).uniform_(0, 180).item())
        rad = math.radians(angle)
        kernel = torch.zeros(kernel_size, kernel_size)
        center = kernel_size // 2
        for i in range(kernel_size):
            t = (i - center) / center
            x_pos = int(round(center + t * center * math.cos(rad)))
            y_pos = int(round(center + t * center * math.sin(rad)))
            if 0 <= x_pos < kernel_size and 0 <= y_pos < kernel_size:
                kernel[y_pos, x_pos] = 1.0
        kernel = kernel / kernel.sum()
        kernel = kernel.view(1, 1, kernel_size, kernel_size).repeat(3, 1, 1, 1)
        x_blurred = F.conv2d(
            x.unsqueeze(0), kernel.to(x.device), groups=3,
            padding=kernel_size // 2
        ).squeeze(0)
        return torch.clamp(x_blurred, 0, 1)

    @staticmethod
    def zoom_blur(x, severity=1):
        """Zoom blur: create concentric scaled versions and blend."""
        intensities = [0.15, 0.25, 0.35, 0.45, 0.55][min(severity, 5) - 1]
        c, h, w = x.shape
        result = torch.zeros_like(x)
        num_steps = 10
        for step in range(num_steps):
            factor = 1.0 - step / num_steps * intensities
            result += _clipped_zoom(x, factor)
        result /= num_steps
        return result

    @staticmethod
    def snow(x, severity=1):
        """Snow effect: add brightness + motion blur with speckle."""
        snow_level = [0.1, 0.2, 0.3, 0.4, 0.5][min(severity, 5) - 1]
        c, h, w = x.shape
        snow_layer = torch.rand(1, h, w, device=x.device)
        snow_layer = F.interpolate(snow_layer.unsqueeze(0), size=(h, w),
                                    mode='bilinear', align_corners=False).squeeze(0)
        snow_mask = (snow_layer > 0.7).float()
        result = x * (1 + snow_level * 0.5)
        # Apply motion blur to snow
        kernel = _gaussian_kernel(5, 1.0)
        kernel = kernel.view(1, 1, 5, 5).repeat(3, 1, 1, 1)
        snow_mask_blurred = F.conv2d(
            snow_mask.unsqueeze(0), kernel.to(x.device), groups=1,
            padding=2
        ).squeeze(0)
        result = torch.where(snow_mask_blurred > 0.3,
                             result, x)
        return torch.clamp(result, 0, 1)

    @staticmethod
    def frost(x, severity=1):
        """Frost effect: blend image with frost pattern (simulated)."""
        intensity = [0.1, 0.2, 0.3, 0.4, 0.5][min(severity, 5) - 1]
        c, h, w = x.shape
        # Simulate frost pattern with Perlin-like noise
        frost_pattern = torch.rand(c, h, w, device=x.device)
        frost_pattern = (frost_pattern > intensity).float()
        kernel = _gaussian_kernel(3, 1.0)
        kernel = kernel.view(1, 1, 3, 3)
        # Apply blur to frost pattern for smoother look
        for _ in range(3):
            frost_pattern = F.conv2d(frost_pattern.unsqueeze(0),
                                     kernel.repeat(c, 1, 1, 1).to(x.device),
                                     groups=c, padding=1).squeeze(0).clamp(0, 1)
        result = x * (1 - frost_pattern * 0.5) + frost_pattern * 0.8
        return torch.clamp(result, 0, 1)

    @staticmethod
    def fog(x, severity=1):
        """Fog effect: blend with uniform haze, stronger for distant pixels."""
        fog_level = [0.2, 0.3, 0.4, 0.5, 0.6][min(severity, 5) - 1]
        c, h, w = x.shape
        # Create depth-like gradient fog
        fog_map = torch.ones(h, w, device=x.device)
        for i in range(h):
            fog_map[i, :] = 1.0 - fog_level * (i / h)
        fog_color = torch.ones(c, 1, 1, device=x.device) * 0.5
        result = x * (1 - fog_map.unsqueeze(0)) + fog_color * fog_map.unsqueeze(0)
        return torch.clamp(result, 0, 1)

    @staticmethod
    def brightness(x, severity=1):
        delta = [0.05, 0.1, 0.15, 0.2, 0.3][min(severity, 5) - 1]
        return torch.clamp(x + delta, 0, 1)

    @staticmethod
    def contrast(x, severity=1):
        factor = [0.85, 0.65, 0.45, 0.25, 0.1][min(severity, 5) - 1]
        mean = x.mean(dim=(1, 2), keepdim=True)
        return torch.clamp(factor * (x - mean) + mean, 0, 1)

    @staticmethod
    def elastic_transform(x, severity=1):
        """Elastic transformation using random displacement fields."""
        sigma = [0.2, 0.3, 0.4, 0.5, 0.6][min(severity, 5) - 1]
        c, h, w = x.shape
        # Create random displacement fields
        dx = torch.rand(1, h, w, device=x.device) * 2 - 1
        dy = torch.rand(1, h, w, device=x.device) * 2 - 1
        # Apply gaussian blur to displacement fields
        kernel_size = int(sigma * 10) | 1  # make odd
        if kernel_size < 3:
            kernel_size = 3
        kernel = _gaussian_kernel(kernel_size, sigma * 4)
        kernel_2d = kernel.view(1, 1, kernel_size, kernel_size)
        dx = F.conv2d(dx.unsqueeze(0), kernel_2d.to(x.device),
                       padding=kernel_size // 2).squeeze(0)
        dy = F.conv2d(dy.unsqueeze(0), kernel_2d.to(x.device),
                       padding=kernel_size // 2).squeeze(0)
        # Scale displacement
        scale = [2, 4, 6, 8, 10][min(severity, 5) - 1]
        dx, dy = dx * scale, dy * scale
        # Create sampling grid
        grid_y, grid_x = torch.meshgrid(
            torch.arange(h, dtype=torch.float32, device=x.device),
            torch.arange(w, dtype=torch.float32, device=x.device),
            indexing='ij',
        )
        grid_x = (grid_x + dx[0]) / (w - 1) * 2 - 1
        grid_y = (grid_y + dy[0]) / (h - 1) * 2 - 1
        grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)
        result = F.grid_sample(
            x.unsqueeze(0), grid, mode='bilinear',
            padding_mode='reflection', align_corners=True
        ).squeeze(0)
        return torch.clamp(result, 0, 1)

    @staticmethod
    def pixelate(x, severity=1):
        scales = [0.9, 0.7, 0.5, 0.33, 0.2][min(severity, 5) - 1]
        _, h, w = x.shape
        new_h, new_w = max(1, int(h * scales)), max(1, int(w * scales))
        x_small = F.interpolate(x.unsqueeze(0), size=(new_h, new_w), mode='nearest')
        x_large = F.interpolate(x_small, size=(h, w), mode='nearest').squeeze(0)
        return x_large

    @staticmethod
    def jpeg_compression(x, severity=1):
        quality = [25, 18, 15, 10, 7][min(severity, 5) - 1]
        x_np = (x.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        img = Image.fromarray(x_np)
        buf = io.BytesIO()
        img.save(buf, format='JPEG', quality=quality)
        buf.seek(0)
        img_compressed = Image.open(buf)
        result = np.array(img_compressed).astype(np.float32) / 255.0
        return torch.from_numpy(result).permute(2, 0, 1)


# ========== Corruption Dataset Wrappers ==========

class CorruptedDataset(Dataset):
    """Base class for corrupted test datasets."""

    def __init__(self, dataset, corruption_fn, severity=5):
        self.dataset = dataset
        self.corruption_fn = corruption_fn
        self.severity = severity

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        x = self.corruption_fn(x, self.severity)
        return x, y


class PrecomputedCDataset(Dataset):
    """Load pre-computed CIFAR-C / ImageNet-C from .npy files.

    5 severity levels are stacked in the data file:
        {root_path}/{corruption}.npy   shape (N*5, H, W, 3), uint8
        {root_path}/labels.npy         shape (N,) or (N*5,) int64/uint8

    severity=1 -> data rows [0:N], =2 -> [N:2N], ..., =5 -> [4N:5N]
    If labels has same length as data, slice it per-severity too.
    If labels has 1/5 the length, use the full labels array (shared across severities).
    """

    def __init__(self, root_path, corruption_type, severity=5, transform=None):
        self.corruption_type = corruption_type
        self.severity = severity

        data_path = os.path.join(root_path, f"{corruption_type}.npy")
        labels_path = os.path.join(root_path, "labels.npy")

        if not os.path.exists(data_path):
            raise FileNotFoundError(
                f"Pre-computed corruption file not found: {data_path}\n"
                f"Expected at: {root_path}/{corruption_type}.npy"
            )
        if not os.path.exists(labels_path):
            raise FileNotFoundError(
                f"Labels file not found: {labels_path}\n"
                f"Expected at: {root_path}/labels.npy"
            )

        full_data = np.load(data_path)  # (N*5, H, W, 3), uint8
        labels = np.load(labels_path)   # (N,) or (N*5,)

        n_per_severity = full_data.shape[0] // 5
        start = (severity - 1) * n_per_severity
        end = severity * n_per_severity
        self.data = full_data[start:end]  # (N, H, W, 3)

        if len(labels) == full_data.shape[0]:
            # Labels also stacked per-severity
            self.targets = labels[start:end]
        else:
            # Single set of labels shared across all severities
            self.targets = labels

        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]  # (H, W, 3), uint8
        target = int(self.targets[idx])

        # Convert uint8 numpy to PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)
        return img, target


class NormalizedCorruptedDataset(Dataset):
    """Wrap a corrupted dataset with normalization."""

    def __init__(self, corr_ds, mean, std):
        self.corr_ds = corr_ds
        self.norm = transforms.Normalize(mean, std)

    def __len__(self):
        return len(self.corr_ds)

    def __getitem__(self, idx):
        x, y = self.corr_ds[idx]
        return self.norm(x), y


# ========== Corrupted Test Data Loader ==========

def load_corrupted_test(configs, corruption_type, severity=5):
    """
    Load a corrupted test dataset.

    Two modes:
      1) Pre-computed C (CIFAR-C / ImageNet-C): set configs.precomputed_c_path
         Loads from .npy files: {precomputed_c_path}/{corruption}.npy
      2) Online corruption (fallback): applies corruption on the fly
    """
    name = configs.dataset
    cfg = DATASET_CONFIGS[name]

    # ---- Mode 1: Pre-computed C ----
    c_path = getattr(configs, "precomputed_c_path", None)
    if c_path is not None:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(cfg["mean"], cfg["std"]),
        ])
        dataset = PrecomputedCDataset(
            root_path=c_path,
            corruption_type=corruption_type,
            severity=severity,
            transform=transform,
        )
        return dataset

    # ---- Mode 2: Online corruption (existing) ----
    data_path = configs.dataset_path or DATASET_PATHS[name]
    base_transform = transforms.Compose([transforms.ToTensor()])

    if name == "cifar10":
        base_dataset = torchvision.datasets.CIFAR10(
            root=data_path, train=False,
            transform=base_transform, download=True,
        )
    elif name == "cifar100":
        base_dataset = torchvision.datasets.CIFAR100(
            root=data_path, train=False,
            transform=base_transform, download=True,
        )
    else:
        raise ValueError(f"Corrupted data not supported for {name}")

    corrupt_fn = CIFARCorruption.get_fn(corruption_type)
    corrupted = CorruptedDataset(base_dataset, corrupt_fn, severity)

    return NormalizedCorruptedDataset(corrupted, cfg["mean"], cfg["std"])
