"""
Target client builder for Federated Test-Time Adaptation (FedTTA).

Constructs single-corruption target clients for CIFAR-10-C / CIFAR-100-C.

Each target client contains only ONE corruption type, enabling:
  - Oracle domain-aware aggregation (known corruption labels)
  - Per-domain adapter learning
  - Evaluation of domain-specific TTA performance

Usage:
    from dataset.target_clients import build_single_corruption_target_clients
    clients = build_single_corruption_target_clients(
        dataset_name="cifar10",
        corruptions=["gaussian_noise", "motion_blur", "fog"],
        severity=5,
        clients_per_corruption=2,
        samples_per_client=500,
    )
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

from dataset.dataset import (
    DATASET_CONFIGS,
    STANDARD_CORRUPTIONS,
    load_corrupted_test,
)


def _extract_labels_fast(dataset, num_samples=None):
    """Extract labels from a (possibly wrapped) dataset without iterating __getitem__.

    Handles PrecomputedCDataset, NormalizedCorruptedDataset, CorruptedDataset,
    and plain torchvision datasets by walking the wrapper chain for .targets.
    Falls back to iteration only when necessary.
    """
    total = num_samples if num_samples is not None else len(dataset)

    # Walk the wrapper chain looking for .targets
    targets = None
    ds = dataset
    seen = set()
    while ds is not None and id(ds) not in seen:
        seen.add(id(ds))
        if hasattr(ds, 'targets'):
            targets = ds.targets
            break
        # Unwrap known wrappers
        for attr in ('corr_ds', 'dataset'):
            if hasattr(ds, attr):
                ds = getattr(ds, attr)
                break
        else:
            break

    if targets is not None:
        if isinstance(targets, (list, np.ndarray)):
            return np.array(targets[:total])
        elif isinstance(targets, torch.Tensor):
            return targets[:total].cpu().numpy()

    # Fallback: extract via __getitem__ (only label, no heavy transform)
    return np.array([dataset[i][1] for i in range(total)])


def _assign_samples_to_clients(
    total_samples: int,
    clients_per_corruption: int,
    label_skew: bool,
    num_classes: int,
    labels: np.ndarray,
    alpha: float,
    seed: int,
):
    """Assign sample indices to clients, optionally with Dirichlet label skew."""
    if label_skew:
        # Dirichlet label-skewed partition within a corruption type
        rng = np.random.RandomState(seed)
        class_indices = [np.where(labels == c)[0] for c in range(num_classes)]
        client_indices = [[] for _ in range(clients_per_corruption)]

        for c in range(num_classes):
            proportions = rng.dirichlet(np.repeat(alpha, clients_per_corruption))
            proportions = (proportions * len(class_indices[c])).astype(int)

            diff = len(class_indices[c]) - proportions.sum()
            if diff != 0:
                proportions[0] += diff

            perm = rng.permutation(class_indices[c])
            start = 0
            for k in range(clients_per_corruption):
                client_indices[k].extend(
                    perm[start: start + proportions[k]].tolist()
                )
                start += proportions[k]
        return client_indices
    else:
        # Uniform partition (label-IID): each client gets equal share
        rng = np.random.RandomState(seed)
        perm = rng.permutation(total_samples)
        samples_per = total_samples // clients_per_corruption
        remainder = total_samples % clients_per_corruption

        client_indices = []
        start = 0
        for k in range(clients_per_corruption):
            extra = 1 if k < remainder else 0
            end = start + samples_per + extra
            client_indices.append(perm[start:end].tolist())
            start = end
        return client_indices


def _fix_empty_clients(client_indices, rng, log_warning=True):
    """
    Redistribute samples from largest clients to empty ones.

    For each empty client, re-scans all clients to find the current
    largest donor with > 1 sample, then splits that donor.
    """
    empty = [k for k, idxs in enumerate(client_indices) if len(idxs) == 0]

    if not empty:
        return client_indices

    if log_warning:
        print(f"  WARNING: {len(empty)} empty clients found after Dirichlet split. "
              f"Redistributing from largest clients.")

    for k in empty:
        # Find the current best donor (recalc each iteration)
        donors = [
            (i, list(idxs))
            for i, idxs in enumerate(client_indices)
            if len(idxs) > 1
        ]
        donors.sort(key=lambda x: len(x[1]), reverse=True)

        if not donors:
            print(f"  WARNING: Cannot fix empty client {k}: "
                  f"no donor client has more than 1 sample.")
            continue

        largest_k, largest_idxs = donors[0]
        rng.shuffle(largest_idxs)

        split = max(1, len(largest_idxs) // 2)
        taken = list(largest_idxs[:split])
        kept = list(largest_idxs[split:])

        client_indices[largest_k] = kept
        client_indices[k] = taken

    return client_indices


def build_single_corruption_target_clients(
    dataset_name: str = "cifar10",
    root: str = None,
    corruptions: list = None,
    severity: int = 5,
    clients_per_corruption: int = 1,
    samples_per_client: int = None,
    batch_size: int = 64,
    label_skew: bool = False,
    alpha: float = None,
    precomputed_c_path: str = None,
    seed: int = 0,
):
    """
    Build single-corruption target clients for federated TTA.

    Each target client contains only ONE corruption type.
    Each corruption type can be split into multiple clients.

    Args:
        dataset_name: "cifar10" or "cifar100"
        root: root data path (used for online corruption)
        corruptions: list of corruption names. If None, use all 15 CIFAR-C corruptions.
        severity: corruption severity (1-5)
        clients_per_corruption: number of clients per corruption type
        samples_per_client: max samples per client. If None, use all samples.
        batch_size: DataLoader batch size
        label_skew: if True, apply Dirichlet label skew within each corruption
        alpha: Dirichlet alpha for label skew (required if label_skew=True)
        precomputed_c_path: path to pre-computed CIFAR-C .npy files
        seed: random seed

    Returns:
        target_clients: list of dicts, each containing:
            - client_id: int
            - corruption: str
            - domain_id: int (unique per corruption)
            - test_loader: DataLoader
            - num_samples: int
            - labels: list of int (ground truth labels)
    """
    if corruptions is None:
        corruptions = STANDARD_CORRUPTIONS

    cfg = DATASET_CONFIGS[dataset_name]
    num_classes = cfg["num_classes"]

    # Create a mapping corruption -> domain_id
    all_unique_corruptions = sorted(set(corruptions))
    domain_map = {corr: idx for idx, corr in enumerate(all_unique_corruptions)}

    class MockConfigs:
        """Minimal configs object for load_corrupted_test compatibility."""
        def __init__(self, name, data_path, c_path):
            self.dataset = name
            self.dataset_path = data_path
            self.precomputed_c_path = c_path

    target_clients = []
    client_id_counter = 0
    rng = np.random.RandomState(seed)

    for corr_idx, corruption in enumerate(corruptions):
        mock_configs = MockConfigs(
            name=dataset_name,
            data_path=root,
            c_path=precomputed_c_path,
        )

        # Load the full corrupted test dataset for this corruption
        full_dataset = load_corrupted_test(mock_configs, corruption, severity)

        if samples_per_client is not None and samples_per_client < len(full_dataset):
            # Subsample: select samples_per_client * clients_per_corruption total samples
            total_needed = samples_per_client * clients_per_corruption
            total_available = len(full_dataset)
            total_to_use = min(total_needed, total_available)

            # Get labels for partition (fast path)
            all_labels = _extract_labels_fast(full_dataset, total_available)

            all_indices = list(range(total_available))
            rng.shuffle(all_indices)
            selected_indices = all_indices[:total_to_use]
            selected_labels = all_labels[selected_indices]

            # Assign to clients
            client_indices_mapping = _assign_samples_to_clients(
                total_samples=total_to_use,
                clients_per_corruption=clients_per_corruption,
                label_skew=label_skew,
                num_classes=num_classes,
                labels=selected_labels,
                alpha=alpha or 1.0,
                seed=seed + corr_idx,
            )

            # Fix empty clients if any
            client_indices_mapping = _fix_empty_clients(client_indices_mapping, rng)

            for k in range(clients_per_corruption):
                indices = [selected_indices[i] for i in client_indices_mapping[k]]
                subset = Subset(full_dataset, indices)

                target_clients.append({
                    "client_id": client_id_counter,
                    "corruption": corruption,
                    "domain_id": domain_map[corruption],
                    "test_loader": DataLoader(
                        subset,
                        batch_size=batch_size,
                        shuffle=False,
                        num_workers=0,
                    ),
                    "num_samples": len(indices),
                })
                client_id_counter += 1
                print(f"  Target client {client_id_counter - 1}: corruption={corruption}, "
                      f"domain_id={domain_map[corruption]}, samples={len(indices)}")
        else:
            # Use all samples
            total_available = len(full_dataset)
            all_labels = _extract_labels_fast(full_dataset, total_available)

            client_indices_mapping = _assign_samples_to_clients(
                total_samples=total_available,
                clients_per_corruption=clients_per_corruption,
                label_skew=label_skew,
                num_classes=num_classes,
                labels=all_labels,
                alpha=alpha or 1.0,
                seed=seed + corr_idx,
            )

            # Fix empty clients if any
            client_indices_mapping = _fix_empty_clients(client_indices_mapping, rng)

            for k in range(clients_per_corruption):
                subset = Subset(full_dataset, client_indices_mapping[k])

                target_clients.append({
                    "client_id": client_id_counter,
                    "corruption": corruption,
                    "domain_id": domain_map[corruption],
                    "test_loader": DataLoader(
                        subset,
                        batch_size=batch_size,
                        shuffle=False,
                        num_workers=0,
                    ),
                    "num_samples": len(client_indices_mapping[k]),
                })
                client_id_counter += 1
                print(f"  Target client {client_id_counter - 1}: corruption={corruption}, "
                      f"domain_id={domain_map[corruption]}, samples={len(client_indices_mapping[k])}")

    return target_clients
