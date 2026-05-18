"""Utility functions: seed, save paths, algorithm dispatch."""

import logging
import os
import random

import numpy as np
import torch

from src.algorithm.FedACG.client_fedacg import FedACGClient
from src.algorithm.FedACG.server_fedacg import FedACGServer

try:
    from src.algorithm.FedAvg.client_fedavg import FedAvgClient
    from src.algorithm.FedAvg.server_fedavg import FedAvgServer
except ImportError:
    # FedAvg folder may be deleted (not needed for FedACG)
    FedAvgClient = None
    FedAvgServer = None


def set_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_save_path(configs):
    root_path = os.path.dirname(os.path.dirname(__file__))
    root_path = os.path.join(root_path, "results")
    equal_tag = "_equal" if getattr(configs, "equal_size", False) else ""
    dataset_setting = (
        f"{configs.dataset}_{configs.num_client}clients_alpha{configs.alpha}{equal_tag}"
    )
    task_path = (
        f"{dataset_setting}/{configs.method}_{configs.backbone}/{configs.task_name}"
    )
    save_path = os.path.join(root_path, task_path)
    os.makedirs(save_path, exist_ok=True)
    return save_path


def prepare_server_and_clients(configs):
    """Map configs.method to (ServerClass, ClientClass)."""
    if configs.method == "fedacg":
        return FedACGServer, FedACGClient
    elif configs.method == "fedavg":
        return FedAvgServer, FedAvgClient
    else:
        raise ValueError(f"Unknown method: {configs.method}")


def compute_ece(probs, labels, n_bins=15):
    """Compute Expected Calibration Error."""
    confidences, predictions = probs.max(dim=1)
    correct = (predictions == labels).float()
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (
            confidences <= bin_boundaries[i + 1]
        )
        prop = in_bin.float().mean()
        if prop > 0:
            acc_bin = correct[in_bin].mean()
            conf_bin = confidences[in_bin].mean()
            ece += torch.abs(acc_bin - conf_bin) * prop
    return ece.item()


def compute_accuracy(logits, labels):
    pred = logits.argmax(dim=1)
    return (pred == labels).float().mean().item()


def load_checkpoint_weights(model, model_path, device="cpu"):
    """
    Universal checkpoint loader. Compatible with:
      - ACG-C format:  {'epoch': N, 'model_state_dict': dict}
      - Torch save:    {'epoch': N, 'state_dict': dict}
      - Raw state_dict:  dict without wrapper keys
      - TorchScript:    torch.jit.load
    """
    _log = logging.getLogger(__name__)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    # Strip DataParallel 'module.' prefix
    if isinstance(state_dict, dict) and any(k.startswith("module.") for k in state_dict):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    missing, unexpected = model.load_state_dict(state_dict, strict=True)

    # Promote fc layer mismatch to error-level visibility
    fc_keys = {"fc.weight", "fc.bias", "module.fc.weight", "module.fc.bias"}
    fc_missing = [k for k in missing if k in fc_keys]
    fc_unexpected = [k for k in unexpected if k in fc_keys]
    if fc_missing or fc_unexpected:
        _log.warning(
            f"  fc layer mismatch! missing={fc_missing}, unexpected={fc_unexpected}. "
            f"This usually means --num_class is wrong (checkpoint vs config)."
        )
        if fc_missing:
            _log.warning(f"  Missing fc keys → checkpoint has FEWER classes than model ({len(state_dict.get('fc.weight', []))})")
        if fc_unexpected:
            ckpt_fc = state_dict.get('fc.weight', state_dict.get('module.fc.weight'))
            ckpt_classes = ckpt_fc.shape[0] if ckpt_fc is not None else '?'
            _log.warning(f"  Unexpected fc keys → checkpoint has MORE classes ({ckpt_classes}) than model")

    if missing and not fc_missing:
        _log.info(f"  Load: {len(missing)} missing keys (e.g., {missing[:3]})")
    if unexpected and not fc_unexpected:
        _log.info(f"  Load: {len(unexpected)} unexpected keys (e.g., {unexpected[:3]})")
    if not missing and not unexpected:
        _log.info("  Load: all keys matched perfectly")
