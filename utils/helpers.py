"""Utility functions: seed, save paths, algorithm dispatch."""

import logging
import os
import random

import numpy as np
import torch

from train.fedacg.client import FedACGClient
from train.fedacg.server import FedACGServer
from train.fedavg.client import FedAvgClient
from train.fedavg.server import FedAvgServer
from train.fedprox.client import FedProxClient
from train.fedprox.server import FedProxServer
from train.fedadam import FedAdamClient, FedAdamServer
from train.feddyn import FedDynClient, FedDynServer
from train.scaffold import ScaffoldClient, ScaffoldServer


def set_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_save_path(configs):
    root_path = os.path.dirname(os.path.dirname(__file__))
    root_path = os.path.join(root_path, "train", "results")
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
    if configs.method == "fedavg":
        return FedAvgServer, FedAvgClient
    if configs.method == "fedprox":
        return FedProxServer, FedProxClient
    if configs.method == "scaffold":
        return ScaffoldServer, ScaffoldClient
    if configs.method == "feddyn":
        return FedDynServer, FedDynClient
    if configs.method == "fedadam":
        return FedAdamServer, FedAdamClient
    raise ValueError(
        "Unknown method: %s. Supported: fedacg, fedavg, fedprox, scaffold, "
        "feddyn, fedadam." % configs.method
    )


def select_checkpoint_state(checkpoint, ckpt_key="model_state_dict"):
    """Select a state_dict from a FedACG-compatible checkpoint object.

    ``ckpt_key="auto"`` prefers the FedACG lookahead anchor because it is the
    recommended initialization for test-time adaptation.
    """
    if not isinstance(checkpoint, dict):
        if ckpt_key not in ("auto", "state_dict"):
            raise KeyError(
                f"Requested ckpt_key={ckpt_key}, but checkpoint is not a dict"
            )
        return checkpoint, "state_dict"

    if ckpt_key == "auto":
        for candidate in (
            "lookahead_anchor",
            "adaptation_anchor",
            "model_state_dict",
            "state_dict",
            "model",
        ):
            if candidate in checkpoint:
                return checkpoint[candidate], candidate
        if "fc.weight" in checkpoint:
            return checkpoint, "state_dict"
        raise KeyError(
            "ckpt_key=auto could not find lookahead_anchor, adaptation_anchor, "
            "model_state_dict, state_dict, or model"
        )

    if ckpt_key == "state_dict" and "state_dict" not in checkpoint:
        if "fc.weight" in checkpoint:
            return checkpoint, "state_dict"
    if ckpt_key not in checkpoint:
        raise KeyError(
            f"ckpt_key={ckpt_key} not found in checkpoint keys={list(checkpoint.keys())}"
        )
    return checkpoint[ckpt_key], ckpt_key


def load_checkpoint_weights(model, model_path, device="cpu", ckpt_key="model_state_dict"):
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

    state_dict, used_key = select_checkpoint_state(checkpoint, ckpt_key=ckpt_key)
    _log.info(f"  Load: selected checkpoint key '{used_key}'")

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
