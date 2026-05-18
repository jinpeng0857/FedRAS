"""
Adapter utilities for Federated Test-Time Adaptation.

Provides functions to extract, load, and manipulate only the
normalization affine parameters (BN/LN/GN weight & bias) that
are updated during TTA.

This enables communication-efficient FedTTA: only adapt_state
(not full model) is exchanged between server and clients.
"""

from collections import OrderedDict
from typing import Dict, List, Optional

import torch
import torch.nn as nn


def extract_adapt_state(model: nn.Module) -> OrderedDict:
    """
    Extract normalization affine parameters from a model.

    Returns:
        OrderedDict mapping param_name -> param_data (detached, CPU)
        Only contains norm affine params: {name}.weight, {name}.bias
        where name is a BatchNorm/LayerNorm/GroupNorm/InstanceNorm module.
    """
    adapt_state = OrderedDict()
    for name, module in model.named_modules():
        if isinstance(module, (
            nn.BatchNorm1d,
            nn.BatchNorm2d,
            nn.BatchNorm3d,
            nn.LayerNorm,
            nn.GroupNorm,
            nn.InstanceNorm1d,
            nn.InstanceNorm2d,
            nn.InstanceNorm3d,
        )):
            for pname in ("weight", "bias"):
                param = getattr(module, pname, None)
                if param is not None:
                    full_name = f"{name}.{pname}" if name else pname
                    adapt_state[full_name] = param.data.detach().cpu().clone()
    return adapt_state


def load_adapt_state(model: nn.Module, adapt_state: Dict[str, torch.Tensor]):
    """
    Load adapt state into model's normalization affine parameters.

    Args:
        model: model to load into (modified in-place)
        adapt_state: dict of {param_name: tensor} from extract_adapt_state
    """
    model_state = model.state_dict()
    for name, tensor in adapt_state.items():
        if name in model_state:
            model_state[name].copy_(tensor)
        # Also try matching via named_parameters for safety
    # Use _load_from_state_dict for strict matching
    missing, unexpected = model.load_state_dict(model_state, strict=False)
    # Ignore non-adapt params (they should not be in adapt_state)


def state_delta(
    new_state: Dict[str, torch.Tensor],
    old_state: Dict[str, torch.Tensor],
) -> OrderedDict:
    """
    Compute delta = new_state - old_state for each key in new_state.

    Args:
        new_state: post-adaptation adapter state
        old_state: pre-adaptation adapter state

    Returns:
        OrderedDict of {key: delta_tensor} (on CPU)
    """
    delta = OrderedDict()
    for k in new_state:
        if k in old_state:
            delta[k] = new_state[k].cpu() - old_state[k].cpu()
        else:
            # Key only in new_state -> use new_state value as delta
            delta[k] = new_state[k].cpu().clone()
    return delta


def apply_state_delta(
    state: Dict[str, torch.Tensor],
    delta: Dict[str, torch.Tensor],
    scale: float = 1.0,
) -> OrderedDict:
    """
    Apply delta to a state: new_state[k] = state[k] + scale * delta[k].

    Args:
        state: base state dict
        delta: delta to apply
        scale: scaling factor for delta

    Returns:
        new OrderedDict
    """
    new_state = OrderedDict()
    for k in state:
        if k in delta:
            new_state[k] = state[k].cpu() + scale * delta[k].cpu()
        else:
            new_state[k] = state[k].cpu().clone()
    return new_state


def weighted_average_delta(
    deltas: List[Dict[str, torch.Tensor]],
    weights: List[float],
) -> OrderedDict:
    """
    Weighted average of multiple deltas.

    Args:
        deltas: list of delta dicts from multiple clients
        weights: list of weights (same length as deltas)

    Returns:
        weighted average delta as OrderedDict
    """
    if not deltas:
        return OrderedDict()

    total_weight = sum(weights)
    if total_weight <= 0:
        total_weight = 1.0
        weights = [1.0 / len(deltas)] * len(deltas)

    norm_weights = [w / total_weight for w in weights]

    avg_delta = OrderedDict()
    for k in deltas[0]:
        avg_delta[k] = torch.zeros_like(deltas[0][k])
        for i, d in enumerate(deltas):
            avg_delta[k] = avg_delta[k] + d[k] * norm_weights[i]

    return avg_delta


def zeros_like_state(state: Dict[str, torch.Tensor]) -> OrderedDict:
    """
    Create a zero-initialized copy of a state dict.

    Used for initializing momentum buffers.

    Args:
        state: reference state dict

    Returns:
        OrderedDict with same keys, zero tensors
    """
    return OrderedDict(
        {k: torch.zeros_like(v) for k, v in state.items()}
    )


def count_adapt_params(state: Dict[str, torch.Tensor]) -> int:
    """
    Count total number of adapter parameters.

    Args:
        state: adapter state dict

    Returns:
        total parameter count
    """
    return sum(v.numel() for v in state.values())


def add_state(
    state_a: Dict[str, torch.Tensor],
    state_b: Dict[str, torch.Tensor],
    scale: float = 1.0,
) -> OrderedDict:
    """
    Element-wise: result[k] = state_a[k] + scale * state_b[k].

    Args:
        state_a: base state dict
        state_b: delta state dict
        scale: scaling factor for state_b

    Returns:
        new OrderedDict
    """
    new_state = OrderedDict()
    for k in state_a:
        if k in state_b:
            new_state[k] = state_a[k].cpu() + scale * state_b[k].cpu()
        else:
            new_state[k] = state_a[k].cpu().clone()
    return new_state


def scale_state(
    state: Dict[str, torch.Tensor],
    scale: float,
) -> OrderedDict:
    """
    Element-wise: result[k] = scale * state[k].

    Args:
        state: input state dict
        scale: scaling factor

    Returns:
        new OrderedDict
    """
    return OrderedDict(
        {k: v.cpu() * scale for k, v in state.items()}
    )


def adapt_state_to_device(
    state: Dict[str, torch.Tensor],
    device: torch.device,
) -> OrderedDict:
    """
    Move adapt state to device.

    Args:
        state: adapter state dict (CPU tensors)
        device: target device

    Returns:
        OrderedDict with tensors on target device
    """
    return OrderedDict(
        {k: v.to(device) for k, v in state.items()}
    )
