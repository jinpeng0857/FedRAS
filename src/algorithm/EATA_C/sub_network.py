"""
Sub-network generation for EATA-C.

Generates stochastic sub-network predictions using two strategies:
  1. MC Dropout: toggles Dropout layers to train mode (general-purpose)
  2. return_sub_network: uses model's built-in stochastic depth (e.g. EATACResNet)

The function auto-detects which strategy the model supports.
"""

from typing import Dict

import torch
import torch.nn as nn


# Global flag: when False, only MC Dropout is used (EATA-C alignment mode).
# When True, auto-detects and prefers return_sub_network / stochastic depth.
_USE_STOCHASTIC_DEPTH = True


def set_use_stochastic_depth(value: bool):
    """Set whether to prefer stochastic depth over MC Dropout."""
    global _USE_STOCHASTIC_DEPTH
    _USE_STOCHASTIC_DEPTH = value


def _try_inspect_forward(model: nn.Module) -> bool:
    """Check if model.forward accepts return_sub_network kwarg."""
    import inspect
    try:
        sig = inspect.signature(model.forward)
        return 'return_sub_network' in sig.parameters
    except (ValueError, TypeError):
        return False


def _get_subnetwork_prediction_with_flag(
    model: nn.Module, x: torch.Tensor
) -> torch.Tensor:
    """Use model's built-in stochastic depth via force_sub_network.

    Temporarily sets model.force_sub_network = True to bypass the
    (self.training or self.force_sub_network) guard in EATACResNet.forward(),
    allowing stochastic depth to activate even when the model is in eval()
    mode. The model's physical state (train/eval, BN running stats) is
    preserved — no side effects.
    """
    original = getattr(model, 'force_sub_network', False)
    model.force_sub_network = True
    try:
        logits = model(x, return_sub_network=True)
        result = logits["logit"] if isinstance(logits, dict) else logits
        return result
    finally:
        model.force_sub_network = original


def _get_subnetwork_prediction_mc_dropout(
    model: nn.Module, x: torch.Tensor
) -> torch.Tensor:
    """Use MC Dropout: enable dropout layers for stochastic forward pass."""
    original_states: Dict[nn.Module, bool] = {}
    for module in model.modules():
        if isinstance(module, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            original_states[module] = module.training
            module.train()

    logits = model(x)
    result = logits["logit"] if isinstance(logits, dict) else logits

    for module, state in original_states.items():
        module.train(state)

    return result


def get_subnetwork_prediction(
    model: nn.Module, x: torch.Tensor
) -> torch.Tensor:
    """
    Get sub-network prediction.

    When ``_USE_STOCHASTIC_DEPTH`` is True and the model supports it,
    uses the model's built-in stochastic depth (``return_sub_network``).
    Otherwise falls back to MC Dropout (enabling dropout layers).

    Args:
        model: the model to query
        x: input tensor

    Returns:
        sub_logits: prediction from a stochastic sub-network
    """
    if _USE_STOCHASTIC_DEPTH and _try_inspect_forward(model):
        return _get_subnetwork_prediction_with_flag(model, x)
    return _get_subnetwork_prediction_mc_dropout(model, x)
