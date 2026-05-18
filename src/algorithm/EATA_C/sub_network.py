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
    """Use model's built-in return_sub_network flag. Caller must set train() mode."""
    logits = model(x, return_sub_network=True)
    result = logits["logit"] if isinstance(logits, dict) else logits
    return result


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

    Auto-detects the model's sub-network mechanism:
      - If forward() accepts ``return_sub_network`` (e.g. EATACResNet),
        uses the model's built-in stochastic depth.
      - Otherwise falls back to MC Dropout (enabling dropout layers).

    Args:
        model: the model to query
        x: input tensor

    Returns:
        sub_logits: prediction from a stochastic sub-network
    """
    if _try_inspect_forward(model):
        return _get_subnetwork_prediction_with_flag(model, x)
    return _get_subnetwork_prediction_mc_dropout(model, x)
