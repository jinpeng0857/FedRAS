"""
Fisher Information computation for EATA-C TTA.
Based on ACG-C compute_fisher.py and EATA-C paper Eqn.(7,9).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from typing import Dict, Optional


def compute_fisher_information(
    model: nn.Module,
    id_samples: torch.Tensor,
    batch_size: int = 64,
    device: Optional[torch.device] = None,
) -> Dict[str, torch.Tensor]:
    """
    Compute diagonal Fisher information for normalization layer params.
    Eqn.(9): ω(θ_i) = (1/Q) Σ (∂L_CE/∂θ_i)^2
    Uses pseudo-labels from the original model.

    Args:
        model: trained model (f_Θo)
        id_samples: (Q, C, H, W) unlabeled ID test samples
        batch_size: processing batch size
        device: computation device

    Returns:
        fisher: dict of {param_name: fisher_values}
    """
    if device is not None:
        model = model.to(device)
    model.eval()

    # Identify normalization layer parameter names
    norm_param_names = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d,
                                nn.LayerNorm, nn.GroupNorm)):
            for pname in ("weight", "bias"):
                full_name = f"{name}.{pname}"
                if any(p[0] == full_name for p in model.named_parameters()):
                    norm_param_names.append(full_name)

    # Initialize Fisher accumulator for norm param names
    fisher = OrderedDict()
    for name in norm_param_names:
        fisher[name] = 0.0

    total_samples = id_samples.size(0)
    num_batches = (total_samples + batch_size - 1) // batch_size

    for i in range(num_batches):
        start = i * batch_size
        end = min(start + batch_size, total_samples)
        batch = id_samples[start:end]
        if device is not None:
            batch = batch.to(device)

        # Per-sample gradient computation for diagonal Fisher
        # Eqn.(9): ω(θ_i) = (1/Q) Σ_q (∂L_CE(x_q; θ)/∂θ_i)^2
        # Using per-sample gradients (not batch average gradient squared)
        for j in range(batch.size(0)):
            sample = batch[j:j+1]
            result = model(sample)
            logits = result["logit"] if isinstance(result, dict) else result
            pseudo_label = torch.argmax(logits.detach(), dim=1)
            # Use reduction="sum" so loss = CE for a single sample (gradient w.r.t. one sample)
            loss = F.cross_entropy(logits, pseudo_label, reduction="sum")
            model.zero_grad()
            loss.backward()

            for name, param in model.named_parameters():
                if name in norm_param_names and param.grad is not None:
                    fisher[name] += (param.grad.detach().clone() ** 2).cpu()

    # Average over total samples
    for name in fisher:
        fisher[name] = fisher[name] / total_samples

    return fisher


def collect_original_params(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Collect the original values of normalization layer parameters."""
    orig_params = OrderedDict()
    for name, module in model.named_modules():
        if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d,
                                nn.LayerNorm, nn.GroupNorm)):
            for pname in ("weight", "bias"):
                full_name = f"{name}.{pname}"
                if hasattr(module, pname):
                    param = getattr(module, pname)
                    if param is not None:
                        orig_params[full_name] = param.data.clone().detach()
    return orig_params


def fisher_regularization(
    model: nn.Module,
    original_params: Dict[str, torch.Tensor],
    fisher_values: Dict[str, torch.Tensor],
) -> torch.Tensor:
    """
    Eqn.(7): R(Θ̃, Θ̃^o) = Σ ω(θ_i) (θ_i - θ_i^o)^2
    """
    reg_loss = 0.0
    for name, param in model.named_parameters():
        if name in original_params and name in fisher_values:
            orig = original_params[name].to(param.device)
            fisher = fisher_values[name].to(param.device)
            diff_sq = (param - orig) ** 2
            reg_loss += (fisher * diff_sq).sum()
    return reg_loss
