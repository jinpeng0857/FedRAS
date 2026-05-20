"""
Fisher Information computation for EATA-C TTA.
Based on ACG-C compute_fisher.py and EATA-C paper Eqn.(7,9).

Optimized with:
  - Pre-collected norm param references (no repeated named_parameters())
  - Batched pseudo-label computation
  - zero_grad(set_to_none=True) for faster clearing
  - Optional vmap-based vectorization for PyTorch >= 2.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple


def _collect_norm_params(model: nn.Module) -> List[Tuple[str, nn.Parameter]]:
    """Collect (name, param) pairs for all normalization layer affine parameters."""
    norm_param_names = set()
    for name, module in model.named_modules():
        if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d,
                               nn.LayerNorm, nn.GroupNorm,
                               nn.InstanceNorm2d, nn.InstanceNorm1d)):
            for pname in ("weight", "bias"):
                full_name = f"{name}.{pname}"
                norm_param_names.add(full_name)

    result = []
    for n, p in model.named_parameters():
        if n in norm_param_names:
            result.append((n, p))
    return result


def _compute_fisher_loop(
    model: nn.Module,
    id_samples: torch.Tensor,
    norm_params: List[Tuple[str, nn.Parameter]],
    batch_size: int,
    device: Optional[torch.device],
) -> Dict[str, torch.Tensor]:
    """
    Per-sample gradient loop (PyTorch < 2.0 fallback).
    Works with all PyTorch versions.
    """
    total_samples = id_samples.size(0)
    fisher = OrderedDict()
    for name, _ in norm_params:
        fisher[name] = 0.0

    for i in range(0, total_samples, batch_size):
        batch = id_samples[i:i + batch_size]
        if device is not None:
            batch = batch.to(device)

        # Single forward pass for pseudo-labels (entire batch at once)
        with torch.no_grad():
            result = model(batch)
            logits = result["logit"] if isinstance(result, dict) else result
            pseudo_labels = logits.argmax(dim=1)

        # Per-sample gradient accumulation
        for j in range(batch.size(0)):
            sample = batch[j:j + 1]
            label = pseudo_labels[j:j + 1]

            model.zero_grad(set_to_none=True)
            result = model(sample)
            logits_j = result["logit"] if isinstance(result, dict) else result
            loss = F.cross_entropy(logits_j, label, reduction="sum")
            loss.backward()

            for name, param in norm_params:
                if param.grad is not None:
                    fisher[name] += (param.grad.detach() ** 2).cpu()

    for name in fisher:
        fisher[name] = fisher[name] / total_samples

    return fisher


def _compute_fisher_vmap(
    model: nn.Module,
    id_samples: torch.Tensor,
    norm_params: List[Tuple[str, nn.Parameter]],
    batch_size: int,
    device: Optional[torch.device],
) -> Dict[str, torch.Tensor]:
    """
    Vectorized per-sample gradient computation via torch.func.vmap.
    Requires PyTorch >= 2.0.
    """
    from torch.func import functional_call, grad, vmap

    model.eval()
    if device is not None:
        model = model.to(device)

    norm_param_names = [n for n, _ in norm_params]

    # Extract functional params/buffers (detached copies, no grad tracking)
    params = {k: v.detach() for k, v in model.named_parameters()}
    buffers = {k: v.detach() for k, v in model.named_buffers()}

    total_samples = id_samples.size(0)
    fisher = OrderedDict()
    for name in norm_param_names:
        fisher[name] = torch.zeros(params[name].shape)

    def loss_fn(p, b, x, target):
        result = functional_call(model, (p, b), x.unsqueeze(0))
        logits = result["logit"] if isinstance(result, dict) else result
        return F.cross_entropy(logits, target.unsqueeze(0), reduction="sum")

    def sample_grad(p, b, x, target):
        grads = grad(loss_fn)(p, b, x, target)
        return tuple(grads.get(n, torch.zeros_like(params[n])) for n in norm_param_names)

    batch_vmap = vmap(sample_grad, in_dims=(None, None, 0, 0), randomness="different")

    for i in range(0, total_samples, batch_size):
        batch = id_samples[i:i + batch_size]
        if device is not None:
            batch = batch.to(device)

        # Pseudo-labels from full model (single forward)
        with torch.no_grad():
            result = functional_call(model, (params, buffers), batch)
            logits = result["logit"] if isinstance(result, dict) else result
            labels = logits.argmax(dim=1)

        # Vectorized per-sample gradients
        grads_tuple = batch_vmap(params, buffers, batch, labels)

        for name, grad_t in zip(norm_param_names, grads_tuple):
            if grad_t is not None:
                fisher[name] = fisher[name].to(grad_t.device)
                fisher[name] += (grad_t ** 2).sum(dim=0).cpu()

    for name in fisher:
        fisher[name] = fisher[name] / total_samples

    return fisher


def compute_fisher_information(
    model: nn.Module,
    id_samples: torch.Tensor,
    batch_size: int = 64,
    device: Optional[torch.device] = None,
) -> Dict[str, torch.Tensor]:
    """
    Compute diagonal Fisher information for normalization layer params.

    Eqn.(9): omega(theta_i) = (1/Q) * sum_q (dL_CE(x_q; theta) / dtheta_i)^2
    Uses pseudo-labels from the original model.

    Auto-selects between:
      - vmap vectorized (PyTorch >= 2.0, ~10x faster on GPU)
      - per-sample loop (fallback, works with all PyTorch versions)

    Args:
        model: trained model (f_Theta_o)
        id_samples: (Q, C, H, W) unlabeled ID test samples
        batch_size: processing batch size
        device: computation device

    Returns:
        fisher: dict of {param_name: fisher_values}
    """
    if device is not None:
        model = model.to(device)
    model.eval()

    # Collect norm param references once (avoids repeated named_parameters())
    norm_params = _collect_norm_params(model)
    if not norm_params:
        return OrderedDict()

    # Try vmap vectorized path first (PyTorch >= 2.0)
    use_vmap = False
    if device is not None and device.type == "cuda":
        try:
            import torch.func
            use_vmap = True
        except (ImportError, AttributeError):
            pass

    if use_vmap:
        try:
            return _compute_fisher_vmap(
                model, id_samples, norm_params, batch_size, device
            )
        except (RuntimeError, TypeError) as e:
            print(f"  vmap Fisher failed ({e}), falling back to per-sample loop.")
            pass

    return _compute_fisher_loop(
        model, id_samples, norm_params, batch_size, device
    )


def collect_original_params(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Collect the original values of normalization layer parameters."""
    orig_params = OrderedDict()
    for name, module in model.named_modules():
        if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d,
                                nn.LayerNorm, nn.GroupNorm,
                                nn.InstanceNorm2d, nn.InstanceNorm1d)):
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
    Eqn.(7): R(Theta_tilde, Theta_tilde^o) = Sum omega(theta_i) * (theta_i - theta_i^o)^2
    """
    reg_loss = 0.0
    for name, param in model.named_parameters():
        if name in original_params and name in fisher_values:
            orig = original_params[name].to(param.device)
            fisher = fisher_values[name].to(param.device)
            diff_sq = (param - orig) ** 2
            reg_loss += (fisher * diff_sq).sum()
    return reg_loss
