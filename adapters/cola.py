"""CoLA principal-agent adapter for the unified FEDEATA protocol.

The upstream CoLA repository targets ViT and replaces LayerNorm with a
domain-vector fusion layer.  The paper also specifies the corresponding
ResNet formulation: store affine BatchNorm deltas and reprogram them with a
softmax over the accumulated domain vectors.  This module implements that
formulation for the ResNet checkpoints used by FedACG.

The adaptation objective is the official ETA/EATA entropy objective used by
the CoLA paper.  CoLA itself only changes the parameterization of the
normalization affine tensors and the domain-vector memory; it does not use
labels or the FedRAS/DSR restoration machinery.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from adapters.base import DisabledSparseRestorer
from tests.tta_baselines import _ExternalAdapterBase, softmax_entropy


class CoLAState(nn.Module):
    """Shared CoLA state and domain-vector pool.

    ``domain_vectors[0]`` is the null vector from Eq. (2) in the paper.  Each
    later item maps a BatchNorm module name to its affine delta from the
    deployment anchor.  The BatchNorm modules keep a non-registered reference
    to this object, so expanding the pool replaces the alpha parameter once
    and all modules see the new parameter immediately.
    """

    def __init__(
        self,
        stem: nn.Module,
        *,
        weight_margin: float = 0.01,
        shift_detection_threshold: float = 0.1,
        domain_ema: float = 0.2,
        max_num_vectors: Optional[int] = None,
    ):
        super().__init__()
        if weight_margin <= 0:
            raise ValueError("CoLA weight_margin must be positive")
        if shift_detection_threshold < 0:
            raise ValueError("CoLA shift_detection_threshold must be non-negative")
        if not 0 < domain_ema <= 1:
            raise ValueError("CoLA domain_ema must be in (0, 1]")
        if max_num_vectors is not None and max_num_vectors <= 0:
            raise ValueError("CoLA max_num_vectors must be positive or None")

        device = next(stem.parameters()).device
        self.alpha = nn.Parameter(torch.zeros(1, device=device))
        self.alpha_scale = nn.Parameter(torch.ones(1, device=device))
        self.weight_margin = float(weight_margin)
        self.shift_detection_threshold = float(shift_detection_threshold)
        self.domain_ema = float(domain_ema)
        self.max_num_vectors = (
            None if max_num_vectors is None else int(max_num_vectors)
        )

        # Do not register the stem as a child of the state.  The root model
        # already owns it, and registering it here would create a module cycle.
        object.__setattr__(self, "_stem", stem)
        object.__setattr__(self, "_modules_by_name", {})
        object.__setattr__(self, "_optimizer", None)

        self.domain_vectors: List[Dict[str, Tuple[torch.Tensor, torch.Tensor]]] = []
        self.domain_vectors.append({})
        self.domain_mean: Optional[torch.Tensor] = None
        self.domain_var: Optional[torch.Tensor] = None
        self.domain_infos: List[Tuple[torch.Tensor, torch.Tensor]] = []
        self.shift_events = 0
        self.saved_vectors = 0
        self.current_domain_batches = 0
        self.dirty = False
        self.last_shift_distance = 0.0

    @property
    def vector_count(self) -> int:
        return len(self.domain_vectors)

    def set_modules(self, modules: Mapping[str, "CoLABatchNorm2d"]):
        if not modules:
            raise RuntimeError("CoLA requires at least one BatchNorm module")
        object.__setattr__(self, "_modules_by_name", dict(modules))
        self.domain_vectors[0] = {
            name: (
                module.origin_weight.detach().clone(),
                module.origin_bias.detach().clone(),
            )
            for name, module in modules.items()
        }
        # The null vector is defined as zero relative to the deployment anchor.
        self.domain_vectors[0] = {
            name: (
                torch.zeros_like(weight),
                torch.zeros_like(bias),
            )
            for name, (weight, bias) in self.domain_vectors[0].items()
        }

    def set_optimizer(self, optimizer: "CoLAOptimizer"):
        object.__setattr__(self, "_optimizer", optimizer)

    def coefficients(self) -> torch.Tensor:
        return F.softmax(self.alpha * self.alpha_scale, dim=0)

    @torch.no_grad()
    def fused_affine(self, name: str):
        module = self._modules_by_name[name]
        coefficients = self.coefficients()
        weight = module.origin_weight + module.weight
        bias = module.origin_bias + module.bias
        for coefficient, vector in zip(coefficients, self.domain_vectors):
            if name in vector:
                delta_weight, delta_bias = vector[name]
                weight = weight + coefficient * delta_weight
                bias = bias + coefficient * delta_bias
        return weight, bias

    def differentiable_fused_affine(self, name: str):
        """Return the fused affine tensors while preserving gradients."""
        module = self._modules_by_name[name]
        coefficients = self.coefficients()
        weight = module.origin_weight + module.weight
        bias = module.origin_bias + module.bias
        for coefficient, vector in zip(coefficients, self.domain_vectors):
            if name in vector:
                delta_weight, delta_bias = vector[name]
                weight = weight + coefficient * delta_weight
                bias = bias + coefficient * delta_bias
        return weight, bias

    @torch.no_grad()
    def _stem_statistics(self, images: torch.Tensor):
        features = self._stem(images)
        if features.ndim == 4:
            dimensions = (0, 2, 3)
        elif features.ndim == 3:
            dimensions = (0, 1)
        elif features.ndim == 2:
            dimensions = (0,)
        else:
            raise ValueError(
                f"unsupported CoLA stem feature shape: {tuple(features.shape)}"
            )
        variance, mean = torch.var_mean(
            features.float(), dim=dimensions, unbiased=False
        )
        return variance.clamp_min(1e-6), mean

    @staticmethod
    @torch.no_grad()
    def _symmetric_gaussian_distance(
        first_var: torch.Tensor,
        first_mean: torch.Tensor,
        second_var: torch.Tensor,
        second_mean: torch.Tensor,
    ) -> torch.Tensor:
        first_var = first_var.clamp_min(1e-6)
        second_var = second_var.clamp_min(1e-6)
        squared_mean_gap = (first_mean - second_mean).square()
        first_to_second = (
            first_var + squared_mean_gap
        ) / (2.0 * second_var) - 0.5
        second_to_first = (
            second_var + squared_mean_gap
        ) / (2.0 * first_var) - 0.5
        return (first_to_second + second_to_first).mean()

    @torch.no_grad()
    def observe(self, images: torch.Tensor):
        """Update domain statistics and save a vector on a detected shift."""
        current_var, current_mean = self._stem_statistics(images)
        shift = False
        distance = 0.0
        if self.domain_var is not None and self.domain_mean is not None:
            distance = float(
                self._symmetric_gaussian_distance(
                    self.domain_var,
                    self.domain_mean,
                    current_var,
                    current_mean,
                ).cpu()
            )
            if distance > self.shift_detection_threshold:
                self.save_current_weight()
                self.domain_infos.append(
                    (self.domain_var.detach().clone(), self.domain_mean.detach().clone())
                )
                self.domain_mean = None
                self.domain_var = None
                self.shift_events += 1
                shift = True

        if self.domain_var is None or self.domain_mean is None:
            self.domain_var = current_var.detach().clone()
            self.domain_mean = current_mean.detach().clone()
        else:
            self.domain_var = (
                self.domain_ema * current_var
                + (1.0 - self.domain_ema) * self.domain_var
            )
            self.domain_mean = (
                self.domain_ema * current_mean
                + (1.0 - self.domain_ema) * self.domain_mean
            )
        self.current_domain_batches += 1
        self.last_shift_distance = distance
        return shift, distance

    @torch.no_grad()
    def save_current_weight(self):
        """Append the current fused affine tensors as a domain vector."""
        if not self._modules_by_name:
            return False
        if self.max_num_vectors is not None and self.vector_count >= self.max_num_vectors:
            # Preserve the source/null vector and discard the oldest learned
            # vector.  The matching alpha logit must be removed as well;
            # otherwise the softmax denominator grows without a corresponding
            # vector and silently attenuates every fused affine tensor.
            if self.vector_count > 1:
                fused_before_removal = {
                    name: self.fused_affine(name)
                    for name in self._modules_by_name
                }
                self.domain_vectors.pop(1)
                if self.alpha.numel() > 1:
                    old_alpha = self.alpha.detach()
                    self.alpha = nn.Parameter(
                        torch.cat((old_alpha[:1], old_alpha[2:]))
                    )
                for name, module in self._modules_by_name.items():
                    fused_after_removal = self.fused_affine(name)
                    module.weight.add_(
                        fused_before_removal[name][0] - fused_after_removal[0]
                    )
                    module.bias.add_(
                        fused_before_removal[name][1] - fused_after_removal[1]
                    )

        previous_fused = {
            name: self.fused_affine(name)
            for name in self._modules_by_name
        }
        new_vector = {
            name: (
                (weight - self._modules_by_name[name].origin_weight).detach().clone(),
                (bias - self._modules_by_name[name].origin_bias).detach().clone(),
            )
            for name, (weight, bias) in previous_fused.items()
        }
        self.domain_vectors.append(new_vector)

        current_logit = self.alpha.detach() * self.alpha_scale.detach()
        max_epsilon = max(
            max(
                float(module.weight.detach().abs().mean().cpu()),
                float(module.bias.detach().abs().mean().cpu()),
            )
            for module in self._modules_by_name.values()
        )
        scale_factor = max(1.0, max_epsilon / self.weight_margin)
        new_logit = float("-inf")
        if scale_factor > 1.0:
            new_logit = float(
                torch.log(
                    torch.exp(current_logit).sum()
                    * (scale_factor - 1.0)
                ).cpu()
            )
        new_logit = max(float(current_logit.max().cpu()), new_logit)
        scale = max(float(self.alpha_scale.detach().cpu()), 1e-6)
        new_alpha = torch.cat(
            [self.alpha.detach(), self.alpha.detach().new_tensor([new_logit / scale])]
        ).clamp(-10.0, 10.0)
        self.alpha = nn.Parameter(new_alpha)

        # Expanding the pool changes the softmax denominator.  Offset the
        # current epsilon so that saving a vector is functionally lossless.
        for name, module in self._modules_by_name.items():
            new_fused = self.fused_affine(name)
            old_weight, old_bias = previous_fused[name]
            module.weight.add_(old_weight - new_fused[0])
            module.bias.add_(old_bias - new_fused[1])

        self.saved_vectors += 1
        self.dirty = False
        if self._optimizer is not None:
            self._optimizer.on_vector_appended()
        return True

    @torch.no_grad()
    def mark_updated(self):
        self.dirty = True

    @torch.no_grad()
    def actual_drift_stats(self):
        delta_square = 0.0
        anchor_square = 0.0
        gamma_delta = 0.0
        gamma_anchor = 0.0
        beta_delta = 0.0
        beta_anchor = 0.0
        for module in self._modules_by_name.values():
            current_weight, current_bias = self.fused_affine(module.cola_name)
            for current, anchor, group in (
                (current_weight, module.origin_weight, "gamma"),
                (current_bias, module.origin_bias, "beta"),
            ):
                delta = float((current - anchor).square().sum().cpu())
                anchor_value = float(anchor.square().sum().cpu())
                delta_square += delta
                anchor_square += anchor_value
                if group == "gamma":
                    gamma_delta += delta
                    gamma_anchor += anchor_value
                else:
                    beta_delta += delta
                    beta_anchor += anchor_value
        return {
            "total": math.sqrt(delta_square) / (math.sqrt(anchor_square) + 1e-12),
            "gamma": math.sqrt(gamma_delta) / (math.sqrt(gamma_anchor) + 1e-12),
            "beta": math.sqrt(beta_delta) / (math.sqrt(beta_anchor) + 1e-12),
        }


class CoLABatchNorm2d(nn.BatchNorm2d):
    """BatchNorm whose affine tensors are reprogrammed by CoLA."""

    def __init__(
        self,
        source: nn.BatchNorm2d,
        state: CoLAState,
        cola_name: str,
    ):
        if not source.affine:
            raise ValueError("CoLA requires affine BatchNorm layers")
        super().__init__(
            source.num_features,
            eps=source.eps,
            momentum=source.momentum,
            affine=True,
            track_running_stats=False,
        )
        self.to(device=source.weight.device, dtype=source.weight.dtype)
        self.register_buffer("origin_weight", source.weight.detach().clone())
        self.register_buffer("origin_bias", source.bias.detach().clone())
        with torch.no_grad():
            self.weight.zero_()
            self.bias.zero_()
        self.cola_name = str(cola_name)
        # Avoid registering the same state object under every BatchNorm child.
        object.__setattr__(self, "_cola_state", state)

    def forward(self, input: torch.Tensor):
        weight, bias = self._cola_state.differentiable_fused_affine(self.cola_name)
        # CoLA follows the TTA convention of forcing batch statistics.  This is
        # also what BatchNorm does when running_mean/running_var are None.
        return F.batch_norm(
            input,
            None,
            None,
            weight,
            bias,
            True,
            self.momentum,
            self.eps,
        )


class CoLAProxyRestorer(DisabledSparseRestorer):
    """Protocol-compatible drift tracker for CoLA's fused BN tensors."""

    def __init__(self, model: nn.Module):
        super().__init__(model)
        self.cola_state = model.cola_state

    @torch.no_grad()
    def drift_stats(self):
        return self.cola_state.actual_drift_stats()


class CoLAOptimizer:
    """Official ETA+CoLA optimizer split.

    The alpha logits and temperature use AdamW (lr=0.1, wd=0.1).  The current
    domain delta uses SGD with the integrated ETA settings and the CoLA paper's
    delta weight decay (0.4 for ETA in the lifelong experiment).
    """

    def __init__(
        self,
        state: CoLAState,
        epsilon_parameters: Iterable[nn.Parameter],
        *,
        epsilon_lr: float = 0.001,
        epsilon_momentum: float = 0.9,
        epsilon_weight_decay: float = 0.4,
        alpha_lr: float = 0.1,
        alpha_weight_decay: float = 0.1,
    ):
        self.state = state
        self.epsilon_parameters = list(epsilon_parameters)
        if not self.epsilon_parameters:
            raise RuntimeError("CoLA found no trainable BN affine parameters")
        self.epsilon_lr = float(epsilon_lr)
        self.epsilon_momentum = float(epsilon_momentum)
        self.epsilon_weight_decay = float(epsilon_weight_decay)
        self.alpha_lr = float(alpha_lr)
        self.alpha_weight_decay = float(alpha_weight_decay)
        self.epsilon_optimizer = torch.optim.SGD(
            self.epsilon_parameters,
            lr=self.epsilon_lr,
            momentum=self.epsilon_momentum,
            weight_decay=self.epsilon_weight_decay,
        )
        self.alpha_optimizer = self._new_alpha_optimizer()
        state.set_optimizer(self)

    def _new_alpha_optimizer(self):
        return torch.optim.AdamW(
            [
                {"params": [self.state.alpha], "lr": self.alpha_lr, "weight_decay": self.alpha_weight_decay},
                {"params": [self.state.alpha_scale], "lr": self.alpha_lr},
            ]
        )

    def on_vector_appended(self):
        # This mirrors the official implementation, which rebuilds the alpha
        # optimizer after expanding the vector pool.
        self.alpha_optimizer = self._new_alpha_optimizer()

    def zero_grad(self, set_to_none: bool = True):
        self.alpha_optimizer.zero_grad(set_to_none=set_to_none)
        self.epsilon_optimizer.zero_grad(set_to_none=set_to_none)

    def step(self):
        if self.state.vector_count > 1:
            self.alpha_optimizer.step()
        self.epsilon_optimizer.step()

    def state_dict(self):
        return {
            "vector_count": self.state.vector_count,
            "alpha_optimizer": self.alpha_optimizer.state_dict(),
            "epsilon_optimizer": self.epsilon_optimizer.state_dict(),
        }


def configure_cola_model(
    model: nn.Module,
    *,
    weight_margin: float = 0.01,
    shift_detection_threshold: float = 0.1,
    domain_ema: float = 0.2,
    max_num_vectors: Optional[int] = None,
):
    """Replace all affine BatchNorm layers with CoLA-aware versions."""
    model.train()
    source_bns = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, nn.BatchNorm2d)
    ]
    if not source_bns:
        raise RuntimeError("CoLA requires BatchNorm2d layers")
    stem = getattr(model, "conv1", None)
    if stem is None:
        raise RuntimeError("CoLA ResNet integration requires model.conv1")

    state = CoLAState(
        stem,
        weight_margin=weight_margin,
        shift_detection_threshold=shift_detection_threshold,
        domain_ema=domain_ema,
        max_num_vectors=max_num_vectors,
    )
    model.cola_state = state

    def replace(parent: nn.Module, prefix: str = ""):
        for child_name, child in list(parent.named_children()):
            if child_name == "cola_state":
                continue
            full_name = f"{prefix}.{child_name}" if prefix else child_name
            if isinstance(child, nn.BatchNorm2d) and not isinstance(child, CoLABatchNorm2d):
                setattr(parent, child_name, CoLABatchNorm2d(child, state, full_name))
            else:
                replace(child, full_name)

    replace(model)
    modules = {
        name: module
        for name, module in model.named_modules()
        if isinstance(module, CoLABatchNorm2d)
    }
    state.set_modules(modules)

    model.requires_grad_(False)
    for module in modules.values():
        module.train()
        module.weight.requires_grad_(True)
        module.bias.requires_grad_(True)
    state.alpha.requires_grad_(True)
    state.alpha_scale.requires_grad_(True)

    params = [param for param in model.parameters() if param.requires_grad]
    names = [
        name for name, param in model.named_parameters() if param.requires_grad
    ]
    if not params:
        raise RuntimeError("CoLA found no trainable parameters")
    return params, names


class CoLAAdapter(_ExternalAdapterBase):
    """ETA+CoLA principal-agent adaptation under the FEDEATA contract."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: CoLAOptimizer,
        restorer: CoLAProxyRestorer,
        *,
        num_classes: int,
        entropy_margin_factor: float = 0.4,
        diversity_margin: float = 0.05,
    ):
        super().__init__(model, optimizer, restorer)
        self.cola_state: CoLAState = model.cola_state
        self.e_margin = float(entropy_margin_factor) * math.log(float(num_classes))
        self.d_margin = float(diversity_margin)
        self.current_model_probs: Optional[torch.Tensor] = None
        self.sample_filter = "eta"
        self.counters.update(
            {
                "cola_shift_events": 0,
                "cola_saved_vectors": 0,
                "cola_domain_distance_sum": 0.0,
                "cola_domain_distance_count": 0,
                "cola_vector_count": 1,
            }
        )

    @torch.enable_grad()
    def adapt(self, images: torch.Tensor):
        self.optimizer.zero_grad(set_to_none=True)
        shift, distance = self.cola_state.observe(images)
        outputs = self.model(images)
        entropies = softmax_entropy(outputs)
        reliable_indices = (entropies.detach() < self.e_margin).nonzero(
            as_tuple=False
        ).flatten()

        selected_indices = reliable_indices
        if reliable_indices.numel() > 0:
            reliable_probs = outputs.softmax(1).detach().index_select(
                0, reliable_indices
            )
            if self.current_model_probs is not None:
                similarities = F.cosine_similarity(
                    self.current_model_probs.unsqueeze(0), reliable_probs, dim=1
                ).abs()
                keep = similarities < self.d_margin
                selected_indices = reliable_indices[keep]
                selected_probs = reliable_probs[keep]
            else:
                selected_probs = reliable_probs
            if selected_probs.numel() > 0:
                batch_mean = selected_probs.mean(dim=0)
                self.current_model_probs = (
                    batch_mean
                    if self.current_model_probs is None
                    else 0.9 * self.current_model_probs + 0.1 * batch_mean
                )

        updated = False
        loss_value = 0.0
        nonfinite = False
        if selected_indices.numel() > 0:
            selected_entropy = entropies.index_select(0, selected_indices)
            coefficients = torch.exp(
                selected_entropy.detach() * -1.0 + self.e_margin
            )
            loss = (selected_entropy * coefficients).mean()
            if torch.isfinite(loss):
                loss_value = float(loss.detach().cpu())
                loss.backward()
                self.optimizer.step()
                self.cola_state.mark_updated()
                updated = True
            else:
                nonfinite = True
            self.optimizer.zero_grad(set_to_none=True)

        recovered = self.restorer.step()
        self.counters["cola_shift_events"] += int(shift)
        self.counters["cola_saved_vectors"] = self.cola_state.saved_vectors
        self.counters["cola_domain_distance_sum"] += float(distance)
        self.counters["cola_domain_distance_count"] += 1
        self.counters["cola_vector_count"] = self.cola_state.vector_count
        stats = self._record(
            images.size(0),
            int(reliable_indices.numel()),
            int(selected_indices.numel()),
            loss_value,
            updated,
            nonfinite=nonfinite,
            recovered=recovered,
            cola_domain_shift=bool(shift),
            cola_domain_distance=float(distance),
            cola_vector_count=int(self.cola_state.vector_count),
            cola_saved_vectors=int(self.cola_state.saved_vectors),
        )
        return outputs.detach(), stats

    def finalize(self):
        if self.cola_state.dirty and self.cola_state.domain_mean is not None:
            self.cola_state.save_current_weight()
        self.counters["cola_saved_vectors"] = self.cola_state.saved_vectors
        self.counters["cola_vector_count"] = self.cola_state.vector_count
        return False

    def diagnostics(self):
        """Return JSON-safe CoLA state needed to audit the domain-vector path."""
        distance_count = int(self.counters.get("cola_domain_distance_count", 0))
        distance_sum = float(self.counters.get("cola_domain_distance_sum", 0.0))
        return {
            "shift_events": int(self.cola_state.shift_events),
            "saved_vectors": int(self.cola_state.saved_vectors),
            "vector_count": int(self.cola_state.vector_count),
            "mean_domain_distance": distance_sum / max(1, distance_count),
            "last_shift_distance": float(self.cola_state.last_shift_distance),
            "actual_affine_drift": self.cola_state.actual_drift_stats(),
            "entropy_margin": float(self.e_margin),
            "diversity_margin": float(self.d_margin),
        }
