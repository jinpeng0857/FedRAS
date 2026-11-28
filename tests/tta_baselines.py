"""Unified external TTA baselines for the FEDEATA evaluation protocol.

The implementations are adapted to the local ``adapt(images)`` contract while
preserving the defining update rules from the official MIT-licensed projects:

* SAR: https://github.com/mr-eggplant/SAR
  (SAM backend is vendored verbatim in ``test/sar_official/sam.py``.)
* RoTTA/PeTTA: https://github.com/hthieu166/petta
* CoTTA: https://github.com/qinenergy/cotta
* DeYO: https://github.com/Jhyun17/DeYO

The evaluator, rather than these classes, owns the Domain-wise/Continual reset
policy and the predict-then-update accuracy accounting.
"""

from copy import deepcopy
import math
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from tests.sar_official.sam import SAM


def softmax_entropy(logits: torch.Tensor) -> torch.Tensor:
    return -(logits.softmax(1) * logits.log_softmax(1)).sum(1)


def configure_cotta_model(model: nn.Module):
    """Official CoTTA scope: all weight/bias tensors, with batch BN statistics."""
    model.train()
    model.requires_grad_(False)
    params = []
    names = []
    for module_name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            module.track_running_stats = False
            module.running_mean = None
            module.running_var = None
        for parameter_name, parameter in module.named_parameters(recurse=False):
            if parameter_name not in ("weight", "bias") or parameter is None:
                continue
            parameter.requires_grad_(True)
            params.append(parameter)
            names.append(
                f"{module_name}.{parameter_name}"
                if module_name
                else parameter_name
            )
    if not params:
        raise RuntimeError("CoTTA found no trainable weight/bias parameters")
    return params, names


def configure_deyo_model(model: nn.Module):
    """Official DeYO CNN scope: BN affine tensors except the final ResNet stage."""
    model.train()
    model.requires_grad_(False)
    params = []
    names = []
    for module_name, module in model.named_modules():
        if not isinstance(module, nn.BatchNorm2d):
            continue
        module.track_running_stats = False
        module.running_mean = None
        module.running_var = None
        update = not (
            module_name == "layer4" or module_name.startswith("layer4.")
        )
        for parameter_name, parameter in module.named_parameters(recurse=False):
            if parameter_name not in ("weight", "bias") or parameter is None:
                continue
            parameter.requires_grad_(update)
            if update:
                params.append(parameter)
                names.append(f"{module_name}.{parameter_name}")
    if not params:
        raise RuntimeError("DeYO found no trainable BN affine parameters")
    return params, names


class _ExternalAdapterBase:
    def __init__(self, model, optimizer, restorer):
        self.model = model
        self.optimizer = optimizer
        self.restorer = restorer
        self.counters: Dict[str, float] = {
            "samples": 0,
            "quality_passed": 0,
            "margin_passed": 0,
            "margin_sum": 0.0,
            "entropy_passed": 0,
            "relative_entropy_passed": 0,
            "absolute_entropy_passed": 0,
            "relative_entropy_quality_batches": 0,
            "relative_entropy_threshold_sum": 0.0,
            "effective_entropy_threshold_sum": 0.0,
            "nonredundant": 0,
            "region_reliable": 0,
            "selected": 0,
            "recap_soft_weight_sum": 0.0,
            "recap_soft_weight_square_sum": 0.0,
            "recap_soft_weight_count": 0,
            "gradient_norm_sum": 0.0,
            "gradient_norm_batches": 0,
            "gradient_clipped_batches": 0,
            "nonfinite_gradient_batches": 0,
            "batches": 0,
            "updated_batches": 0,
            "skipped_batches": 0,
            "nonfinite_batches": 0,
            "loss_sum": 0.0,
            "recap_loss_sum": 0.0,
            "eata_regularization_sum": 0.0,
        }

    def _record(self, batch_size, entropy_passed, selected, loss, updated, **extra):
        nonfinite = not math.isfinite(loss)
        stats = {
            "samples": int(batch_size),
            "quality_passed": int(selected),
            "entropy_passed": int(entropy_passed),
            "nonredundant": int(entropy_passed),
            "region_reliable": int(selected),
            "selected": int(selected),
            "updated": bool(updated),
            "nonfinite": bool(nonfinite),
            "loss": float(loss) if math.isfinite(loss) else 0.0,
            "recap_loss": 0.0,
            "eata_regularization": 0.0,
            "recovered": 0,
            **extra,
        }
        self.counters["samples"] += stats["samples"]
        self.counters["quality_passed"] += stats["quality_passed"]
        self.counters["entropy_passed"] += stats["entropy_passed"]
        self.counters["nonredundant"] += stats["nonredundant"]
        self.counters["region_reliable"] += stats["region_reliable"]
        self.counters["selected"] += stats["selected"]
        self.counters["batches"] += 1
        self.counters["updated_batches"] += int(updated)
        self.counters["skipped_batches"] += int(not updated)
        self.counters["nonfinite_batches"] += int(nonfinite)
        self.counters["loss_sum"] += stats["loss"]
        return stats

    @torch.inference_mode()
    def predict(self, images):
        return self.model(images)

    def finalize(self):
        """Flush adapter state at a corruption/stream boundary when supported."""
        return False


def configure_sar_model(model: nn.Module):
    """Configure the official SAR normalization-only parameter scope."""
    model.train()
    model.requires_grad_(False)
    params, names = [], []
    for module_name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            module.train()
            module.track_running_stats = False
            module.running_mean = None
            module.running_var = None
        if not isinstance(module, (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
            continue
        if module_name == "layer4" or module_name.startswith("layer4."):
            continue
        for parameter_name, parameter in module.named_parameters(recurse=False):
            if parameter_name not in ("weight", "bias") or parameter is None:
                continue
            parameter.requires_grad_(True)
            params.append(parameter)
            names.append(f"{module_name}.{parameter_name}")
    if not params:
        raise RuntimeError("SAR found no trainable normalization parameters")
    return params, names


class SARAdapter(_ExternalAdapterBase):
    """Unified SAR adapter: reliable entropy + two-step SAM + recovery."""

    def __init__(
        self,
        model,
        optimizer: SAM,
        restorer,
        margin: float,
        reset_threshold: float = 0.2,
        steps: int = 1,
    ):
        super().__init__(model, optimizer, restorer)
        if steps <= 0 or margin <= 0:
            raise ValueError("SAR steps and margin must be positive")
        self.margin = float(margin)
        self.reset_threshold = float(reset_threshold)
        self.steps = int(steps)
        self.ema = None
        self.source_state = {
            name: value.detach().clone() for name, value in model.state_dict().items()
        }
        self.source_optimizer_state = deepcopy(optimizer.state_dict())

    @torch.enable_grad()
    def adapt(self, images):
        outputs = None
        first_selected = 0
        second_selected = 0
        loss_value = 0.0
        updated = False
        reset = False
        finite = True
        for _ in range(self.steps):
            self.optimizer.zero_grad(set_to_none=True)
            outputs = self.model(images)
            entropy = softmax_entropy(outputs)
            first_ids = (entropy.detach() < self.margin).nonzero(
                as_tuple=False
            ).flatten()
            first_selected = int(first_ids.numel())
            if first_ids.numel() == 0:
                finite = True
                break
            first_loss = entropy.index_select(0, first_ids).mean()
            if not bool(torch.isfinite(first_loss)):
                finite = False
                break
            first_loss.backward()
            self.optimizer.first_step(zero_grad=True)

            second_entropy = softmax_entropy(self.model(images)).index_select(
                0, first_ids
            )
            second_ids = (second_entropy.detach() < self.margin).nonzero(
                as_tuple=False
            ).flatten()
            second_selected = int(second_ids.numel())
            if second_ids.numel() == 0:
                # The official SAM implementation restores parameters in
                # ``second_step`` when gradients are present.  Materialize
                # zero gradients here so a rejected second pass restores the
                # perturbation without changing the model update rule.
                for group in self.optimizer.param_groups:
                    for parameter in group["params"]:
                        if parameter.grad is None:
                            parameter.grad = torch.zeros_like(parameter)
                self.optimizer.second_step(zero_grad=True)
                break
            second_loss = second_entropy.index_select(0, second_ids).mean()
            finite = bool(torch.isfinite(second_loss))
            if not finite:
                for group in self.optimizer.param_groups:
                    for parameter in group["params"]:
                        if parameter.grad is None:
                            parameter.grad = torch.zeros_like(parameter)
                self.optimizer.second_step(zero_grad=True)
                break
            loss_value = float(second_loss.detach().cpu())
            self.ema = (
                loss_value
                if self.ema is None
                else 0.9 * self.ema + 0.1 * loss_value
            )
            second_loss.backward()
            self.optimizer.second_step(zero_grad=True)
            updated = True
            if self.ema is not None and self.ema < self.reset_threshold:
                self.model.load_state_dict(self.source_state, strict=True)
                self.optimizer.load_state_dict(self.source_optimizer_state)
                self.ema = None
                reset = True
            break

        recovered = self.restorer.step()
        stats = self._record(
            images.size(0),
            first_selected,
            second_selected,
            loss_value if finite else float("nan"),
            updated and finite,
            sar_first_selected=first_selected,
            sar_second_selected=second_selected,
            sar_ema=self.ema,
            sar_reset=reset,
            recovered=recovered,
        )
        return outputs.detach(), stats


class RobustBatchNorm2d(nn.BatchNorm2d):
    """RoTTA/PeTTA robust BN while retaining BatchNorm2d identity for DSR."""

    def __init__(self, source: nn.BatchNorm2d, momentum: float):
        super().__init__(
            source.num_features,
            eps=source.eps,
            momentum=None,
            affine=source.affine,
            track_running_stats=False,
        )
        if source.affine:
            self.weight.data.copy_(source.weight.detach())
            self.bias.data.copy_(source.bias.detach())
        self.momentum = float(momentum)
        self.register_buffer("source_mean", source.running_mean.detach().clone())
        self.register_buffer("source_var", source.running_var.detach().clone())

    def forward(self, x):
        if self.training:
            batch_var, batch_mean = torch.var_mean(
                x, dim=(0, 2, 3), unbiased=False, keepdim=False
            )
            mean = (1.0 - self.momentum) * self.source_mean + self.momentum * batch_mean
            var = (1.0 - self.momentum) * self.source_var + self.momentum * batch_var
            self.source_mean.copy_(mean.detach())
            self.source_var.copy_(var.detach())
        else:
            mean, var = self.source_mean, self.source_var
        mean = mean.view(1, -1, 1, 1)
        var = var.view(1, -1, 1, 1)
        result = (x - mean) / torch.sqrt(var + self.eps)
        if self.affine:
            result = result * self.weight.view(1, -1, 1, 1)
            result = result + self.bias.view(1, -1, 1, 1)
        return result


def _replace_bn(module: nn.Module, alpha: float):
    for name, child in list(module.named_children()):
        if isinstance(child, nn.BatchNorm2d) and not isinstance(child, RobustBatchNorm2d):
            replacement = RobustBatchNorm2d(child, alpha)
            replacement.to(device=child.weight.device, dtype=child.weight.dtype)
            setattr(module, name, replacement)
        else:
            _replace_bn(child, alpha)


def configure_rotta_model(model, alpha: float = 0.05):
    model.train()
    _replace_bn(model, alpha)
    model.requires_grad_(False)
    params, names = [], []
    for module_name, module in model.named_modules():
        if not isinstance(module, RobustBatchNorm2d):
            continue
        for parameter_name, parameter in module.named_parameters(recurse=False):
            if parameter_name in ("weight", "bias") and parameter is not None:
                parameter.requires_grad_(True)
                params.append(parameter)
                names.append(f"{module_name}.{parameter_name}")
    if not params:
        raise RuntimeError("RoTTA/PeTTA found no trainable robust-BN parameters")
    return params, names


class _ClassBalancedMemory:
    def __init__(self, capacity: int, num_classes: int, lambda_t=1.0, lambda_u=1.0):
        if capacity <= 0 or num_classes <= 0:
            raise ValueError("memory capacity and number of classes must be positive")
        self.capacity = int(capacity)
        self.num_classes = int(num_classes)
        self.lambda_t = float(lambda_t)
        self.lambda_u = float(lambda_u)
        self.items: List[List[Tuple[torch.Tensor, float, int]]] = [
            [] for _ in range(self.num_classes)
        ]

    def _score(self, age, uncertainty):
        return self.lambda_t / (1.0 + math.exp(-age / self.capacity)) + self.lambda_u * uncertainty / math.log(self.num_classes)

    def add(self, data, prediction, uncertainty):
        prediction = int(prediction)
        if not 0 <= prediction < self.num_classes:
            return
        for class_items in self.items:
            for index, item in enumerate(class_items):
                class_items[index] = (item[0], item[1], item[2] + 1)
        class_items = self.items[prediction]
        new_score = self._score(0, float(uncertainty))
        occupancy = sum(len(item) for item in self.items)
        per_class_limit = self.capacity / self.num_classes
        candidates = []
        if len(class_items) >= per_class_limit or occupancy >= self.capacity:
            classes = [prediction] if len(class_items) >= per_class_limit else [
                index for index, item in enumerate(self.items)
                if len(item) == max(len(group) for group in self.items)
            ]
            for class_id in classes:
                candidates.extend((self._score(age, uncertainty), class_id, idx)
                                  for idx, (_x, uncertainty, age) in enumerate(self.items[class_id]))
        if candidates:
            worst_score, worst_class, worst_index = max(candidates, key=lambda value: value[0])
            if worst_score <= new_score:
                return
            self.items[worst_class].pop(worst_index)
        elif occupancy >= self.capacity:
            return
        class_items.append((data.detach(), float(uncertainty), 0))

    def get(self):
        values, ages = [], []
        for class_items in self.items:
            for data, _uncertainty, age in class_items:
                values.append(data)
                ages.append(float(age) / self.capacity)
        return values, ages


def _strong_tensor_augment(images, mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761)):
    # Inputs are normalized tensors. Convert to image space using dataset-specific mean and std
    if not isinstance(mean, torch.Tensor):
        mean = images.new_tensor(mean).view(1, -1, 1, 1)
    else:
        mean = mean.to(device=images.device, dtype=images.dtype).view(1, -1, 1, 1)
    if not isinstance(std, torch.Tensor):
        std = images.new_tensor(std).view(1, -1, 1, 1)
    else:
        std = std.to(device=images.device, dtype=images.dtype).view(1, -1, 1, 1)
    augmented = (images * std + mean).clamp(0.0, 1.0)
    from torchvision.transforms import InterpolationMode
    from torchvision.transforms import functional as TF

    augmented = TF.adjust_brightness(
        augmented, float(torch.empty(()).uniform_(0.6, 1.4))
    )
    augmented = TF.adjust_contrast(
        augmented, float(torch.empty(()).uniform_(0.7, 1.3))
    )
    augmented = TF.pad(augmented, [16, 16, 16, 16], padding_mode="edge")
    augmented = TF.affine(
        augmented,
        angle=float(torch.empty(()).uniform_(-15.0, 15.0)),
        translate=[0, 0],
        scale=float(torch.empty(()).uniform_(0.9, 1.1)),
        shear=[0.0, 0.0],
        interpolation=InterpolationMode.BILINEAR,
    )
    augmented = TF.center_crop(augmented, [32, 32])
    flip = torch.rand(images.size(0), device=images.device) < 0.5
    augmented[flip] = augmented[flip].flip(-1)
    augmented = (augmented + torch.randn_like(augmented) * 0.005).clamp(0.0, 1.0)
    return (augmented - mean) / std


class RoTTAAdapter(_ExternalAdapterBase):
    """Unified RoTTA adapter with class-balanced memory and robust BN."""

    def __init__(self, model, optimizer, restorer, num_classes, memory_size=64,
                 update_frequency=64, nu=0.001, lambda_t=1.0, lambda_u=1.0,
                 input_mean=(0.5071, 0.4867, 0.4408), input_std=(0.2675, 0.2565, 0.2761)):
        super().__init__(model, optimizer, restorer)
        self.memory = _ClassBalancedMemory(memory_size, num_classes, lambda_t, lambda_u)
        self.input_mean = input_mean
        self.input_std = input_std
        self.update_frequency = int(update_frequency)
        if self.update_frequency <= 0:
            raise ValueError("RoTTA update_frequency must be positive")
        self.nu = float(nu)
        self.current_instance = 0
        self.next_update = self.update_frequency
        self.last_update_instance = 0
        self.model_ema = deepcopy(model).eval()
        self.model_ema.requires_grad_(False)

    @torch.no_grad()
    def _update_teacher(self):
        for teacher, student in zip(self.model_ema.parameters(), self.model.parameters()):
            teacher.mul_(1.0 - self.nu).add_(student, alpha=self.nu)

    @torch.enable_grad()
    def _update_model(self):
        values, ages = self.memory.get()
        if not values:
            return None
        data = torch.stack(values)
        ages_tensor = data.new_tensor(ages)
        weights = torch.exp(-ages_tensor) / (1.0 + torch.exp(-ages_tensor))
        self.model.train()
        self.model_ema.train()
        with torch.no_grad():
            teacher_logits = self.model_ema(data)
        student_logits = self.model(_strong_tensor_augment(data, self.input_mean, self.input_std))
        loss_per_sample = -teacher_logits.softmax(1) * student_logits.log_softmax(1)
        loss = (loss_per_sample.sum(1) * weights).mean()
        if not bool(torch.isfinite(loss)):
            return None
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self._update_teacher()
        return float(loss.detach().cpu())

    @torch.enable_grad()
    def adapt(self, images):
        self.model.eval()
        self.model_ema.eval()
        with torch.no_grad():
            outputs = self.model_ema(images)
            probability = outputs.softmax(1)
            pseudo = probability.argmax(1)
            uncertainty = softmax_entropy(outputs)
            for data, label, entropy in zip(images, pseudo, uncertainty):
                self.memory.add(data, label.item(), entropy.item())
        self.current_instance += int(images.size(0))
        update_loss = None
        updated = False
        if self.current_instance >= self.next_update:
            update_loss = self._update_model()
            updated = update_loss is not None
            self.next_update = (
                self.current_instance // self.update_frequency + 1
            ) * self.update_frequency
            self.last_update_instance = self.current_instance
        recovered = self.restorer.step()
        stats = self._record(
            images.size(0), images.size(0), images.size(0),
            update_loss if update_loss is not None else 0.0,
            updated, rotta_memory_occupancy=sum(len(group) for group in self.memory.items),
            rotta_update_loss=update_loss, recovered=recovered,
        )
        return outputs.detach(), stats

    def finalize(self):
        """Apply one final update to samples after the last full memory period."""
        if self.current_instance <= self.last_update_instance:
            return False
        update_loss = self._update_model()
        self.last_update_instance = self.current_instance
        return update_loss is not None

    @torch.inference_mode()
    def predict(self, images):
        return self.model_ema(images)


class PeTTAAdapter(RoTTAAdapter):
    """PeTTA extension: source prototypes, adaptive regularization and anchor loss."""

    def __init__(self, model, optimizer, restorer, num_classes, source_mean=None,
                 source_variance=None, memory_size=64, update_frequency=64, nu=0.001,
                 lambda_0=10.0, anchor_weight=1.0, alpha_0=0.001,
                 regularizer="cosine", loss_func="sce",
                 input_mean=(0.5071, 0.4867, 0.4408), input_std=(0.2675, 0.2565, 0.2761),
                 **kwargs):
        if source_mean is None and "class_feature_mean" in kwargs:
            source_mean = kwargs.pop("class_feature_mean")
        if source_variance is None and "class_feature_variance" in kwargs:
            source_variance = kwargs.pop("class_feature_variance")
        if "petta_lambda" in kwargs:
            lambda_0 = kwargs.pop("petta_lambda")
        if "petta_alpha" in kwargs:
            alpha_0 = kwargs.pop("petta_alpha")
        super().__init__(model, optimizer, restorer, num_classes, memory_size,
                         update_frequency, nu, input_mean=input_mean, input_std=input_std)
        if source_mean is None or source_mean.ndim != 2 or source_mean.size(0) != num_classes:
            raise ValueError(
                "PeTTA requires class_feature_mean with shape "
                f"[{num_classes}, feature_dim]"
            )
        if source_variance is None or source_variance.shape != source_mean.shape:
            raise ValueError("PeTTA class feature moments must have matching shapes")
        classifier = getattr(model, "fc", None)
        if classifier is None or source_mean.size(1) != classifier.in_features:
            raise ValueError("PeTTA feature moments do not match model.fc.in_features")
        self.source_mean = source_mean.detach().float().to(next(model.parameters()).device)
        self.source_variance = source_variance.detach().float().to(self.source_mean.device).clamp_min(1e-6)
        self.model_init = deepcopy(model).eval()
        self.model_init.requires_grad_(False)
        self.init_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
        self.lambda_0 = float(lambda_0)
        self.anchor_weight = float(anchor_weight)
        self.alpha_0 = float(alpha_0)
        self.nu = self.alpha_0
        if regularizer not in ("l2", "cosine", "none"):
            raise ValueError("PeTTA regularizer must be l2, cosine, or none")
        if loss_func not in ("sce", "ce"):
            raise ValueError("PeTTA loss_func must be sce or ce")
        self.regularizer = regularizer
        self.loss_func = loss_func
        self._classifier_name = "fc"
        self.last_divergence = 0.0
        self.current_prototypes = self.source_mean.clone()

    def _features(self, model, images):
        captured = {}
        classifier = getattr(model, self._classifier_name, None)
        if classifier is None:
            raise RuntimeError("PeTTA requires a model.fc classifier")
        handle = classifier.register_forward_pre_hook(
            lambda _module, inputs: captured.setdefault("features", inputs[0])
        )
        try:
            logits = model(images)
        finally:
            handle.remove()
        features = captured.get("features")
        if features is None:
            raise RuntimeError("PeTTA failed to capture classifier features")
        return logits, features.reshape(features.size(0), -1)

    def _regularizer(self):
        if self.regularizer == "none":
            return self.source_mean.new_zeros(())
        value = None
        count = 0
        for name, parameter in self.model.named_parameters():
            if not parameter.requires_grad or name not in self.init_state:
                continue
            anchor = self.init_state[name].to(device=parameter.device, dtype=parameter.dtype)
            if self.regularizer == "cosine":
                term = -F.cosine_similarity(
                    parameter.reshape(1, -1), anchor.reshape(1, -1), dim=1
                ).mean()
            else:
                term = (parameter - anchor).square().mean()
            value = term if value is None else value + term
            count += 1
        if value is None or count == 0:
            return self.source_mean.new_zeros(())
        return value / float(count)

    @torch.enable_grad()
    def _update_model(self):
        values, ages = self.memory.get()
        if not values:
            return None
        data = torch.stack(values)
        ages_tensor = data.new_tensor(ages)
        with torch.no_grad():
            teacher_logits, teacher_features = self._features(self.model_ema, data)
            initial_logits = self.model_init(data)
        original_logits, _ = self._features(self.model, data)
        student_logits, _ = self._features(self.model, _strong_tensor_augment(data))
        pseudo = teacher_logits.argmax(1)
        divergences = []
        for class_id in pseudo.unique().tolist():
            mask = pseudo == int(class_id)
            current_mean = teacher_features[mask].mean(0)
            divergences.append(
                ((current_mean - self.current_prototypes[class_id]).square()
                 / self.source_variance[class_id]).mean()
            )
            self.current_prototypes[class_id] = (
                0.95 * self.current_prototypes[class_id]
                + 0.05 * current_mean.detach()
            )
        raw_divergence = torch.stack(divergences).mean() if divergences else data.new_zeros(())
        divergence = (1.0 - torch.exp(-raw_divergence)).detach().clamp(0.0, 1.0)
        self.last_divergence = float(divergence.cpu())
        adaptive_lambda = self.lambda_0 * divergence
        adaptive_alpha = self.alpha_0 * (1.0 - divergence)
        if self.loss_func == "sce":
            teacher_prob = teacher_logits.softmax(1)
            original_prob = original_logits.softmax(1)
            student_prob = student_logits.softmax(1)
            classification = (
                -0.25 * (teacher_prob * original_logits.log_softmax(1)).sum(1)
                -0.25 * (original_prob * teacher_logits.log_softmax(1)).sum(1)
                -0.25 * (teacher_prob * student_logits.log_softmax(1)).sum(1)
                -0.25 * (student_prob * teacher_logits.log_softmax(1)).sum(1)
            ).mean()
        else:
            classification = (
                -(teacher_logits.softmax(1) * student_logits.log_softmax(1))
                .sum(1)
                .mean()
            )
        regularization = self._regularizer()
        anchor = -(initial_logits.softmax(1) * student_logits.log_softmax(1)).sum(1).mean()
        loss = classification + adaptive_lambda * regularization + self.anchor_weight * anchor
        if not bool(torch.isfinite(loss)):
            return None
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.nu = float(adaptive_alpha.detach().cpu())
        self._update_teacher()
        return float(loss.detach().cpu())


class CoTTAAdapter(_ExternalAdapterBase):
    """CoTTA mean teacher, augmentation averaging, and stochastic restoration."""

    def __init__(
        self,
        model,
        optimizer,
        restorer,
        mt_alpha=0.999,
        restore_probability=0.01,
        anchor_confidence=0.72,
        augmentation_count=32,
        input_mean=(0.0, 0.0, 0.0),
        input_std=(1.0, 1.0, 1.0),
    ):
        super().__init__(model, optimizer, restorer)
        self.model_anchor = deepcopy(model).eval()
        self.model_ema = deepcopy(model).eval()
        self.source_state = {
            name: value.detach().clone()
            for name, value in model.state_dict().items()
        }
        for frozen in (self.model_anchor, self.model_ema):
            frozen.requires_grad_(False)
        self.mt_alpha = float(mt_alpha)
        self.restore_probability = float(restore_probability)
        self.anchor_confidence = float(anchor_confidence)
        self.augmentation_count = int(augmentation_count)
        self.input_mean = tuple(float(value) for value in input_mean)
        self.input_std = tuple(float(value) for value in input_std)

    def _augment(self, images):
        """Official CoTTA/KATANA CIFAR transform in normalized-input space."""
        from torchvision.transforms import InterpolationMode
        from torchvision.transforms import functional as TF

        mean = images.new_tensor(self.input_mean).view(1, -1, 1, 1)
        std = images.new_tensor(self.input_std).view(1, -1, 1, 1)
        augmented = (images * std + mean).clamp(0.0, 1.0)

        transforms = [
            lambda x: TF.adjust_brightness(
                x, float(torch.empty(()).uniform_(0.6, 1.4))
            ),
            lambda x: TF.adjust_contrast(
                x, float(torch.empty(()).uniform_(0.7, 1.3))
            ),
            lambda x: TF.adjust_saturation(
                x, float(torch.empty(()).uniform_(0.5, 1.5))
            ),
            lambda x: TF.adjust_hue(
                x, float(torch.empty(()).uniform_(-0.06, 0.06))
            ),
            lambda x: TF.adjust_gamma(
                x.clamp(1e-8, 1.0),
                float(torch.empty(()).uniform_(0.7, 1.3)),
            ),
        ]
        for index in torch.randperm(len(transforms)).tolist():
            augmented = transforms[index](augmented)

        augmented = TF.pad(
            augmented, [16, 16, 16, 16], padding_mode="edge"
        )
        augmented = TF.affine(
            augmented,
            angle=float(torch.empty(()).uniform_(-15.0, 15.0)),
            translate=[
                int(torch.randint(-2, 3, ()).item()),
                int(torch.randint(-2, 3, ()).item()),
            ],
            scale=float(torch.empty(()).uniform_(0.9, 1.1)),
            shear=[0.0, 0.0],
            interpolation=InterpolationMode.BILINEAR,
        )
        augmented = TF.gaussian_blur(
            augmented,
            kernel_size=[5, 5],
            sigma=float(torch.empty(()).uniform_(0.001, 0.5)),
        )
        augmented = TF.center_crop(augmented, [32, 32])
        if bool(torch.rand(()) < 0.5):
            augmented = augmented.flip(-1)
        augmented = (augmented + torch.randn_like(augmented) * 0.005).clamp(
            0.0, 1.0
        )
        return (augmented - mean) / std

    @torch.no_grad()
    def _update_teacher(self):
        for teacher, student in zip(
            self.model_ema.parameters(), self.model.parameters()
        ):
            teacher.mul_(self.mt_alpha).add_(
                student, alpha=1.0 - self.mt_alpha
            )

    @torch.no_grad()
    def _stochastic_restore(self):
        restored = 0
        for name, parameter in self.model.named_parameters():
            if not parameter.requires_grad:
                continue
            anchor = self.source_state[name].to(
                device=parameter.device, dtype=parameter.dtype
            )
            mask = torch.rand_like(parameter) < self.restore_probability
            parameter.copy_(torch.where(mask, anchor, parameter))
            restored += int(mask.sum().item())
        return restored

    @torch.enable_grad()
    def adapt(self, images):
        self.optimizer.zero_grad(set_to_none=True)
        student_logits = self.model(images)
        with torch.no_grad():
            anchor_confidence = self.model_anchor(images).softmax(1).amax(1)
            standard_teacher_logits = self.model_ema(images)
            used_augmentation = (
                float(anchor_confidence.mean().item()) < self.anchor_confidence
                and self.augmentation_count > 0
            )
            if self.augmentation_count > 0:
                augmented_teacher_logits = torch.stack(
                    [
                        self.model_ema(self._augment(images))
                        for _ in range(self.augmentation_count)
                    ]
                ).mean(0)
            else:
                augmented_teacher_logits = None
            teacher_logits = (
                augmented_teacher_logits
                if used_augmentation
                else standard_teacher_logits
            )

        loss = -(
            teacher_logits.softmax(1) * student_logits.log_softmax(1)
        ).sum(1).mean()
        finite = bool(torch.isfinite(loss))
        if finite:
            loss.backward()
            self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        if finite:
            self._update_teacher()
            restored = self._stochastic_restore()
        else:
            restored = 0
        self.restorer.step()
        stats = self._record(
            images.size(0),
            images.size(0),
            images.size(0),
            float(loss.detach().cpu()) if finite else float("nan"),
            finite,
            teacher_augmentation_used=used_augmentation,
            stochastic_restored_scalars=restored,
        )
        # CoTTA's teacher prediction is computed before the student update.
        return teacher_logits.detach(), stats

    @torch.inference_mode()
    def predict(self, images):
        return self.model_ema(images)


class DeYOAdapter(_ExternalAdapterBase):
    """DeYO entropy/PLPD filtering and joint entropy/PLPD reweighting."""

    def __init__(
        self,
        model,
        optimizer,
        restorer,
        num_classes,
        entropy_margin_factor=0.5,
        reweight_margin_factor=0.4,
        plpd_threshold=0.2,
        patch_grid=4,
    ):
        super().__init__(model, optimizer, restorer)
        self.entropy_margin = float(entropy_margin_factor) * math.log(num_classes)
        self.reweight_margin = (
            float(reweight_margin_factor) * math.log(num_classes)
        )
        self.plpd_threshold = float(plpd_threshold)
        self.patch_grid = int(patch_grid)

    def _shuffle_patches(self, images):
        batch, channels, height, width = images.shape
        grid = self.patch_grid
        if height % grid or width % grid:
            raise ValueError(
                f"DeYO patch grid {grid} must divide image size {height}x{width}"
            )
        patch_h, patch_w = height // grid, width // grid
        patches = images.reshape(
            batch, channels, grid, patch_h, grid, patch_w
        ).permute(0, 2, 4, 1, 3, 5).reshape(
            batch, grid * grid, channels, patch_h, patch_w
        )
        order = torch.argsort(
            torch.rand(batch, grid * grid, device=images.device), dim=1
        )
        patches = patches[
            torch.arange(batch, device=images.device).unsqueeze(1), order
        ]
        return patches.reshape(
            batch, grid, grid, channels, patch_h, patch_w
        ).permute(0, 3, 1, 4, 2, 5).reshape(
            batch, channels, height, width
        )

    @torch.enable_grad()
    def adapt(self, images):
        self.optimizer.zero_grad(set_to_none=True)
        logits = self.model(images)
        entropies = softmax_entropy(logits)
        entropy_indices = (
            entropies.detach() < self.entropy_margin
        ).nonzero(as_tuple=False).flatten()
        selected = 0
        loss_value = 0.0
        finite = True
        if entropy_indices.numel() > 0:
            filtered_images = images.index_select(0, entropy_indices).detach()
            with torch.no_grad():
                destroyed_logits = self.model(
                    self._shuffle_patches(filtered_images)
                )
            filtered_probabilities = logits.index_select(
                0, entropy_indices
            ).softmax(1)
            pseudo_labels = filtered_probabilities.argmax(1, keepdim=True)
            original_confidence = filtered_probabilities.gather(
                1, pseudo_labels
            ).squeeze(1)
            destroyed_confidence = destroyed_logits.softmax(1).gather(
                1, pseudo_labels
            ).squeeze(1)
            plpd = original_confidence - destroyed_confidence
            plpd_mask = plpd.detach() > self.plpd_threshold
            selected = int(plpd_mask.sum().item())
            if selected:
                selected_entropy = entropies.index_select(
                    0, entropy_indices
                )[plpd_mask]
                selected_plpd = plpd[plpd_mask]
                weights = torch.exp(
                    self.reweight_margin - selected_entropy.detach()
                ) + torch.exp(selected_plpd.detach())
                loss = (weights * selected_entropy).mean()
                finite = bool(torch.isfinite(loss))
                if finite:
                    loss.backward()
                    self.optimizer.step()
                    loss_value = float(loss.detach().cpu())
        self.optimizer.zero_grad(set_to_none=True)
        self.restorer.step()
        stats = self._record(
            images.size(0),
            entropy_indices.numel(),
            selected,
            loss_value if finite else float("nan"),
            finite and selected > 0,
            plpd_passed=selected,
        )
        return logits.detach(), stats
