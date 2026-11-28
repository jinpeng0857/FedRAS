"""Dual reliability and ReCAP / EATA adaptation adapter."""

import math
from copy import deepcopy
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from sparse_restore import PeriodicSparseRestorer
from adapters.base import (
    BASE_METHODS,
    RGA_METHODS,
    DUAL_RELIABILITY_METHODS,
    EATA_D_MARGIN,
    EATA_E_MARGIN_FACTOR,
    EATA_FISHER_ALPHA,
    DSR_GRAD_CLIP_NORM,
    L0_FACTOR,
    MARGIN_QUALITY_THRESHOLD,
    RECAP_SOFT_WEIGHT_TEMPERATURE_FACTOR,
    TAU_RE_FACTOR,
    RegionConfidenceProxy,
    get_linear_classifier,
    resolve_tta_method,
)


def top1_top2_probability_margin(probability: torch.Tensor):
    """Return the absolute Top-1 minus Top-2 probability margin."""
    if probability.ndim != 2 or probability.size(1) < 2:
        raise ValueError("probability must have shape [batch, classes>=2]")
    top_two = probability.topk(k=2, dim=1, largest=True, sorted=True).values
    return top_two[:, 0] - top_two[:, 1]


def recap_region_soft_weight(
    regional_entropy: torch.Tensor,
    neutral_entropy: float,
    temperature: float,
):
    """Bounded monotone ReCAP weight with unit weight at the neutral point."""
    from adapters.base import RECAP_SOFT_WEIGHT_LOGIT_LIMIT, REWEIGHT_CAP

    if not math.isfinite(float(temperature)) or float(temperature) <= 0:
        raise ValueError("temperature must be finite and positive")
    scaled_reliability = (
        float(neutral_entropy) - regional_entropy.detach()
    ) / float(temperature)
    scaled_reliability = scaled_reliability.clamp(
        min=-RECAP_SOFT_WEIGHT_LOGIT_LIMIT,
        max=RECAP_SOFT_WEIGHT_LOGIT_LIMIT,
    )
    return REWEIGHT_CAP * torch.sigmoid(scaled_reliability)


class DualReliabilityAdapter:
    """Matched Tent, EATA, ReCAP, and DSR comparison adapter."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        feature_variance: torch.Tensor,
        restorer: PeriodicSparseRestorer,
        fishers,
        tta_method: str = None,
        eata_e_margin_factor: float = EATA_E_MARGIN_FACTOR,
        eata_d_margin: float = EATA_D_MARGIN,
        eata_fisher_alpha: float = EATA_FISHER_ALPHA,
        margin_threshold: float = MARGIN_QUALITY_THRESHOLD,
        recap_soft_weight_temperature_factor: float = (
            RECAP_SOFT_WEIGHT_TEMPERATURE_FACTOR
        ),
        dsr_grad_clip_norm: float = DSR_GRAD_CLIP_NORM,
        confidence_gate: str = "none",
        gate_margin_headroom: float = 0.10,
    ):
        _classifier_name, classifier = get_linear_classifier(model)
        log_classes = math.log(float(classifier.out_features))
        self.model = model
        self.optimizer = optimizer
        self.proxy = RegionConfidenceProxy(classifier.weight, feature_variance)
        self.tau_re = TAU_RE_FACTOR * log_classes
        self.l0 = L0_FACTOR * log_classes
        self.restorer = restorer
        self.tta_method = resolve_tta_method(tta_method)
        if confidence_gate not in {"none", "source_agreement", "source_calibrated"}:
            raise ValueError(f"unsupported confidence gate: {confidence_gate}")
        if gate_margin_headroom < 0:
            raise ValueError("gate_margin_headroom must be non-negative")
        self.confidence_gate = confidence_gate
        self.gate_margin_headroom = float(gate_margin_headroom)
        self.use_eata = self.tta_method.startswith("eata")
        self.use_base = self.tta_method in BASE_METHODS
        self.use_recap = (
            "recap" in self.tta_method
            or self.tta_method in RGA_METHODS
            or self.use_base
        )
        self.use_dsr = self.tta_method.endswith("_dsr")
        requested_grad_clip_norm = (
            None
            if dsr_grad_clip_norm is None
            else float(dsr_grad_clip_norm)
        )
        if self.use_dsr and requested_grad_clip_norm is not None and (
            not math.isfinite(requested_grad_clip_norm)
            or requested_grad_clip_norm < 0
        ):
            raise ValueError(
                "dsr_grad_clip_norm must be finite and non-negative"
            )
        # Zero explicitly disables clipping. Strict mechanism ablations use
        # this so DSR represents sparse restoration only.
        self.dsr_grad_clip_norm = (
            requested_grad_clip_norm
            if (
                self.use_dsr
                and requested_grad_clip_norm is not None
                and requested_grad_clip_norm > 0
            )
            else None
        )
        self._trainable_parameters = [
            param for param in self.model.parameters() if param.requires_grad
        ]
        if not self._trainable_parameters:
            raise RuntimeError("adapter requires trainable parameters")
        self.anchor_parameters = {
            name: parameter.detach().clone()
            for name, parameter in self.model.named_parameters()
            if parameter.requires_grad
        }
        self.source_model = None
        if self.confidence_gate != "none":
            self.source_model = deepcopy(self.model)
            self.source_model.eval()
            self.source_model.requires_grad_(False)
        self.sample_filter = (
            "eata"
            if self.use_eata
            else (
                "all_samples"
                if self.tta_method == "tent" or self.tta_method in BASE_METHODS
                else (
                    "soft_reliability_all_samples"
                    if self.tta_method in RGA_METHODS
                    else "margin"
                )
            )
        )
        self.eata_e_margin = float(eata_e_margin_factor) * log_classes
        self.eata_d_margin = float(eata_d_margin)
        self.eata_fisher_alpha = float(eata_fisher_alpha)
        self.margin_threshold = float(margin_threshold)
        self.recap_soft_weight_temperature_factor = float(
            recap_soft_weight_temperature_factor
        )
        self.recap_soft_weight_temperature = (
            self.recap_soft_weight_temperature_factor * log_classes
        )
        if not 0.0 <= self.margin_threshold < 1.0:
            raise ValueError("margin_threshold must be in [0, 1)")
        if (
            not math.isfinite(self.recap_soft_weight_temperature_factor)
            or self.recap_soft_weight_temperature_factor <= 0
        ):
            raise ValueError(
                "recap_soft_weight_temperature_factor must be finite and positive"
            )
        self.use_margin_quality_gate = (
            self.use_recap
            and not self.use_eata
            and self.tta_method not in RGA_METHODS
            and self.tta_method not in BASE_METHODS
        )
        self.current_model_probs = None
        if (
            self.tta_method not in DUAL_RELIABILITY_METHODS
            and self.tta_method not in BASE_METHODS
            and self.tta_method not in RGA_METHODS
        ):
            raise ValueError(
                "DualReliabilityAdapter only supports "
                f"{DUAL_RELIABILITY_METHODS}, got {self.tta_method}"
            )
        if self.eata_d_margin < 0:
            raise ValueError("eata_d_margin must be non-negative")
        if self.eata_fisher_alpha < 0:
            raise ValueError("eata_fisher_alpha must be non-negative")

        self.eata_fishers = {}
        if self.use_eata and self.eata_fisher_alpha > 0:
            for name, param in model.named_parameters():
                if not param.requires_grad or name not in fishers:
                    continue
                fisher_item = fishers[name]
                if not isinstance(fisher_item, (tuple, list)) or len(fisher_item) < 2:
                    raise TypeError(
                        f"EATA Fisher entry must contain (importance, anchor): {name}"
                    )
                importance = torch.as_tensor(
                    fisher_item[0], device=param.device, dtype=param.dtype
                )
                anchor = torch.as_tensor(
                    fisher_item[1], device=param.device, dtype=param.dtype
                )
                if importance.shape != param.shape or anchor.shape != param.shape:
                    raise ValueError(f"EATA Fisher shape mismatch: {name}")
                self.eata_fishers[name] = (importance.detach(), anchor.detach())
            missing = [
                name
                for name, param in model.named_parameters()
                if param.requires_grad and name not in self.eata_fishers
            ]
            if missing:
                raise KeyError(
                    f"EATA Fisher entries are missing for trainable parameters: {missing[:10]}"
                )

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
            "gradient_norm_max": 0.0,
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
            "rga_batch_reliability_sum": 0.0,
            "rga_batch_reliability_square_sum": 0.0,
            "rga_batch_reliability_count": 0,
            "rga_ess_sum": 0.0,
            "rga_ess_square_sum": 0.0,
            "rga_ess_ratio_sum": 0.0,
            "rga_ess_count": 0,
            "rga_dual_consistency_sum": 0.0,
            "rga_dual_consistency_count": 0,
        }

    @torch.no_grad()
    def _update_model_probs(self, new_probs: torch.Tensor):
        """Official EATA moving average: initialize by mean, then EMA with 0.9."""
        if new_probs.numel() == 0:
            return
        batch_mean = new_probs.mean(dim=0)
        if self.current_model_probs is None:
            self.current_model_probs = batch_mean
        else:
            self.current_model_probs = (
                0.9 * self.current_model_probs + 0.1 * batch_mean
            )

    def _eata_regularization(self):
        if not self.eata_fishers or self.eata_fisher_alpha <= 0:
            return None
        regularization = None
        for name, param in self.model.named_parameters():
            if name not in self.eata_fishers:
                continue
            importance, anchor = self.eata_fishers[name]
            value = (importance * (param - anchor).square()).sum()
            regularization = value if regularization is None else regularization + value
        if regularization is None:
            return None
        return self.eata_fisher_alpha * regularization

    @torch.no_grad()
    def anchor_drift(self):
        trainable = dict(self.model.named_parameters())
        diff_sq = 0.0
        anchor_sq = 0.0
        for name, anchor in self.anchor_parameters.items():
            current = trainable[name].detach()
            anchor_value = anchor.to(device=current.device, dtype=current.dtype)
            diff_sq += float(
                (current - anchor_value).float().square().sum().cpu()
            )
            anchor_sq += float(anchor_value.float().square().sum().cpu())
        return math.sqrt(diff_sq / max(anchor_sq, 1e-12))

    @torch.enable_grad()
    def adapt(self, images: torch.Tensor):
        self.optimizer.zero_grad(set_to_none=True)
        logits = self.model(images)

        log_probability = F.log_softmax(logits, dim=1)
        probability = log_probability.exp()
        prediction_entropy = -(probability * log_probability).sum(dim=1)
        prediction_margin = top1_top2_probability_margin(
            probability.detach()
        )
        margin_mask = prediction_margin >= self.margin_threshold
        source_agreement_mask = torch.ones_like(margin_mask, dtype=torch.bool)
        source_agreement = None
        source_margin = None
        if self.source_model is not None:
            with torch.no_grad():
                source_logits = self.source_model(images)
                source_probability = F.softmax(source_logits, dim=1)
                source_prediction = source_probability.argmax(dim=1)
                current_prediction = probability.detach().argmax(dim=1)
                source_agreement_mask = source_prediction.eq(current_prediction)
                source_agreement = float(source_agreement_mask.float().mean().cpu())
                source_margin = top1_top2_probability_margin(source_probability)
                if self.confidence_gate == "source_calibrated":
                    source_agreement_mask = source_agreement_mask & (
                        prediction_margin
                        <= source_margin
                        + self.gate_margin_headroom
                    )
        relative_entropy_mask = None
        absolute_entropy_mask = None
        relative_entropy_threshold = None
        effective_entropy_threshold = None
        if self.use_eata:
            entropy_mask = prediction_entropy.detach() < self.eata_e_margin
            nonredundant_mask = torch.zeros_like(entropy_mask)
            reliable_indices = entropy_mask.nonzero(as_tuple=False).flatten()
            if reliable_indices.numel() > 0:
                reliable_probs = probability.detach().index_select(
                    0, reliable_indices
                )
                if self.current_model_probs is None:
                    nonredundant_indices = reliable_indices
                    selected_probs = reliable_probs
                else:
                    similarities = F.cosine_similarity(
                        self.current_model_probs.unsqueeze(0),
                        reliable_probs,
                        dim=1,
                    ).abs()
                    keep = similarities < self.eata_d_margin
                    nonredundant_indices = reliable_indices[keep]
                    selected_probs = reliable_probs[keep]
                nonredundant_mask[nonredundant_indices] = True
                self._update_model_probs(selected_probs)
        else:
            entropy_mask = torch.ones_like(margin_mask)
            nonredundant_mask = torch.ones_like(margin_mask)

        if self.use_recap:
            regional_entropy, regional_instability = self.proxy(logits)
            region_mask = regional_entropy.detach() < self.tau_re
            if self.use_base:
                regional_soft_weight = torch.ones_like(regional_entropy)
            else:
                regional_soft_weight = recap_region_soft_weight(
                    regional_entropy,
                    neutral_entropy=self.l0,
                    temperature=self.recap_soft_weight_temperature,
                )
        else:
            regional_entropy = None
            regional_instability = None
            region_mask = torch.ones_like(margin_mask)
            regional_soft_weight = torch.ones_like(prediction_entropy)

        if self.use_base:
            selected_mask = torch.ones_like(margin_mask, dtype=torch.bool)
        elif self.use_margin_quality_gate:
            selected_mask = margin_mask & nonredundant_mask & source_agreement_mask
        else:
            selected_mask = entropy_mask & nonredundant_mask
        selected_indices = selected_mask.nonzero(as_tuple=False).flatten()

        loss_value = 0.0
        recap_loss_value = 0.0
        eata_regularization_value = 0.0
        updated = False
        nonfinite = False
        gradient_norm_before_clip = None
        gradient_clipped = False
        nonfinite_gradient = False
        if selected_indices.numel() > 0:
            if self.use_recap:
                selected_re = regional_entropy.index_select(0, selected_indices)
                selected_weights = regional_soft_weight.index_select(
                    0, selected_indices
                )
                recap_loss = (selected_weights * selected_re).mean()
                loss = recap_loss
                recap_loss_value = float(recap_loss.detach().cpu())
            else:
                loss = prediction_entropy.index_select(
                    0, selected_indices
                ).mean()
            regularization = self._eata_regularization()
            if regularization is not None:
                loss = loss + regularization
                eata_regularization_value = float(regularization.detach().cpu())
            if torch.isfinite(loss):
                loss_value = float(loss.detach().cpu())
                loss.backward()
                if self.dsr_grad_clip_norm is not None:
                    gradient_norm_tensor = torch.nn.utils.clip_grad_norm_(
                        self._trainable_parameters,
                        max_norm=self.dsr_grad_clip_norm,
                        error_if_nonfinite=False,
                    )
                    gradient_norm_before_clip = float(
                        gradient_norm_tensor.detach().cpu()
                    )
                    if math.isfinite(gradient_norm_before_clip):
                        gradient_clipped = (
                            gradient_norm_before_clip
                            > self.dsr_grad_clip_norm
                        )
                        self.optimizer.step()
                        updated = True
                    else:
                        nonfinite_gradient = True
                        nonfinite = True
                else:
                    gradient_squares = [
                        param.grad.detach().float().square().sum()
                        for param in self._trainable_parameters
                        if param.grad is not None
                    ]
                    gradient_norm_before_clip = float(
                        torch.sqrt(
                            torch.stack(gradient_squares).sum()
                        ).cpu()
                    ) if gradient_squares else 0.0
                    self.optimizer.step()
                    updated = True
            else:
                nonfinite = True
            self.optimizer.zero_grad(set_to_none=True)

        recovered = self.restorer.step()
        selected_weight_values = regional_soft_weight.index_select(
            0, selected_indices
        ).detach()
        if selected_weight_values.numel() > 0:
            weight_min = float(selected_weight_values.min().cpu())
            weight_max = float(selected_weight_values.max().cpu())
        else:
            weight_min = 0.0
            weight_max = 0.0
        batch_stats = {
            "samples": int(images.size(0)),
            "quality_passed": int(
                selected_mask.sum().item()
                if self.use_margin_quality_gate
                else margin_mask.sum().item()
            ),
            "margin_passed": int(margin_mask.sum().item()),
            "margin_sum": float(prediction_margin.sum().item()),
            "entropy_passed": int(entropy_mask.sum().item()),
            "relative_entropy_passed": (
                int(relative_entropy_mask.sum().item())
                if relative_entropy_mask is not None
                else 0
            ),
            "absolute_entropy_passed": (
                int(absolute_entropy_mask.sum().item())
                if absolute_entropy_mask is not None
                else 0
            ),
            "relative_entropy_threshold": relative_entropy_threshold,
            "effective_entropy_threshold": effective_entropy_threshold,
            "nonredundant": int(nonredundant_mask.sum().item()),
            "region_reliable": int(region_mask.sum().item()),
            "selected": int(selected_indices.numel()),
            "recap_soft_weight_sum": float(selected_weight_values.sum().cpu()),
            "recap_soft_weight_square_sum": float(
                selected_weight_values.square().sum().cpu()
            ),
            "recap_soft_weight_count": int(selected_weight_values.numel()),
            "recap_soft_weight_min": weight_min,
            "recap_soft_weight_max": weight_max,
            "gradient_norm_before_clip": gradient_norm_before_clip,
            "gradient_clipped": bool(gradient_clipped),
            "nonfinite_gradient": bool(nonfinite_gradient),
            "updated": updated,
            "nonfinite": nonfinite,
            "loss": loss_value,
            "prediction_entropy": float(prediction_entropy.mean().detach().cpu()),
            "source_agreement": source_agreement,
            "source_margin": (
                float(source_margin.mean().detach().cpu())
                if source_margin is not None
                else None
            ),
            "anchor_drift": self.anchor_drift(),
            "recap_loss": recap_loss_value,
            "eata_regularization": eata_regularization_value,
            "recovered": recovered,
        }
        self.counters["samples"] += batch_stats["samples"]
        self.counters["quality_passed"] += batch_stats["quality_passed"]
        self.counters["margin_passed"] += batch_stats["margin_passed"]
        self.counters["margin_sum"] += batch_stats["margin_sum"]
        self.counters["entropy_passed"] += batch_stats["entropy_passed"]
        self.counters["relative_entropy_passed"] += batch_stats[
            "relative_entropy_passed"
        ]
        self.counters["absolute_entropy_passed"] += batch_stats[
            "absolute_entropy_passed"
        ]
        self.counters["nonredundant"] += batch_stats["nonredundant"]
        self.counters["region_reliable"] += batch_stats["region_reliable"]
        self.counters["selected"] += batch_stats["selected"]
        self.counters["recap_soft_weight_sum"] += batch_stats[
            "recap_soft_weight_sum"
        ]
        self.counters["recap_soft_weight_square_sum"] += batch_stats[
            "recap_soft_weight_square_sum"
        ]
        self.counters["recap_soft_weight_count"] += batch_stats[
            "recap_soft_weight_count"
        ]
        if gradient_norm_before_clip is not None and math.isfinite(
            gradient_norm_before_clip
        ):
            self.counters["gradient_norm_sum"] += gradient_norm_before_clip
            self.counters["gradient_norm_max"] = max(
                self.counters["gradient_norm_max"],
                gradient_norm_before_clip,
            )
            self.counters["gradient_norm_batches"] += 1
        self.counters["gradient_clipped_batches"] += int(gradient_clipped)
        self.counters["nonfinite_gradient_batches"] += int(nonfinite_gradient)
        self.counters["batches"] += 1
        self.counters["updated_batches"] += int(updated)
        self.counters["skipped_batches"] += int(not updated)
        self.counters["nonfinite_batches"] += int(nonfinite)
        self.counters["loss_sum"] += batch_stats["loss"]
        self.counters["recap_loss_sum"] += batch_stats["recap_loss"]
        self.counters["eata_regularization_sum"] += batch_stats[
            "eata_regularization"
        ]
        return logits.detach(), batch_stats

    @torch.inference_mode()
    def predict(self, images: torch.Tensor):
        return self.model(images)

    def finalize(self):
        return False
