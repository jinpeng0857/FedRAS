"""DLRS (Dynamic Local Reliability Scaling) adapter with Bounded Normalization, Skip-Update Protection, and Anchor Disagreement."""

from copy import deepcopy
import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from sparse_restore import PeriodicSparseRestorer
from tests.tta_baselines import DeYOAdapter
from adapters.base import (
    DLRS_METHODS,
    DLRS_REGION_SCALE_MAX,
    DLRS_REGION_SCALE_MIN,
    DLRS_RELIABILITY_ALPHA,
    FSLR_GRAD_CLIP_NORM,
    REGION_SCALE,
)
from adapters.dual_reliability import (
    DualReliabilityAdapter,
    recap_region_soft_weight,
    top1_top2_probability_margin,
)


def create_composite_weak_perturbation(images: torch.Tensor) -> torch.Tensor:
    """Create a lightweight composite weak view independent of specific corruption types.
    
    Combines:
    1. Mild spatial smoothing: 0.85 * x + 0.15 * AvgPool2d(3, stride=1, padding=1)
    2. Mild brightness/contrast variation: x * (1 + delta_b) + delta_c
    3. Tiny Gaussian noise: delta_n ~ N(0, 0.015^2)
    """
    with torch.no_grad():
        smoothed = F.avg_pool2d(images, kernel_size=3, stride=1, padding=1)
        x_blend = 0.85 * images + 0.15 * smoothed
        batch_size = images.size(0)
        device = images.device
        dtype = images.dtype
        scale = 1.0 + (torch.rand(batch_size, 1, 1, 1, device=device, dtype=dtype) - 0.5) * 0.08
        shift = (torch.rand(batch_size, 1, 1, 1, device=device, dtype=dtype) - 0.5) * 0.04
        x_scaled = x_blend * scale + shift
        noise = torch.randn_like(images) * 0.015
        return x_scaled + noise


def prediction_js_divergence(
    first_probability: torch.Tensor,
    second_probability: torch.Tensor,
):
    """Return per-sample Jensen-Shannon divergence in nats."""
    if first_probability.shape != second_probability.shape:
        raise ValueError("probability tensors must have matching shapes")
    first = first_probability.detach().float().clamp_min(1e-8)
    second = second_probability.detach().float().clamp_min(1e-8)
    midpoint = 0.5 * (first + second)
    return 0.5 * (
        (first * (first.log() - midpoint.log())).sum(dim=1)
        + (second * (second.log() - midpoint.log())).sum(dim=1)
    )


def dual_perturbation_consistency(
    original_probability: torch.Tensor,
    perturbed_probability: torch.Tensor,
):
    """Combine distributional and argmax consistency into [0, 1]."""
    js_divergence = prediction_js_divergence(
        original_probability, perturbed_probability
    )
    distribution_consistency = (
        1.0 - js_divergence / math.log(2.0)
    ).clamp(0.0, 1.0)
    label_consistency = (
        original_probability.argmax(dim=1)
        == perturbed_probability.argmax(dim=1)
    ).float()
    return 0.5 * (distribution_consistency + label_consistency)


def dlrs_sample_reliability(
    prediction_entropy: torch.Tensor,
    plpd: torch.Tensor,
    probability: torch.Tensor,
    plpd_threshold: float,
    reliability_alpha: float,
    hybrid_threshold: Optional[float] = None,
    cue_ablation: str = "full",
):
    """Detached DLRS reliability from entropy, PLPD, and class margin.

    Entropy and PLPD form the DeYO-derived sample score. The Top-1/Top-2
    probability margin explicitly represents distance from a decision boundary.
    The returned score is bounded in [0, 1] and does not create a gradient path
    through the sample-aware region controller.
    """
    if probability.ndim != 2 or probability.size(1) < 2:
        raise ValueError("probability must have shape [batch, classes>=2]")
    if prediction_entropy.shape != plpd.shape:
        raise ValueError("prediction_entropy and plpd must have matching shape")
    if prediction_entropy.numel() != probability.size(0):
        raise ValueError("reliability inputs must share the same batch size")
    if not 0.0 <= float(reliability_alpha) <= 1.0:
        raise ValueError("reliability_alpha must be in [0, 1]")

    detached_entropy = prediction_entropy.detach().float()
    detached_plpd = plpd.detach().float()
    detached_probability = probability.detach().float()
    log_classes = math.log(float(detached_probability.size(1)))
    entropy_reliability = (1.0 - detached_entropy / log_classes).clamp(0.0, 1.0)
    plpd_span = max(1.0 - float(plpd_threshold), 1e-6)
    plpd_reliability = (
        (detached_plpd - float(plpd_threshold)) / plpd_span
    ).clamp(0.0, 1.0)
    prediction_margin = top1_top2_probability_margin(detached_probability)
    if cue_ablation not in {
        "full",
        "minus_entropy",
        "minus_patch_shuffle",
        "minus_margin",
        "reduced",
    }:
        raise ValueError(
            "cue_ablation must be full, minus_entropy, minus_patch_shuffle, "
            "minus_margin, or reduced for sample-side reliability"
        )

    # The original score assigns alpha/2 to entropy and PLPD and 1-alpha to
    # margin. Single-cue ablations renormalize the surviving coefficients.
    # The integrated reduced mechanism deliberately retains PLPD only.
    coefficients = {
        "entropy": float(reliability_alpha) * 0.5,
        "patch_shuffle": float(reliability_alpha) * 0.5,
        "margin": 1.0 - float(reliability_alpha),
    }
    cue_values = {
        "entropy": entropy_reliability,
        "patch_shuffle": plpd_reliability,
        "margin": prediction_margin,
    }
    removed = {
        "full": set(),
        "minus_entropy": {"entropy"},
        "minus_patch_shuffle": {"patch_shuffle"},
        "minus_margin": {"margin"},
        "reduced": {"entropy", "margin"},
    }[cue_ablation]
    active = [cue for cue in cue_values if cue not in removed]
    total_coefficient = sum(coefficients[cue] for cue in active)
    reliability = (
        sum(
            (coefficients[cue] / total_coefficient) * cue_values[cue]
            for cue in active
        ).clamp(0.0, 1.0)
        if total_coefficient > 0.0
        else torch.ones_like(prediction_margin)
    )

    if hybrid_threshold is not None and hybrid_threshold > 0:
        hard_zero_mask = reliability < float(hybrid_threshold)
        reliability = reliability.masked_fill(hard_zero_mask, 0.0)

    return reliability, prediction_margin


def dlrs_joint_reliability_weight(
    sample_reliability: torch.Tensor,
    regional_reliability: torch.Tensor,
):
    """Combine sample-level and regional reliability without hard rejection."""
    if sample_reliability.shape != regional_reliability.shape:
        raise ValueError(
            "sample_reliability and regional_reliability must have matching shape"
        )
    detached_sample_reliability = sample_reliability.detach().to(
        device=regional_reliability.device,
        dtype=regional_reliability.dtype,
    )
    if (
        not torch.isfinite(detached_sample_reliability).all()
        or not torch.isfinite(regional_reliability).all()
        or (detached_sample_reliability < 0).any()
        or (regional_reliability < 0).any()
    ):
        raise ValueError("DLRS reliability weights must be finite and non-negative")
    return detached_sample_reliability * regional_reliability


class DLRSAdapter(DualReliabilityAdapter):
    """Soft sample reliability with reliability-controlled ReCAP regions.

    Entropy, PLPD, and the Top-1/Top-2 prediction margin form a continuous
    sample reliability. Reliability controls a detached sample-wise ReCAP
    region scale and is multiplied by regional reliability in the final
    optimization weight.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        feature_variance: torch.Tensor,
        restorer: PeriodicSparseRestorer,
        fishers,
        tta_method: str = "dlrs",
        deyo_entropy_margin_factor: float = 0.5,
        deyo_plpd_threshold: float = 0.2,
        deyo_patch_grid: int = 4,
        region_scale_min: float = DLRS_REGION_SCALE_MIN,
        region_scale_max: float = DLRS_REGION_SCALE_MAX,
        reliability_alpha: float = DLRS_RELIABILITY_ALPHA,
        sample_reliability_mode: str = "full",
        normalize_by_weight: bool = False,
        c_max: float = 2.0,
        skip_update_tau_s: float = 0.0,
        skip_update_tau_ess: float = 0.0,
        weight_mode: str = "product",
        dual_perturbation: bool = False,
        consistency_weight: float = 0.5,
        anchor_disagreement_diagnostic: bool = False,
        disable_fslr_reliability_gate: bool = False,
        fslr_grad_clip_norm: float = FSLR_GRAD_CLIP_NORM,
        early_exit_enabled: bool = False,
        early_exit_margin_threshold: float = 0.80,
        fast_perturbation: bool = False,
        hybrid_threshold: Optional[float] = None,
        anchor_rectify_beta: float = 0.0,
        view_consistency: bool = True,
        view_alpha: float = 0.80,
        view_js_scale: float = 1.0,
        rga_ablation_variant: str = "full",
        rga_cue_ablation: str = "full",
    ):
        if tta_method not in DLRS_METHODS:
            raise ValueError(
                f"DLRSAdapter requires one of {DLRS_METHODS}, got {tta_method}"
            )
        super().__init__(
            model=model,
            optimizer=optimizer,
            feature_variance=feature_variance,
            restorer=restorer,
            fishers=fishers,
            tta_method=tta_method,
            fslr_grad_clip_norm=fslr_grad_clip_norm,
        )
        self.hybrid_threshold = hybrid_threshold
        if self.hybrid_threshold is not None and float(self.hybrid_threshold) > 0.0:
            self.sample_filter = f"hybrid_soft_hard_tau{float(self.hybrid_threshold):g}"
        else:
            self.sample_filter = "soft_reliability_all_samples"
        self.anchor_rectify_beta = float(anchor_rectify_beta)
        if not 0.0 <= self.anchor_rectify_beta < 1.0:
            raise ValueError("anchor_rectify_beta must be in [0.0, 1.0)")
        self.deyo_entropy_margin_factor = float(deyo_entropy_margin_factor)
        self.deyo_entropy_margin = (
            self.deyo_entropy_margin_factor
            * math.log(float(self.proxy.num_classes))
        )
        self.deyo_plpd_threshold = float(deyo_plpd_threshold)
        self.patch_grid = int(deyo_patch_grid)
        self.region_scale_min = float(region_scale_min)
        self.region_scale_max = float(region_scale_max)
        self.reliability_alpha = float(reliability_alpha)
        self.sample_reliability_mode = str(sample_reliability_mode)
        self.normalize_by_weight = bool(normalize_by_weight)
        self.c_max = float(c_max)
        self.skip_update_tau_s = float(skip_update_tau_s)
        self.skip_update_tau_ess = float(skip_update_tau_ess)
        self.weight_mode = str(weight_mode)
        self.dual_perturbation = bool(dual_perturbation)
        self.consistency_weight = float(consistency_weight)
        self.anchor_disagreement_diagnostic = bool(
            anchor_disagreement_diagnostic
        )
        self.disable_fslr_reliability_gate = bool(
            disable_fslr_reliability_gate
        )
        self.early_exit_enabled = bool(early_exit_enabled)
        self.early_exit_margin_threshold = float(early_exit_margin_threshold)
        self.fast_perturbation = bool(fast_perturbation)
        self.view_consistency = bool(view_consistency)
        self.view_alpha = float(view_alpha)
        self.view_js_scale = float(view_js_scale)
        self.rga_ablation_variant = str(rga_ablation_variant)
        if self.rga_ablation_variant not in {
            "full",
            "uniform_recap",
            "regional_only",
            "anchor_only",
            "regional_anchor",
        }:
            raise ValueError(
                "rga_ablation_variant must be one of full, uniform_recap, "
                "regional_only, anchor_only, regional_anchor"
            )
        self.rga_cue_ablation = str(rga_cue_ablation)
        if self.rga_cue_ablation not in {
            "full",
            "minus_entropy",
            "minus_patch_shuffle",
            "minus_margin",
            "minus_anchor",
            "minus_weak_view",
            "reduced",
        }:
            raise ValueError(
                "rga_cue_ablation must be full, minus_entropy, "
                "minus_patch_shuffle, minus_margin, minus_anchor, "
                "minus_weak_view, or reduced"
            )
        # The integrated reduced variant removes weak-view consistency from
        # both the RGA weights and the reliability signal passed to DSR.
        # Keep the CLI's official setting intact for other variants, but expose
        # the effective state here so runtime metadata and diagnostics are clear.
        if self.rga_cue_ablation == "reduced":
            self.view_consistency = False

        if self.c_max < 1.0:
            raise ValueError("c_max must be >= 1.0")
        if self.weight_mode not in ("product", "sample_only", "region_only", "uniform"):
            raise ValueError(f"unsupported DLRS weight mode: {self.weight_mode}")
        if (
            not math.isfinite(self.consistency_weight)
            or not 0.0 <= self.consistency_weight <= 1.0
        ):
            raise ValueError("consistency_weight must be finite and in [0, 1]")
        if not 0.0 <= self.view_alpha <= 1.0:
            raise ValueError("view_alpha must be in [0, 1]")
        if self.view_js_scale < 0.0:
            raise ValueError("view_js_scale must be non-negative")
        if self.patch_grid <= 0:
            raise ValueError("deyo_patch_grid must be positive")
        if not math.isfinite(self.deyo_plpd_threshold):
            raise ValueError("deyo_plpd_threshold must be finite")
        if (
            not math.isfinite(self.region_scale_min)
            or self.region_scale_min <= 0
        ):
            raise ValueError("region_scale_min must be finite and positive")
        if (
            not math.isfinite(self.region_scale_max)
            or self.region_scale_max < self.region_scale_min
        ):
            raise ValueError(
                "region_scale_max must be finite and >= region_scale_min"
            )
        if not 0.0 <= self.reliability_alpha <= 1.0:
            raise ValueError("reliability_alpha must be in [0, 1]")
        if self.sample_reliability_mode not in ("full", "entropy"):
            raise ValueError(
                "sample_reliability_mode must be either 'full' or 'entropy'"
            )

        # Maintain frozen Lookahead Anchor model when required
        self.needs_anchor = bool(
            (
                self.anchor_rectify_beta > 0.0
                and self.rga_cue_ablation != "minus_anchor"
            )
            or self.rga_ablation_variant in {"anchor_only", "regional_anchor"}
            or self.anchor_disagreement_diagnostic
            or (self.use_fslr and getattr(self.restorer, "event_trigger", False))
        )
        if self.needs_anchor:
            self.anchor_model = deepcopy(model)
            self.anchor_model.eval()
            self.anchor_model.requires_grad_(False)
        else:
            self.anchor_model = None

        self.counters.update(
            {
                "deyo_entropy_passed": 0,
                "plpd_passed": 0,
                "dlrs_selected": 0,
                "dlrs_reliability_sum": 0.0,
                "dlrs_region_scale_sum": 0.0,
                "dlrs_reliability_count": 0,
                "dlrs_joint_weight_sum": 0.0,
                "dlrs_joint_weight_square_sum": 0.0,
                "dlrs_joint_weight_count": 0,
                "effective_sample_size_sum": 0.0,
                "effective_sample_size_batches": 0,
                "recovery_credit_sum": 0.0,
                "recovery_credit_count": 0,
                "recovery_authorized_batches": 0,
                "early_exit_samples": 0,
                "skipped_low_quality_batches": 0,
                "anchor_disagreement_sum": 0.0,
                "anchor_disagreement_batches": 0,
                "r_view_sum": 0.0,
                "r_view_count": 0,
                "forward_passes_extra": 0,
            }
        )

    def _shuffle_patches(self, images: torch.Tensor) -> torch.Tensor:
        """Reuse the tested DeYO per-sample patch permutation."""
        return DeYOAdapter._shuffle_patches(self, images)

    @torch.enable_grad()
    def adapt(self, images: torch.Tensor):
        self.optimizer.zero_grad(set_to_none=True)
        logits = self.model(images)
        log_probability = F.log_softmax(logits, dim=1)
        probability = log_probability.exp()
        prediction_entropy = -(probability * log_probability).sum(dim=1)
        prediction_margin = top1_top2_probability_margin(probability.detach())

        entropy_mask = prediction_entropy.detach() < self.deyo_entropy_margin

        # Compute Lookahead Anchor Disagreement A_t
        anchor_disagreement = None
        disagree_vector = None
        if self.anchor_model is not None:
            with torch.no_grad():
                anchor_logits = self.anchor_model(images)
                anchor_prediction = anchor_logits.argmax(dim=1)
                current_prediction = logits.detach().argmax(dim=1)
                disagree_vector = current_prediction.ne(anchor_prediction)
                anchor_disagreement = float(
                    disagree_vector.float().mean().cpu()
                )

        # Fast-DLRS Early Exit for high-confidence samples
        early_exit_mask = None
        if self.early_exit_enabled:
            early_exit_mask = (
                (prediction_entropy.detach() < 0.20 * math.log(float(self.proxy.num_classes)))
                & (prediction_margin >= self.early_exit_margin_threshold)
            )

        with torch.no_grad():
            if self.fast_perturbation:
                perturbed_images = F.dropout2d(images.detach(), p=0.1, training=True)
                destroyed_logits = self.model(perturbed_images)
            elif early_exit_mask is not None and early_exit_mask.all():
                destroyed_logits = logits.detach()
            else:
                destroyed_logits = self.model(
                    self._shuffle_patches(images.detach())
                )

        dual_consistency = None
        if self.dual_perturbation:
            with torch.no_grad():
                weak_logits = self.model(images.detach().flip(dims=[3]))
            dual_consistency = dual_perturbation_consistency(
                probability.detach(),
                weak_logits.softmax(dim=1),
            )
        detached_probability = probability.detach()
        pseudo_labels = detached_probability.argmax(1, keepdim=True)
        original_confidence = detached_probability.gather(
            1, pseudo_labels
        ).squeeze(1)
        destroyed_confidence = destroyed_logits.softmax(1).gather(
            1, pseudo_labels
        ).squeeze(1)
        plpd = original_confidence - destroyed_confidence
        plpd_mask = plpd.detach() > self.deyo_plpd_threshold

        def _base_sample_reliability(cue_ablation: str):
            if cue_ablation == "reduced":
                reliability, _ = dlrs_sample_reliability(
                    prediction_entropy=prediction_entropy,
                    plpd=plpd,
                    probability=probability,
                    plpd_threshold=self.deyo_plpd_threshold,
                    reliability_alpha=self.reliability_alpha,
                    hybrid_threshold=self.hybrid_threshold,
                    cue_ablation="reduced",
                )
            elif self.sample_reliability_mode == "entropy":
                if cue_ablation == "minus_entropy":
                    reliability = torch.ones_like(prediction_entropy).float()
                else:
                    reliability = (
                        1.0
                        - prediction_entropy.detach().float()
                        / math.log(float(self.proxy.num_classes))
                    ).clamp(0.0, 1.0)
            else:
                reliability, _ = dlrs_sample_reliability(
                    prediction_entropy=prediction_entropy,
                    plpd=plpd,
                    probability=probability,
                    plpd_threshold=self.deyo_plpd_threshold,
                    reliability_alpha=self.reliability_alpha,
                    hybrid_threshold=self.hybrid_threshold,
                    cue_ablation=(
                        cue_ablation
                        if cue_ablation
                        in {"full", "minus_entropy", "minus_patch_shuffle", "minus_margin"}
                        else "full"
                    ),
                )
            return reliability

        # Single-cue diagnostic rows preserve the official full signal for
        # DSR/FSLR. The integrated reduced mechanism sends its own reliability
        # signal to both RGA and DSR/FSLR.
        integrated_reduced = self.rga_cue_ablation == "reduced"
        if integrated_reduced:
            reduced_sample_reliability = _base_sample_reliability("reduced")
            full_sample_reliability = reduced_sample_reliability
            ablated_sample_reliability = reduced_sample_reliability
        else:
            full_sample_reliability = _base_sample_reliability("full")
            ablated_sample_reliability = _base_sample_reliability(
                self.rga_cue_ablation
            )

        # Compute the weak-view cue once. Its use in the downstream full signal
        # is retained even when the RGA update signal excludes this cue.
        view_factor = torch.ones_like(full_sample_reliability)
        if self.view_consistency and not integrated_reduced:
            with torch.no_grad():
                weak_view_images = create_composite_weak_perturbation(images.detach())
                weak_view_logits = self.model(weak_view_images)
                weak_view_prob = weak_view_logits.softmax(dim=1)
            js_div = prediction_js_divergence(detached_probability, weak_view_prob)
            r_view = torch.exp(-self.view_js_scale * js_div).clamp(0.0, 1.0)
            view_factor = self.view_alpha + (1.0 - self.view_alpha) * r_view
            self.counters["r_view_sum"] += float(r_view.sum().item())
            self.counters["r_view_count"] += int(r_view.numel())
            self.counters["forward_passes_extra"] += int(images.size(0))

        def _compose_auxiliary_cues(base_reliability: torch.Tensor, *, ablated: bool):
            reliability = base_reliability
            if (
                disagree_vector is not None
                and self.anchor_rectify_beta > 0.0
                and (not ablated or self.rga_cue_ablation != "minus_anchor")
            ):
                rectify_factor = 1.0 - self.anchor_rectify_beta * disagree_vector.float()
                reliability = reliability * rectify_factor
            if (
                self.view_consistency
                and not integrated_reduced
                and (not ablated or self.rga_cue_ablation != "minus_weak_view")
            ):
                reliability = reliability * view_factor
            if dual_consistency is not None:
                reliability = (
                    (1.0 - self.consistency_weight) * reliability
                    + self.consistency_weight * dual_consistency
                ).clamp(0.0, 1.0)
            return reliability

        recovery_reliability = _compose_auxiliary_cues(
            full_sample_reliability, ablated=integrated_reduced
        ).detach()
        sample_reliability = recovery_reliability
        rga_cue_reliability = (
            recovery_reliability
            if integrated_reduced
            else _compose_auxiliary_cues(
                ablated_sample_reliability, ablated=True
            )
        )

        if early_exit_mask is not None and early_exit_mask.any():
            sample_reliability[early_exit_mask] = 1.0
            self.counters["early_exit_samples"] += int(early_exit_mask.sum().item())

        if self.rga_ablation_variant == "full":
            rga_sample_reliability = rga_cue_reliability
        elif self.rga_ablation_variant in {"anchor_only", "regional_anchor"}:
            if disagree_vector is None:
                raise RuntimeError(
                    "anchor-based RGA variants require a frozen anchor model"
                )
            rga_sample_reliability = (
                1.0 - self.anchor_rectify_beta * disagree_vector.float()
            ).clamp(0.0, 1.0)
        else:
            rga_sample_reliability = torch.ones_like(sample_reliability)

        if self.rga_ablation_variant != "full":
            # Keep the ReCAP region scale fixed for the simplified rows so that
            # their only RGA factors are the explicitly named sample weights.
            adaptive_region_scale = torch.full_like(
                rga_sample_reliability, REGION_SCALE
            )
        elif self.region_scale_min == self.region_scale_max:
            adaptive_region_scale = torch.full_like(
                rga_sample_reliability, self.region_scale_max
            )
        elif self.weight_mode == "product":
            adaptive_region_scale = self.region_scale_min + (
                self.region_scale_max - self.region_scale_min
            ) * rga_sample_reliability
        else:
            adaptive_region_scale = torch.full_like(
                rga_sample_reliability, REGION_SCALE
            )
        regional_entropy, regional_instability = self.proxy(
            logits,
            region_scale=adaptive_region_scale,
        )
        region_mask = regional_entropy.detach() < self.tau_re
        if self.rga_ablation_variant in {"uniform_recap", "anchor_only"}:
            regional_soft_weight = torch.ones_like(rga_sample_reliability)
        elif self.weight_mode in ("sample_only", "uniform"):
            regional_soft_weight = torch.ones_like(sample_reliability)
        else:
            regional_soft_weight = recap_region_soft_weight(
                regional_entropy,
                neutral_entropy=self.l0,
                temperature=self.recap_soft_weight_temperature,
            )
        if self.rga_ablation_variant != "full":
            selected_mask = torch.ones_like(entropy_mask, dtype=torch.bool)
            selected_indices = torch.arange(images.size(0), device=images.device)
        elif self.hybrid_threshold is not None and self.hybrid_threshold > 0:
            selected_mask = rga_sample_reliability > 0.0
            selected_indices = selected_mask.nonzero(as_tuple=False).flatten()
        else:
            selected_mask = torch.ones_like(entropy_mask, dtype=torch.bool)
            selected_indices = torch.arange(images.size(0), device=images.device)

        loss_value = 0.0
        recap_loss_value = 0.0
        updated = False
        nonfinite = False
        gradient_norm_before_clip = None
        gradient_clipped = False
        nonfinite_gradient = False
        selected_soft_weights = regional_soft_weight.index_select(
            0, selected_indices
        )
        selected_reliability = rga_sample_reliability.index_select(
            0, selected_indices
        )
        if self.rga_ablation_variant == "uniform_recap":
            selected_joint_weights = torch.ones_like(selected_reliability)
        elif self.rga_ablation_variant == "regional_only":
            selected_joint_weights = selected_soft_weights
        elif self.rga_ablation_variant in {"anchor_only", "regional_anchor"}:
            if self.rga_ablation_variant == "anchor_only":
                selected_joint_weights = selected_reliability
            else:
                selected_joint_weights = dlrs_joint_reliability_weight(
                    selected_reliability,
                    selected_soft_weights,
                )
        elif self.weight_mode == "uniform":
            selected_joint_weights = torch.ones_like(selected_reliability)
        elif self.weight_mode == "sample_only":
            selected_joint_weights = selected_reliability
        elif self.weight_mode == "region_only":
            selected_joint_weights = selected_soft_weights
        elif self.weight_mode == "product":
            selected_joint_weights = dlrs_joint_reliability_weight(
                selected_reliability,
                selected_soft_weights,
            )
        else:
            raise ValueError(f"unsupported DLRS weight mode: {self.weight_mode}")
        selected_region_scale = adaptive_region_scale.index_select(
            0, selected_indices
        )

        batch_size_f = float(images.size(0))
        weight_sum = selected_joint_weights.sum()
        joint_weight_square_sum = float(
            selected_joint_weights.detach().square().sum().cpu()
        )
        joint_weight_sum_val = float(weight_sum.detach().cpu())
        effective_sample_size = (
            joint_weight_sum_val * joint_weight_sum_val
            / max(joint_weight_square_sum, 1e-12)
        )

        sb_ratio = float(joint_weight_sum_val / max(batch_size_f, 1.0))
        ess_ratio = float(effective_sample_size / max(batch_size_f, 1.0))
        recovery_credit = float(
            recovery_reliability.mean().clamp(0.0, 1.0).cpu()
        )

        # Check skip-update safety protection for low quality batches
        should_skip_update = (
            (sb_ratio < self.skip_update_tau_s)
            or (ess_ratio < self.skip_update_tau_ess)
        )

        if selected_indices.numel() > 0 and not should_skip_update:
            selected_re = regional_entropy.index_select(0, selected_indices)
            selected_ri = regional_instability.index_select(
                0, selected_indices
            )
            selected_terms = selected_re + 0.0 * selected_ri
            if self.normalize_by_weight:
                # Bounded Scale Normalization without forced >= 1.0 lower bound:
                # scale = min(c_max, B / (S_t + eps))
                scale = min(self.c_max, batch_size_f / max(joint_weight_sum_val, 1e-12))
                recap_loss = scale * (selected_joint_weights * selected_terms).sum() / batch_size_f
            else:
                recap_loss = (
                    selected_joint_weights * selected_terms
                ).mean()
            if torch.isfinite(recap_loss):
                loss_value = float(recap_loss.detach().cpu())
                recap_loss_value = loss_value
                recap_loss.backward()
                if self.fslr_grad_clip_norm is not None:
                    gradient_norm_tensor = torch.nn.utils.clip_grad_norm_(
                        self._trainable_parameters,
                        max_norm=self.fslr_grad_clip_norm,
                        error_if_nonfinite=False,
                    )
                    gradient_norm_before_clip = float(
                        gradient_norm_tensor.detach().cpu()
                    )
                    if math.isfinite(gradient_norm_before_clip):
                        gradient_clipped = (
                            gradient_norm_before_clip
                            > self.fslr_grad_clip_norm
                        )
                        if self.use_fslr:
                            is_healthy = (
                                (recovery_credit >= getattr(self.restorer, "event_tau_r", 0.40))
                                and (anchor_disagreement is None or anchor_disagreement <= getattr(self.restorer, "event_tau_a", 0.50))
                            )
                            self.restorer.update_online_gradients(healthy=is_healthy)
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
                    if self.use_fslr:
                        is_healthy = (
                            (recovery_credit >= getattr(self.restorer, "event_tau_r", 0.40))
                            and (anchor_disagreement is None or anchor_disagreement <= getattr(self.restorer, "event_tau_a", 0.50))
                        )
                        self.restorer.update_online_gradients(healthy=is_healthy)
                    self.optimizer.step()
                    updated = True
            else:
                nonfinite = True
            self.optimizer.zero_grad(set_to_none=True)
        elif should_skip_update:
            self.counters["skipped_low_quality_batches"] += 1

        sample_entropy_mean = float(prediction_entropy.detach().mean().cpu())
        batch_avg_prob = probability.detach().mean(dim=0).clamp_min(1e-12)
        marginal_entropy = float(-(batch_avg_prob * batch_avg_prob.log()).sum().cpu())
        if self.use_fslr:
            recovered = self.restorer.step(
                reliability=recovery_credit,
                entropy=marginal_entropy,
                anchor_disagreement=anchor_disagreement,
                gradient_clipped=gradient_clipped,
                ess_ratio=float(effective_sample_size / batch_size_f),
            )
            recovery_threshold = getattr(self.restorer, "reliability_threshold", None)
            recovery_authorized = (
                True if getattr(self.restorer, "disable_reliability_gate", False)
                else (recovery_credit >= recovery_threshold if recovery_threshold is not None else None)
            )
        else:
            recovery_threshold = None
            recovery_authorized = None
            recovered = self.restorer.step()

        selected_weight_values = selected_soft_weights.detach()
        selected_joint_weight_values = selected_joint_weights.detach()
        if selected_weight_values.numel() > 0:
            weight_min = float(selected_weight_values.min().cpu())
            weight_max = float(selected_weight_values.max().cpu())
        else:
            weight_min = 0.0
            weight_max = 0.0
        if selected_joint_weight_values.numel() > 0:
            joint_weight_min = float(selected_joint_weight_values.min().cpu())
            joint_weight_max = float(selected_joint_weight_values.max().cpu())
        else:
            joint_weight_min = 0.0
            joint_weight_max = 0.0
        if selected_reliability.numel() > 0:
            reliability_min = float(selected_reliability.min().cpu())
            reliability_max = float(selected_reliability.max().cpu())
            region_scale_min = float(selected_region_scale.min().cpu())
            region_scale_max = float(selected_region_scale.max().cpu())
        else:
            reliability_min = 0.0
            reliability_max = 0.0
            region_scale_min = 0.0
            region_scale_max = 0.0

        # The evaluator consumes these transient vectors to verify whether
        # RGA assigns lower update weight to incorrect predictions. They are
        # intentionally kept out of the serialized result payload.
        rga_joint_weight_vector = torch.zeros_like(rga_sample_reliability)
        rga_joint_weight_vector[selected_indices] = selected_joint_weight_values
        rga_applied_weight_vector = (
            rga_joint_weight_vector
            if updated
            else torch.zeros_like(rga_joint_weight_vector)
        )
        batch_stats = {
            "samples": int(images.size(0)),
            "batch_entropy": marginal_entropy,
            "sample_entropy_mean": sample_entropy_mean,
            "batch_mean_reliability": recovery_credit,
            "anchor_disagreement": anchor_disagreement,
            "sb_ratio": sb_ratio,
            "ess_ratio": ess_ratio,
            "skipped_low_quality": should_skip_update,
            "quality_passed": int(selected_mask.sum().item()),
            "margin_passed": int((prediction_margin > 0).sum().item()),
            "margin_sum": float(prediction_margin.sum().item()),
            "entropy_passed": int(entropy_mask.sum().item()),
            "deyo_entropy_passed": int(entropy_mask.sum().item()),
            "plpd_passed": int(plpd_mask.sum().item()),
            "relative_entropy_passed": 0,
            "absolute_entropy_passed": 0,
            "relative_entropy_threshold": None,
            "effective_entropy_threshold": None,
            "nonredundant": int(selected_mask.sum().item()),
            "region_reliable": int(region_mask.sum().item()),
            "selected": int(selected_indices.numel()),
            "recap_soft_weight_sum": float(selected_weight_values.sum().cpu()),
            "recap_soft_weight_square_sum": float(
                selected_weight_values.square().sum().cpu()
            ),
            "recap_soft_weight_count": int(selected_weight_values.numel()),
            "recap_soft_weight_min": weight_min,
            "recap_soft_weight_max": weight_max,
            "dlrs_reliability_min": reliability_min,
            "dlrs_reliability_max": reliability_max,
            "dlrs_reliability_sum": float(
                selected_reliability.detach().sum().cpu()
            ),
            "dlrs_reliability_count": int(selected_reliability.numel()),
            "dlrs_region_scale_min": region_scale_min,
            "dlrs_region_scale_max": region_scale_max,
            "dlrs_region_scale_sum": float(
                selected_region_scale.detach().sum().cpu()
            ),
            "dlrs_joint_weight_min": joint_weight_min,
            "dlrs_joint_weight_max": joint_weight_max,
            "dlrs_joint_weight_sum": joint_weight_sum_val,
            "dlrs_joint_weight_square_sum": joint_weight_square_sum,
            "dlrs_joint_weight_count": int(selected_joint_weight_values.numel()),
            "dlrs_effective_sample_size": effective_sample_size,
            "gradient_norm_before_clip": gradient_norm_before_clip,
            "gradient_clipped": bool(gradient_clipped),
            "nonfinite_gradient": bool(nonfinite_gradient),
            "updated": updated,
            "nonfinite": nonfinite,
            "loss": loss_value,
            "recap_loss": recap_loss_value,
            "eata_regularization": 0.0,
            "recovered": recovered,
            "recovery_credit": recovery_credit,
            "recovery_threshold": recovery_threshold,
            "recovery_authorized": recovery_authorized,
            "fslr_reliability_gate_disabled": (
                self.disable_fslr_reliability_gate
            ),
            "rga_joint_weight_vector": rga_joint_weight_vector.detach(),
            "rga_applied_weight_vector": rga_applied_weight_vector.detach(),
        }
        self.counters["samples"] += batch_stats["samples"]
        self.counters["quality_passed"] += batch_stats["quality_passed"]
        self.counters["margin_passed"] += batch_stats["margin_passed"]
        self.counters["margin_sum"] += batch_stats["margin_sum"]
        self.counters["entropy_passed"] += batch_stats["entropy_passed"]
        self.counters["deyo_entropy_passed"] += batch_stats["deyo_entropy_passed"]
        self.counters["plpd_passed"] += batch_stats["plpd_passed"]
        self.counters["nonredundant"] += batch_stats["nonredundant"]
        self.counters["region_reliable"] += batch_stats["region_reliable"]
        self.counters["selected"] += batch_stats["selected"]
        self.counters["dlrs_selected"] += batch_stats["selected"]
        self.counters["recap_soft_weight_sum"] += batch_stats[
            "recap_soft_weight_sum"
        ]
        self.counters["recap_soft_weight_square_sum"] += batch_stats[
            "recap_soft_weight_square_sum"
        ]
        self.counters["recap_soft_weight_count"] += batch_stats[
            "recap_soft_weight_count"
        ]
        self.counters["dlrs_reliability_sum"] += batch_stats[
            "dlrs_reliability_sum"
        ]
        self.counters["dlrs_region_scale_sum"] += batch_stats[
            "dlrs_region_scale_sum"
        ]
        self.counters["dlrs_reliability_count"] += batch_stats[
            "dlrs_reliability_count"
        ]
        self.counters["dlrs_joint_weight_sum"] += batch_stats[
            "dlrs_joint_weight_sum"
        ]
        self.counters["dlrs_joint_weight_square_sum"] += batch_stats[
            "dlrs_joint_weight_square_sum"
        ]
        self.counters["dlrs_joint_weight_count"] += batch_stats[
            "dlrs_joint_weight_count"
        ]
        self.counters["recovery_credit_sum"] += recovery_credit
        self.counters["recovery_credit_count"] += 1
        self.counters["effective_sample_size_sum"] += batch_stats["dlrs_effective_sample_size"]
        self.counters["effective_sample_size_batches"] += 1
        if should_skip_update:
            self.counters["skipped_low_quality_batches"] += 1
        if anchor_disagreement is not None:
            self.counters["anchor_disagreement_sum"] += anchor_disagreement
            self.counters["anchor_disagreement_batches"] += 1
        if recovery_authorized:
            self.counters["recovery_authorized_batches"] += 1
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
        return logits.detach(), batch_stats
