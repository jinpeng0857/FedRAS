"""Base adapter classes, ReCAP RegionConfidenceProxy, and shared constants."""

import math
from typing import Dict, Mapping, Optional, Sequence, Tuple
import torch
import torch.nn.functional as F
from torch import nn

from sparse_restore import PeriodicSparseRestorer

# Common constants
REGION_SCALE = 1.2
OFFICIAL_SCALE = 0.1
REWEIGHT_CAP = 2.0
RECAP_SOFT_WEIGHT_LOGIT_LIMIT = 15.0
TAU_RE_FACTOR = 0.8
L0_FACTOR = 0.7
MARGIN_QUALITY_THRESHOLD = 0.15
RECAP_SOFT_WEIGHT_TEMPERATURE_FACTOR = 0.10
FSLR_GRAD_CLIP_NORM = 15.0
EATA_E_MARGIN_FACTOR = 0.4
EATA_D_MARGIN = 0.05
EATA_FISHER_ALPHA = 2000.0

ASR_OFFICIAL_M0 = 0.15
ASR_OFFICIAL_P0 = 5.0
ASR_OFFICIAL_A0 = 0.5
ASR_OFFICIAL_R0 = 0.5
ASR_OFFICIAL_R1 = 20.0

DLRS_REGION_SCALE_MIN = 0.6
DLRS_REGION_SCALE_MAX = REGION_SCALE
DLRS_RELIABILITY_ALPHA = 0.7

DUAL_RELIABILITY_METHODS = (
    "tent",
    "eata",
    "asr",
    "recap",
    "eata_recap",
)
BASE_METHODS = ("base", "base_fslr")
DLRS_METHODS = ("dlrs", "dlrs_fslr")
TTA_METHODS = (
    DUAL_RELIABILITY_METHODS
    + ("sar", "cotta", "rotta", "petta", "deyo")
    + BASE_METHODS
    + DLRS_METHODS
    + ("cola", "feddr")
)
FEDDR_NAME = "FedDR"
FEDDR_TTA_METHOD = "dlrs_fslr"
FEDDR_DEFINITION = "DLRS+FSLR"
FEDDR_PRIMARY_NORMALIZE_BY_WEIGHT = True
FEDDR_PRIMARY_WEIGHT_MODE = "product"
FEDDR_PRIMARY_DISABLE_FSLR_RELIABILITY_GATE = True
FEDDR_PRIMARY_HYBRID_THRESHOLD = 0.2
FEDDR_PRIMARY_ANCHOR_RECTIFY_BETA = 0.3
FEDDR_PRIMARY_LR = 1.0e-3
FEDDR_PRIMARY_BN_SCOPE = "excluding_layer4"
FEDDR_PRIMARY_FSLR_CONTINUOUS_RISK = True
FEDDR_PRIMARY_DLRS_VIEW_CONSISTENCY = True
FEDDR_PRIMARY_DLRS_VIEW_ALPHA = 0.80
FEDDR_PRIMARY_DLRS_VIEW_JS_SCALE = 1.0
FEDDR_PRIMARY_FSLR_HIERARCHICAL_BUDGET = True
FEDDR_PRIMARY_FSLR_LAYER_QUOTA_CAP = 0.35

TTA_METHOD_LABELS = {
    "tent": "Tent",
    "eata": "EATA",
    "asr": "ASR",
    "sar": "SAR",
    "cotta": "CoTTA",
    "rotta": "RoTTA",
    "petta": "PeTTA",
    "deyo": "DeYO",
    "recap": "ReCAP",
    "eata_recap": "EATA-ReCAP",
    "base": "Base",
    "base_fslr": "FSLR",
    "dlrs": "DLRS",
    "dlrs_fslr": "DLRS+FSLR",
    "cola": "ETA+CoLA",
    "feddr": FEDDR_NAME,
}


def resolve_tta_method(tta_method: Optional[str] = None) -> str:
    if tta_method is None:
        raise ValueError("--tta_method is required; the primary experiment entry is feddr")
    normalized = str(tta_method).strip().lower()
    if normalized == "feddr":
        return FEDDR_TTA_METHOD
    if normalized not in TTA_METHODS:
        raise ValueError(
            f"Unknown tta_method '{tta_method}'. Expected one of {TTA_METHODS}."
        )
    return normalized


def get_linear_classifier(model: nn.Module) -> Tuple[str, nn.Linear]:
    """Find the top-level linear classifier head."""
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and (
            name in ("fc", "classifier", "head") or name.endswith(".fc") or name.endswith(".classifier")
        ):
            return name, module
    # Fallback to last Linear module
    last_name, last_module = None, None
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            last_name, last_module = name, module
    if last_module is not None:
        return last_name, last_module
    raise ValueError("model contains no nn.Linear classifier")


class RegionConfidenceProxy(nn.Module):
    """Closed-form ReCAP regional entropy and regional instability."""

    def __init__(self, classifier_weight: torch.Tensor, feature_variance: torch.Tensor):
        super().__init__()
        weight = classifier_weight.detach().float()
        variance = feature_variance.to(device=weight.device, dtype=weight.dtype)
        if variance.numel() != weight.size(1):
            raise ValueError(
                "feature/classifier mismatch: "
                f"variance={variance.numel()}, classifier={weight.size(1)}"
            )

        regional_variance = variance * REGION_SCALE / OFFICIAL_SCALE
        difference = weight.unsqueeze(0) - weight.unsqueeze(1)
        pairwise = 0.5 * torch.einsum(
            "ijd,d,ijd->ij", difference, regional_variance, difference
        )
        correction = (
            0.1
            / 2.0
            * (OFFICIAL_SCALE**2)
            * weight.square().sum(dim=1)
        )
        self.num_classes = int(weight.size(0))
        self.register_buffer("pairwise", pairwise)
        self.register_buffer("correction", correction)

    def forward(
        self,
        logits: torch.Tensor,
        region_scale: torch.Tensor = None,
    ):
        log_probability = F.log_softmax(logits.float(), dim=1)
        probability = log_probability.exp()
        if region_scale is None:
            pairwise = self.pairwise.unsqueeze(0)
        else:
            if region_scale.ndim != 1 or region_scale.numel() != logits.size(0):
                raise ValueError("region_scale must have shape [batch]")
            region_scale = region_scale.detach().to(
                device=logits.device,
                dtype=log_probability.dtype,
            )
            if (
                not torch.isfinite(region_scale).all()
                or (region_scale <= 0).any()
            ):
                raise ValueError("region_scale must be finite and positive")
            scale_ratio = region_scale / float(REGION_SCALE)
            pairwise = self.pairwise.unsqueeze(0) * scale_ratio.view(-1, 1, 1)
        augmented_partition = torch.logsumexp(
            log_probability.unsqueeze(1) + pairwise,
            dim=2,
        )
        regional_probability = F.softmax(
            logits.float() + self.correction.unsqueeze(0), dim=1
        )
        regional_entropy = (
            regional_probability * (-log_probability + augmented_partition)
        ).sum(dim=1)
        regional_instability = (probability * augmented_partition).sum(dim=1)
        return regional_entropy, regional_instability


class DisabledSparseRestorer:
    """Track normalization drift while leaving parameters and optimizer untouched."""

    def __init__(self, model: nn.Module):
        self.params = PeriodicSparseRestorer._collect_trainable_norm_parameters(model)
        self.anchors = {
            name: param.detach().clone() for name, param in self.params.items()
        }
        self.selected_count = 0
        self.num_batches = 0

    def step(self, reliability=None, entropy=None, anchor_disagreement=None, gradient_clipped=False, **kwargs):
        self.num_batches += 1
        return 0

    @torch.no_grad()
    def drift_stats(self):
        accumulators = {
            "total": [0.0, 0.0],
            "gamma": [0.0, 0.0],
            "beta": [0.0, 0.0],
        }
        for name, param in self.params.items():
            anchor = self.anchors[name].to(
                device=param.device, dtype=param.dtype
            )
            group = "gamma" if name.endswith(".weight") else "beta"
            delta_square = float((param.detach() - anchor).square().sum().cpu())
            anchor_square = float(anchor.square().sum().cpu())
            accumulators["total"][0] += delta_square
            accumulators["total"][1] += anchor_square
            accumulators[group][0] += delta_square
            accumulators[group][1] += anchor_square
        return {
            name: float(
                math.sqrt(delta_square) / (math.sqrt(anchor_square) + 1e-12)
            )
            for name, (delta_square, anchor_square) in accumulators.items()
        }

    @property
    def stats(self):
        return {
            "budget_mode": "disabled",
            "interval": 0,
            "sparse_ratio": 0.0,
            "drift_thresholds": [],
            "drift_ratios": [],
            "restore_strength": 0.0,
            "total_trainable_elements": sum(
                param.numel() for param in self.params.values()
            ),
            "selected_elements": 0,
            "maximum_selected_elements": 0,
            "batches": self.num_batches,
            "scheduled_restore_events": 0,
            "restore_events": 0,
            "skipped_low_drift_events": 0,
            "skipped_low_reliability_events": 0,
            "budget_event_counts": {},
            "recovered_elements_total": 0,
            "restored_assignments_total": 0,
            "restored_unique_parameters": 0,
            "repeat_restore_mean": 0.0,
            "last_recovered_elements": 0,
            "last_requested_elements": 0,
            "last_positive_score_elements": None,
            "last_selected_drift_energy_ratio": 0.0,
            "last_selection_jaccard": None,
            "mean_selection_jaccard": None,
            "last_fisher_rms": None,
            "last_parameter_drift_rms": None,
            "last_drift_before": self.drift_stats(),
            "last_drift_after": self.drift_stats(),
            "last_drift_reduction": 0.0,
            "last_relative_drift_reduction": 0.0,
            "per_layer": {},
        }
