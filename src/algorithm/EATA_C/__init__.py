"""EATA-C: Efficient Anti-forgetting Test-time Adaptation with Calibration."""

from .eata_c_test import EATACTester, EATACConfig
from .fisher import compute_fisher_information, collect_original_params, fisher_regularization
from .losses import (
    softmax_entropy,
    entropy_loss_fn,
    weighted_entropy_loss,
    consistency_loss,
    prediction_agreement,
    min_max_entropy_regularization,
)
from .sample_selection import ActiveSampleSelector
from .sub_network import get_subnetwork_prediction

__all__ = [
    "EATACTester",
    "EATACConfig",
    "ActiveSampleSelector",
    "get_subnetwork_prediction",
    "compute_fisher_information",
    "collect_original_params",
    "fisher_regularization",
    "softmax_entropy",
    "entropy_loss_fn",
    "weighted_entropy_loss",
    "consistency_loss",
    "prediction_agreement",
    "min_max_entropy_regularization",
]
