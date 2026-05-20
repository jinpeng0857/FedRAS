"""FedTTA: Domain-aware Federated Test-Time Adaptation."""

from .adapter_utils import (
    extract_adapt_state,
    load_adapt_state,
    state_delta,
    apply_state_delta,
    weighted_average_delta,
    zeros_like_state,
    count_adapt_params,
    add_state,
    scale_state,
)
from .client_fedtta import FedTTAClient
from .server_fedtta import FedTTAServer
from utils.metrics import (
    compute_accuracy,
    compute_ece,
    compute_nll,
    compute_brier_score,
    compute_client_reliability,
)

__all__ = [
    "extract_adapt_state",
    "load_adapt_state",
    "state_delta",
    "apply_state_delta",
    "weighted_average_delta",
    "zeros_like_state",
    "count_adapt_params",
    "add_state",
    "scale_state",
    "FedTTAClient",
    "FedTTAServer",
    "compute_accuracy",
    "compute_ece",
    "compute_nll",
    "compute_brier_score",
    "compute_client_reliability",
]
