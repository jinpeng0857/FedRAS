"""
Shared metrics for FEDEATA evaluation.

Single source of truth for:
  - Accuracy, ECE, NLL, Brier score
  - Per-client reliability weighting
"""

import torch
import torch.nn.functional as F


def compute_accuracy(logits, labels):
    """Compute accuracy from logits and ground-truth labels."""
    pred = logits.argmax(dim=1)
    return (pred == labels).float().mean().item()


def compute_ece(probs, labels, n_bins=15):
    """Compute Expected Calibration Error."""
    confidences, predictions = probs.max(dim=1)
    correct = (predictions == labels).float()
    bin_boundaries = torch.linspace(0, 1, n_bins + 1, device=probs.device)
    ece = 0.0
    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (
            confidences <= bin_boundaries[i + 1]
        )
        prop = in_bin.float().mean()
        if prop > 0:
            acc_bin = correct[in_bin].mean()
            conf_bin = confidences[in_bin].mean()
            ece += torch.abs(acc_bin - conf_bin) * prop
    return ece.item()


def compute_nll(logits, labels):
    """Compute Negative Log-Likelihood."""
    log_probs = F.log_softmax(logits, dim=1)
    nll = F.nll_loss(log_probs, labels)
    return nll.item()


def compute_brier_score(probs, labels):
    """Compute Brier score: (1/N) * sum ||one_hot(y) - p(x)||^2.
    Lower is better (0 = perfect, 2 = worst for K>=2).
    """
    num_classes = probs.size(1)
    one_hot = F.one_hot(labels, num_classes=num_classes).float()
    brier = ((probs - one_hot) ** 2).sum(dim=1).mean().item()
    return brier


def compute_client_reliability(stats: dict) -> float:
    """
    Raw reliability score for a client.

    Formula:
      score = 1.0 * selected_ratio
            + 0.5 * mean_confidence
            - 0.5 * mean_entropy_norm
            - 0.5 * collapse_score
            - 0.5 * mean_disagreement

    Higher is more reliable. This is a raw score — the server applies
    per-domain softmax normalization before using as aggregation weights.

    Args:
        stats: dict with keys 'selected_ratio', 'mean_confidence',
               'mean_entropy_norm', 'collapse_score', 'mean_disagreement'

    Returns:
        raw_reliability: float (to be softmax-normalized per domain by server)
    """
    selected_ratio = stats.get("selected_ratio", 0.0)
    mean_conf = stats.get("mean_confidence", 0.0)
    entropy_norm = stats.get("mean_entropy_norm", 1.0)
    collapse = stats.get("collapse_score", 0.0)
    disagreement = stats.get("mean_disagreement", 0.0)

    score = (
        1.0 * selected_ratio
        + 0.5 * mean_conf
        - 0.5 * entropy_norm
        - 0.5 * collapse
        - 0.5 * disagreement
    )
    return float(score)
