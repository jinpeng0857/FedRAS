"""
EATA-C Loss functions:
  - Entropy loss
  - Consistency loss (KL divergence full vs sub-network)
  - Min-max entropy regularization (calibrated by prediction agreement)
"""

import torch
import torch.nn.functional as F


def softmax_entropy(logits: torch.Tensor, reduction: str = "none") -> torch.Tensor:
    """Compute prediction entropy from logits."""
    log_probs = F.log_softmax(logits, dim=1)
    probs = F.softmax(logits, dim=1)
    entropy = -(probs * log_probs).sum(dim=1)
    if reduction == "mean":
        return entropy.mean()
    elif reduction == "sum":
        return entropy.sum()
    return entropy


def entropy_loss_fn(logits: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
    """Standard entropy minimization loss."""
    return softmax_entropy(logits, reduction)


def weighted_entropy_loss(
    logits: torch.Tensor,
    weights: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Weighted entropy minimization loss.
    Used in EATA: S(x) * E(x) where S(x) is the sample selection score.

    Args:
        logits: model predictions (B, C)
        weights: per-sample weights (B,)
        reduction: "mean" or "sum"
    """
    entropy = softmax_entropy(logits, reduction="none")
    weighted = weights * entropy
    if reduction == "mean":
        return weighted.mean()
    elif reduction == "sum":
        return weighted.sum()
    return weighted


def consistency_loss(
    full_logits: torch.Tensor,
    sub_logits: torch.Tensor,
    p: float = 0.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    EATA-C Eqn.(10,11): L_c(x) = D_KL(ŷ_sub || ŷ_fuse)
    ŷ_fuse = (ŷ + (1-p) * ŷ_sub) / (2-p)

    Unidirectional: sub-network prediction is aligned toward the fused prediction.
    KL(sub || fuse) means sub_probs tries to cover where fuse_probs is nonzero
    (zero-avoiding / mode-covering behavior).
    """
    full_probs = F.softmax(full_logits, dim=1).detach()
    sub_probs = F.softmax(sub_logits, dim=1)
    fuse_probs = (full_probs + (1.0 - p) * sub_probs.detach()) / (2.0 - p)

    log_sub = torch.log(sub_probs + 1e-10)
    log_fuse = torch.log(fuse_probs + 1e-10)
    # D_KL(sub || fuse) = sum(sub_probs * (log_sub - log_fuse))
    kl = (sub_probs * (log_sub - log_fuse)).sum(dim=1)

    if reduction == "mean":
        return kl.mean()
    elif reduction == "sum":
        return kl.sum()
    return kl


def prediction_agreement(
    full_logits: torch.Tensor,
    sub_logits: torch.Tensor,
) -> torch.Tensor:
    """
    EATA-C Eqn.(13): C(x) = +1 if argmax agrees, -1 if disagrees.
    """
    full_pred = torch.argmax(full_logits, dim=1)
    sub_pred = torch.argmax(sub_logits, dim=1)
    agree = (full_pred == sub_pred).float()
    return 2.0 * agree - 1.0  # +1 (certain) or -1 (uncertain)


def min_max_entropy_regularization(
    logits: torch.Tensor,
    confidence_signs: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    EATA-C Eqn.(12): min C(x) * E(x; Θ_sub)
    C=+1 -> minimize entropy (certain), C=-1 -> maximize entropy (uncertain)
    """
    entropy = softmax_entropy(logits, reduction="none")
    calibrated = confidence_signs * entropy
    if reduction == "mean":
        return calibrated.mean()
    elif reduction == "sum":
        return calibrated.sum()
    return calibrated
