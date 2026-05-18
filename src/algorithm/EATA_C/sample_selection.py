"""
Active sample selection for EATA-C test-time adaptation.

Implements EATA-C sample selection (Eqn.15):
    S_c(x) = S_div(x) * I{E(x) < E0}

Components:
  - Entropy filtering: keeps only low-entropy (reliable) samples
  - Diversity filtering: removes redundant samples via EMA + cosine similarity
"""

import torch
import torch.nn.functional as F


class ActiveSampleSelector:
    """
    Selects reliable and diverse samples for test-time adaptation.

    Maintains an exponential moving average (EMA) of output probabilities
    to detect redundant samples via cosine similarity (Eqn.4-5).

    EATA-C selection (Eqn.15): S_c(x) = S_div(x) * I{E(x) < E0}
    """

    def __init__(
        self,
        e0: float,
        epsilon: float = 0.05,
        alpha_ma: float = 0.1,
    ):
        """
        Args:
            e0: entropy threshold E0 = margin * ln(num_classes)
            epsilon: cosine similarity threshold for diversity filtering
            alpha_ma: EMA update factor (Eqn.4)
        """
        self.e0 = e0
        self.epsilon = epsilon
        self.alpha_ma = alpha_ma
        self.moving_average: torch.Tensor | None = None

    def _compute_entropy_mask(self, logits: torch.Tensor) -> torch.Tensor:
        """Binary mask: 1 for samples with entropy < e0, 0 otherwise."""
        probs = F.softmax(logits, dim=1)
        log_probs = F.log_softmax(logits, dim=1)
        entropy = -(probs * log_probs).sum(dim=1)
        return (entropy < self.e0).float()

    def _compute_diversity_mask(
        self, logits: torch.Tensor, is_first_batch: bool
    ) -> torch.Tensor:
        """
        Binary mask: 1 for non-redundant samples, 0 for redundant.
        Eqn.5: S_div(x) = I{cos(f(x), m_{t-1}) < epsilon}
        """
        probs = F.softmax(logits, dim=1)

        # First batch: no history, keep all samples
        if is_first_batch or self.moving_average is None:
            self.moving_average = probs.mean(dim=0, keepdim=True)
            return torch.ones(probs.size(0), device=probs.device)

        # Cosine similarity between each sample and the EMA
        cosine_sim = F.cosine_similarity(
            probs, self.moving_average.expand_as(probs), dim=1
        )
        return (cosine_sim < self.epsilon).float()

    def select(
        self, logits: torch.Tensor, is_first_batch: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        EATA-C sample selection (Eqn.15).

        Returns:
            scores: per-sample selection scores (0 or 1)
            selected_indices: indices of selected samples
        """
        entropy_mask = self._compute_entropy_mask(logits)
        diversity_mask = self._compute_diversity_mask(logits, is_first_batch)
        scores = entropy_mask * diversity_mask

        # Update EMA with selected samples
        selected = scores.bool()
        if selected.any():
            selected_probs = F.softmax(logits[selected], dim=1)
            batch_mean = selected_probs.mean(dim=0, keepdim=True)
            if self.moving_average is not None:
                self.moving_average = (
                    self.alpha_ma * batch_mean
                    + (1.0 - self.alpha_ma) * self.moving_average
                )
            else:
                self.moving_average = batch_mean

        selected_indices = scores.nonzero(as_tuple=True)[0]
        return scores, selected_indices

    def reset(self) -> None:
        """Reset EMA state (call when switching domains)."""
        self.moving_average = None
