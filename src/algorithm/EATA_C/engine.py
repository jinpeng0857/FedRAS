"""
EATA-C TTA Engine: shared single-batch adaptation logic.

Extracted from the duplicated code in eata_c_test.py and client_fedtta.py.
Both EATACTester.evaluate() and FedTTAClient.run_local_eatac_tta() call
this engine for the core adaptation step.

Usage:
    engine = EATACEngine(model, device, configs, fisher_values)
    loss = engine.adapt_batch(x_batch, sample_selector, is_first_batch)
"""

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .losses import (
    consistency_loss,
    prediction_agreement,
    min_max_entropy_regularization,
)
from .fisher import collect_original_params, fisher_regularization
from .sub_network import get_subnetwork_prediction


class EATACEngine:
    """
    EATA-C adaptation engine: handles sample selection, loss computation,
    and optimization for a single batch of test-time adaptation.

    Only updates normalization affine parameters (BN/LN/GN weight & bias).
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        configs,
        fisher_values: Optional[Dict[str, torch.Tensor]] = None,
    ):
        self.device = device
        self.configs = configs
        self.model = model

        # Original BN params for Fisher regularization
        self.original_params = collect_original_params(model)

        # Fisher diagonal (pre-computed, optional)
        self.fisher_values = fisher_values
        if fisher_values is not None:
            self.fisher_values = {k: v.to(device) for k, v in fisher_values.items()}

        # Optimizer (lazy init)
        self.optimizer = None

    def setup_adaptation(self):
        """Configure model: only norm affine params trainable, create optimizer."""
        self.model.train()
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d,
                                   nn.LayerNorm, nn.GroupNorm,
                                   nn.InstanceNorm2d, nn.InstanceNorm1d)):
                for p in module.parameters():
                    p.requires_grad = True
            else:
                for p in module.parameters():
                    p.requires_grad = False

        # Optionally freeze BN running stats but keep affine trainable
        if getattr(self.configs, 'freeze_bn', False):
            for m in self.model.modules():
                if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                    m.eval()
                    if m.weight is not None:
                        m.weight.requires_grad = True
                    if m.bias is not None:
                        m.bias.requires_grad = True

        adapt_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.SGD(
            adapt_params,
            lr=getattr(self.configs, "adapt_lr", 0.005),
            momentum=0.9,
        )

    def adapt_batch(
        self,
        x_batch: torch.Tensor,
        sample_selector,
        is_first_batch: bool = False,
        proximal_anchor: Optional[Dict[str, torch.Tensor]] = None,
        gamma_prox: float = 0.0,
    ) -> Dict:
        """
        Run one batch of EATA-C adaptation.

        Steps:
          1. Forward pass (no grad) for sample selection
          2. Active sample selection (entropy + diversity)
          3. Multi-step adaptation:
             - Sub-network consistency loss
             - Min-max entropy regularization
             - Fisher regularization (optional)
             - Proximal loss to anchor (optional, FL-specific)
          4. Return adaptation metrics

        Args:
            x_batch: input batch (B, C, H, W)
            sample_selector: ActiveSampleSelector instance
            is_first_batch: reset EMA state if True
            proximal_anchor: dict of param_name -> tensor for proximal term
            gamma_prox: proximal regularization weight

        Returns:
            dict with keys:
                full_logits: pre-adaptation logits (detached)
                selected_indices: indices of selected samples
                total_forwards: number of forward passes
                total_backwards: number of backward passes
                loss_details: dict of individual loss terms
        """
        # ---- Forward pass (no grad) for selection ----
        with torch.no_grad():
            result = self.model(x_batch)
            full_logits = result["logit"] if isinstance(result, dict) else result

        # ---- Sample selection ----
        scores, selected_indices = sample_selector.select(
            full_logits, is_first_batch=is_first_batch
        )

        loss_details = {}
        adapt_steps = getattr(self.configs, "adapt_steps", 3)
        total_forwards = 1
        total_backwards = 0

        # ---- Adapt on selected samples ----
        if len(selected_indices) > 0:
            reliable_full_logits = full_logits[selected_indices]

            for _ in range(adapt_steps):
                # Sub-network prediction (stochastic forward)
                sub_logits_full = get_subnetwork_prediction(self.model, x_batch)
                total_forwards += 1
                sub_logits = sub_logits_full[selected_indices]

                # Consistency loss (Eqn.10-11): KL(sub || fuse)
                loss_c = consistency_loss(reliable_full_logits.detach(), sub_logits)

                # Min-max entropy (Eqn.12-13)
                conf_signs = prediction_agreement(
                    reliable_full_logits.detach(), sub_logits
                )
                loss_ent = min_max_entropy_regularization(sub_logits, conf_signs)

                loss = (
                    getattr(self.configs, "consistency_weight", 1.0) * loss_c
                    + getattr(self.configs, "entropy_reg_weight", 0.1) * loss_ent
                )

                loss_details["consistency"] = loss_c.item()
                loss_details["entropy"] = loss_ent.item()

                # Fisher regularization (Eqn.7)
                if self.fisher_values is not None and getattr(
                    self.configs, "fisher_alpha", 50.0
                ) > 0:
                    loss_fisher = fisher_regularization(
                        self.model, self.original_params, self.fisher_values
                    )
                    loss = loss + getattr(self.configs, "fisher_alpha", 50.0) * loss_fisher
                    loss_details["fisher"] = loss_fisher.item()

                # Proximal regularization (FedACG-style, FL-specific)
                if gamma_prox > 0 and proximal_anchor is not None:
                    prox_loss = 0.0
                    prox_numel = 0
                    for name, param in self.model.named_parameters():
                        if name in proximal_anchor and param.requires_grad:
                            anchor = proximal_anchor[name].to(param.device)
                            diff = param - anchor
                            prox_loss = prox_loss + (diff ** 2).sum()
                            prox_numel += diff.numel()
                    if prox_numel > 0:
                        prox_loss = prox_loss / prox_numel
                        loss = loss + gamma_prox * prox_loss
                        loss_details["proximal"] = prox_loss.item()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_backwards += 1

        return {
            "full_logits": full_logits.detach(),
            "selected_indices": selected_indices,
            "total_forwards": total_forwards,
            "total_backwards": total_backwards,
            "loss_details": loss_details,
        }
