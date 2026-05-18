"""
FedTTA Client: local EATA-C test-time adaptation for a single-corruption client.

Each client:
  1. Receives init_adapt_state from the server (accelerated adapter)
  2. Loads this state into its local model
  3. Runs EATA-C adaptation on its own single-corruption test data
  4. Only updates normalization affine parameters (adapter)
  5. Computes delta = new_adapt_state - init_adapt_state
  6. Returns delta + metrics to the server

Local loss:
  L = S_c(x) * (L_consistency + alpha * C(x) * entropy_sub)
      + beta * Fisher_regularizer
      + gamma * ||phi - phi_acc||^2  (proximal to accelerated state)
"""

import copy
import math
import time
from collections import OrderedDict
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from src.algorithm.EATA_C.losses import (
    consistency_loss,
    prediction_agreement,
    min_max_entropy_regularization,
)
from src.algorithm.EATA_C.fisher import (
    collect_original_params,
    fisher_regularization,
)
from src.algorithm.EATA_C.sample_selection import ActiveSampleSelector
from src.algorithm.EATA_C.sub_network import get_subnetwork_prediction

from .adapter_utils import (
    extract_adapt_state,
    load_adapt_state,
    state_delta,
)
from .metrics import compute_accuracy, compute_ece


class FedTTAClient:
    """
    Federated TTA client running local EATA-C adaptation.

    Operates on a single-corruption test dataloader.
    Only updates normalization affine parameters.

    Args:
        client_id: unique client identifier
        domain_id: corruption/domain identifier
        corruption: corruption type name
        base_model: trained source model (original weights, used for reset)
        device: computation device
        configs: configuration namespace with EATA-C and FedTTA parameters
        fisher_values: pre-computed Fisher diagonal (optional)
    """

    def __init__(
        self,
        client_id: int,
        domain_id: int,
        corruption: str,
        base_model: nn.Module,
        device: torch.device,
        configs,
        fisher_values: Optional[Dict[str, torch.Tensor]] = None,
    ):
        self.client_id = client_id
        self.domain_id = domain_id
        self.corruption = corruption
        self.device = device
        self.configs = configs

        # Store source model (original weights, never modified)
        self.base_model = copy.deepcopy(base_model).to(device)
        self.base_model.eval()

        # Local model (trainable copy for TTA)
        self.model = copy.deepcopy(self.base_model).to(device)

        # Store original norm params for Fisher regularization
        self.original_params = collect_original_params(self.base_model)

        # Pre-computed Fisher
        self.fisher_values = fisher_values
        if fisher_values is not None:
            self.fisher_values = {k: v.to(device) for k, v in fisher_values.items()}

        # Sample selector
        e0 = getattr(configs, "e_margin", 0.4) * math.log(
            getattr(configs, "num_class", 10)
        )
        self.sample_selector = ActiveSampleSelector(
            e0=e0,
            epsilon=getattr(configs, "epsilon", 0.05),
            alpha_ma=getattr(configs, "alpha_ma", 0.1),
        )

        # Will be created during _setup_adaptation
        self.optimizer = None

    def _setup_adaptation(self):
        """Configure model: only norm affine params trainable, create optimizer."""
        self.model.train()
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d,
                                   nn.LayerNorm, nn.GroupNorm)):
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

    def _reset_model(self):
        """Reset model to original source weights."""
        self.model.load_state_dict(self.base_model.state_dict())

    def _reset_model_and_optimizer(self):
        """Reset model AND re-create optimizer (clears momentum buffers)."""
        self._reset_model()
        self._setup_adaptation()

    def _get_adapt_state(self) -> OrderedDict:
        """Extract current adapter state from local model."""
        return extract_adapt_state(self.model)

    def _load_adapt_state(self, adapt_state: OrderedDict):
        """Load adapter state into local model."""
        load_adapt_state(self.model, adapt_state)

    def evaluate_with_adapt_state(self, adapt_state, test_loader) -> Dict:
        """
        Evaluate with a given adapter state WITHOUT doing any TTA.

        Steps:
          1. Reset to source model
          2. Load server adapter
          3. model.eval(), no_grad pure forward
          4. No optimizer, no TTA, no parameter modification

        This is the paper's primary evaluation protocol.
        Online stats from run_local_eatac_tta() are kept for debugging.

        Returns:
            dict with keys: accuracy, ece, num_samples
        """
        self._reset_model()
        self._load_adapt_state(adapt_state)

        # BN needs batch statistics because running stats from source don't
        # match the adapted affine params. Use train() for BN (batch stats)
        # but keep eval() for everything else (no dropout).
        self.model.eval()
        for m in self.model.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                m.train()

        all_logits, all_labels = [], []
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(self.device), y.to(self.device)
                result = self.model(x)
                logits = result["logit"] if isinstance(result, dict) else result
                all_logits.append(logits.cpu())
                all_labels.append(y.cpu())

        all_logits = torch.cat(all_logits, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        probs = F.softmax(all_logits, dim=1)
        acc = compute_accuracy(all_logits, all_labels)
        ece_val = compute_ece(probs, all_labels)
        return {"accuracy": acc, "ece": ece_val, "num_samples": len(all_logits)}

    def run_local_eatac_tta(
        self,
        init_adapt_state: OrderedDict,
        test_loader,
        num_batches: Optional[int] = None,
        eval_after_adapt: bool = False,
    ) -> Dict:
        """
        Run local EATA-C TTA on single-corruption data.

        Steps:
          0. Reset full model to source weights (prevents cross-round leakage)
          1. Load init_adapt_state (phi_acc from server) into local model
          2. Setup adaptation (create optimizer, enable grad for norm params)
          3. Reset sample selector
          4. Run EATA-C on test_loader
          5. Compute delta = new_adapt_state - init_adapt_state
          6. Return delta + metrics

        Args:
            init_adapt_state: server's accelerated adapter state (phi_acc)
            test_loader: single-corruption test DataLoader
            num_batches: if set, only adapt on this many batches
            eval_after_adapt: if True, re-forward after adaptation for eval

        Returns:
            dict with keys:
                client_id, domain_id, corruption,
                delta (OrderedDict), num_samples,
                stats (dict of accuracy, ece, selected_ratio, ...)
        """
        # 0. Reset full model to source weights (critical: prevents BN stats / adapter leakage)
        self._reset_model()

        # 1. Load server's accelerated adapter
        self._load_adapt_state(init_adapt_state)

        # 2. Save the accelerated state as proximal anchor
        accelerated_anchor = copy.deepcopy(init_adapt_state)
        accelerated_anchor_device = OrderedDict(
            {k: v.to(self.device) for k, v in accelerated_anchor.items()}
        )

        # 3. Setup adaptation + reset selector
        self._setup_adaptation()
        self.sample_selector.reset()

        # 4. Run EATA-C
        all_logits, all_labels = [], []
        total_forwards = 0
        total_backwards = 0
        is_first_batch = True
        total_samples = 0
        total_selected = 0
        total_entropy_pass = 0
        total_consistency_loss = 0.0
        total_disagreement = 0.0
        total_entropy_values = 0.0
        total_confidence = 0.0

        e0 = getattr(self.configs, "e_margin", 0.4) * math.log(
            getattr(self.configs, "num_class", 10)
        )
        gamma_prox = getattr(self.configs, "gamma_prox", 0.0)

        batch_count = 0
        for x_batch, y_batch in tqdm(test_loader, desc=f"Client {self.client_id} TTA"):
            if num_batches is not None and batch_count >= num_batches:
                break
            batch_count += 1

            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            # ---- Forward pass (no grad) for selection ----
            with torch.no_grad():
                result = self.model(x_batch)
                full_logits = result["logit"] if isinstance(result, dict) else result
            total_forwards += 1

            batch_size = x_batch.size(0)
            total_samples += batch_size

            probs = F.softmax(full_logits, dim=1)
            log_probs = F.log_softmax(full_logits, dim=1)
            entropy = -(probs * log_probs).sum(dim=1)
            total_entropy_pass += (entropy < e0).float().sum().item()
            total_entropy_values += entropy.sum().item()

            # Track confidence (sum across all samples, divide later by total_samples)
            batch_conf = probs.max(dim=1)[0]
            total_confidence += batch_conf.sum().item()

            # ---- Sample selection ----
            scores, selected_indices = self.sample_selector.select(
                full_logits, is_first_batch=is_first_batch
            )
            is_first_batch = False
            total_selected += len(selected_indices)

            # ---- Adapt on selected samples ----
            if len(selected_indices) > 0:
                reliable_full_logits = full_logits[selected_indices]

                for _ in range(getattr(self.configs, "adapt_steps", 3)):
                    # Sub-network prediction
                    sub_logits_full = get_subnetwork_prediction(self.model, x_batch)
                    total_forwards += 1
                    sub_logits = sub_logits_full[selected_indices]

                    # Consistency loss (Eqn.10-11)
                    loss_c = consistency_loss(reliable_full_logits.detach(), sub_logits)
                    total_consistency_loss += loss_c.item()

                    # Min-max entropy (Eqn.12-13)
                    conf_signs = prediction_agreement(
                        reliable_full_logits.detach(), sub_logits
                    )
                    loss_ent = min_max_entropy_regularization(sub_logits, conf_signs)
                    total_disagreement += (conf_signs < 0).float().mean().item()

                    loss = (
                        getattr(self.configs, "consistency_weight", 1.0) * loss_c
                        + getattr(self.configs, "entropy_reg_weight", 0.1) * loss_ent
                    )

                    # Fisher regularization (Eqn.7)
                    if self.fisher_values is not None and getattr(
                        self.configs, "fisher_alpha", 50.0
                    ) > 0:
                        loss_fisher = fisher_regularization(
                            self.model, self.original_params, self.fisher_values
                        )
                        loss = loss + getattr(self.configs, "fisher_alpha", 50.0) * loss_fisher

                    # Proximal to accelerated state (FedACG-style) — WITH gradients
                    if gamma_prox > 0:
                        prox_loss = 0.0
                        prox_numel = 0
                        for name, param in self.model.named_parameters():
                            if name in accelerated_anchor_device and param.requires_grad:
                                anchor = accelerated_anchor_device[name].to(param.device)
                                diff = param - anchor
                                prox_loss = prox_loss + (diff ** 2).sum()
                                prox_numel += diff.numel()
                        if prox_numel > 0:
                            prox_loss = prox_loss / prox_numel
                            loss = loss + gamma_prox * prox_loss

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    total_backwards += 1

            # ---- Record predictions ----
            if eval_after_adapt:
                # Re-forward after all adaptation steps for evaluation
                with torch.no_grad():
                    result = self.model(x_batch)
                    eval_logits = result["logit"] if isinstance(result, dict) else result
                total_forwards += 1
                all_logits.append(eval_logits.detach().cpu())
            else:
                all_logits.append(full_logits.detach().cpu())
            all_labels.append(y_batch.cpu())

        # ---- Compute metrics ----
        all_logits = torch.cat(all_logits, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        all_probs = F.softmax(all_logits, dim=1)

        from .metrics import compute_accuracy, compute_ece
        acc = compute_accuracy(all_logits, all_labels)
        ece = compute_ece(all_probs, all_labels)

        avg_selected_ratio = total_selected / max(total_samples, 1)
        avg_entropy = total_entropy_values / max(total_samples, 1)
        avg_consistency = total_consistency_loss / max(total_backwards, 1)
        avg_disagreement = total_disagreement / max(total_backwards, 1)

        # ---- Compute delta ----
        new_adapt_state = self._get_adapt_state()
        delta = state_delta(new_adapt_state, init_adapt_state)

        # delta_norm: RMS of delta (how much this client deviated)
        delta_sq_sum = 0.0
        delta_numel = 0
        for v in delta.values():
            delta_sq_sum += (v ** 2).sum().item()
            delta_numel += v.numel()
        delta_norm_val = math.sqrt(delta_sq_sum / max(delta_numel, 1))

        num_class = getattr(self.configs, "num_class", all_probs.size(1))
        mean_confidence = total_confidence / max(total_samples, 1)
        entropy_norm = avg_entropy / max(math.log(num_class), 1e-8)

        # Distribution-based collapse: 1.0 - normalized prediction entropy
        preds = all_logits.argmax(dim=1)
        pred_hist = torch.bincount(preds, minlength=num_class).float()
        pred_dist = pred_hist / pred_hist.sum().clamp_min(1.0)
        pred_dist_entropy = -(pred_dist * (pred_dist + 1e-8).log()).sum()
        collapse_score = 1.0 - pred_dist_entropy / math.log(num_class)
        collapse_score = float(collapse_score.item())

        # Compute reliability raw score now so it's available for logging
        from .metrics import compute_client_reliability
        reliability_raw = compute_client_reliability({
            "selected_ratio": avg_selected_ratio,
            "mean_confidence": mean_confidence,
            "mean_entropy_norm": entropy_norm,
            "collapse_score": collapse_score,
            "mean_disagreement": avg_disagreement,
        })

        return {
            "client_id": self.client_id,
            "domain_id": self.domain_id,
            "corruption": self.corruption,
            "delta": delta,
            "num_samples": total_samples,
            "stats": {
                "accuracy": acc,
                "ece": ece,
                "selected_ratio": avg_selected_ratio,
                "mean_entropy": avg_entropy,
                "mean_disagreement": avg_disagreement,
                "mean_consistency_loss": avg_consistency,
                "mean_confidence": mean_confidence,
                "mean_entropy_norm": entropy_norm,
                "collapse_score": collapse_score,
                "delta_norm": delta_norm_val,
                "reliability_raw": reliability_raw,
                "forwards": total_forwards,
                "backwards": total_backwards,
            },
        }
