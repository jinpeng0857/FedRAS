"""
EATA-C Test-Time Adaptation evaluation module.

Performs online EATA-C adaptation on corrupted test data:
  - Active sample selection (entropy + diversity filtering, Eqn.3-5,15)
  - Sub-network consistency loss (KL divergence, Eqn.10-11)
  - Min-max entropy regularization (per-sample calibration, Eqn.12-13)
  - Fisher regularization (anti-forgetting, Eqn.7)
"""

import copy
import math
from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from .fisher import collect_original_params, fisher_regularization
from .losses import (
    consistency_loss,
    prediction_agreement,
    min_max_entropy_regularization,
)
from .sample_selection import ActiveSampleSelector
from .sub_network import get_subnetwork_prediction


@dataclass
class EATACConfig:
    """
    Type-safe configuration for EATA-C test-time adaptation.

    Provides sensible defaults for ImageNet ResNet-50.
    Use classmethods for preset configurations.
    """
    # Adaptation loop
    adapt_steps: int = 3
    adapt_lr: float = 0.005

    # Fisher regularization (anti-forgetting)
    fisher_alpha: float = 50.0

    # Sample selection
    e_margin_factor: float = 0.4
    epsilon: float = 0.05       # cosine similarity threshold
    alpha_ma: float = 0.1       # EMA update factor

    # Loss weights
    consistency_weight: float = 1.0
    entropy_reg_weight: float = 0.1

    # Other
    episodic: bool = False
    num_class: int = 10
    eval_after_adapt: bool = False
    freeze_bn: bool = False

    @classmethod
    def for_cifar10(cls) -> "EATACConfig":
        """Preset for CIFAR-10 (10 classes)."""
        return cls(num_class=10)

    @classmethod
    def for_cifar100(cls) -> "EATACConfig":
        """Preset for CIFAR-100 (100 classes)."""
        return cls(num_class=100, adapt_lr=0.001)

    @classmethod
    def for_imagenet(cls) -> "EATACConfig":
        """Preset for ImageNet (1000 classes)."""
        return cls(
            num_class=1000,
            adapt_lr=0.005,
            fisher_alpha=50.0,
            entropy_reg_weight=0.1,
        )


class EATACTester:
    """
    EATA-C Test-Time Adaptation Evaluator.

    Adapts a trained model online on corrupted test data,
    using EATA-C's calibrated adaptation objectives.
    """

    def __init__(
        self,
        model: nn.Module,
        configs,
        device: torch.device,
        fisher_path: Optional[str] = None,
    ):
        """
        Args:
            model: trained model (with BN/Dropout layers)
            configs: EATACConfig object or argparse config
            device: computation device
            fisher_path: path to pre-computed Fisher dict (.pt)
        """
        self.device = device
        self.base_model = copy.deepcopy(model).to(device)
        self.base_model.eval()

        # Convert config if needed
        if not isinstance(configs, EATACConfig):
            configs = self._convert_to_config(configs)
        self.configs = configs

        # Store original BN parameters for Fisher regularization
        self.original_params = collect_original_params(self.base_model)

        # Load pre-computed Fisher if provided
        self.fisher_values = None
        if fisher_path is not None:
            self.fisher_values = torch.load(fisher_path, map_location="cpu")

        # Adaptable model (trainable copy)
        self.model = copy.deepcopy(self.base_model).to(device)

        # Sample selector: entropy + diversity filtering
        e0 = configs.e_margin_factor * math.log(configs.num_class)
        self.sample_selector = ActiveSampleSelector(
            e0=e0,
            epsilon=configs.epsilon,
            alpha_ma=configs.alpha_ma,
        )

    @staticmethod
    def _convert_to_config(cfg) -> EATACConfig:
        """Convert a namespace/dict to EATACConfig."""
        return EATACConfig(
            adapt_steps=getattr(cfg, "adapt_steps", 3),
            adapt_lr=getattr(cfg, "adapt_lr", 0.005),
            fisher_alpha=getattr(cfg, "fisher_alpha", 50.0),
            e_margin_factor=getattr(cfg, "e_margin", 0.4),
            epsilon=getattr(cfg, "epsilon", 0.05),
            alpha_ma=getattr(cfg, "alpha_ma", 0.1),
            consistency_weight=getattr(cfg, "consistency_weight", 1.0),
            entropy_reg_weight=getattr(cfg, "entropy_reg_weight", 0.1),
            episodic=getattr(cfg, "episodic", False),
            num_class=getattr(cfg, "num_class", 10),
            eval_after_adapt=getattr(cfg, "eval_after_adapt", False),
            freeze_bn=getattr(cfg, "freeze_bn", True),
        )

    def _setup_adaptation(self):
        """Configure model for TTA: only update norm affine params (BN/LN/GN)."""
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
            lr=self.configs.adapt_lr,
            momentum=0.9,
        )

    def _reset_model(self):
        """Reset model to original state."""
        self.model.load_state_dict(self.base_model.state_dict())

    def _reset_model_and_optimizer(self):
        """Reset model to original state AND re-create optimizer.

        Needed for episodic mode: optimizer momentum buffers from a previous
        batch would otherwise contaminate adaptation on the current batch.
        """
        self._reset_model()
        self._setup_adaptation()

    def evaluate_source_only(self, dataloader) -> Dict:
        """Evaluate without any adaptation (source model baseline).

        Returns:
            dict with keys: accuracy
        """
        from utils.metrics import compute_accuracy

        self.model.eval()
        all_logits, all_labels = [], []

        with torch.no_grad():
            for x_batch, y_batch in tqdm(dataloader, desc="Source-Only"):
                x_batch = x_batch.to(self.device)
                result = self.model(x_batch)
                logits = result["logit"] if isinstance(result, dict) else result
                all_logits.append(logits.cpu())
                all_labels.append(y_batch.cpu())

        all_logits = torch.cat(all_logits, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        acc = compute_accuracy(all_logits, all_labels)
        return {"accuracy": acc}

    def evaluate(
        self,
        dataloader,
        return_predictions: bool = False,
    ) -> Dict:
        """
        Evaluate EATA-C TTA on a corrupted test dataloader.

        When eval_after_adapt=True: adapt on batch, then re-forward to eval.
        When eval_after_adapt=False: eval on batch first, then adapt (predict-before-adapt).

        Returns:
            dict with keys: accuracy, ece, forwards, backwards,
                            avg_entropy_pass_rate, avg_selected_ratio
        """
        from utils.metrics import compute_ece, compute_accuracy

        self._setup_adaptation()
        self.sample_selector.reset()

        all_logits, all_labels = [], []
        total_forwards = 0
        total_backwards = 0
        is_first_batch = True
        total_samples = 0
        total_entropy_passed = 0
        total_selected = 0

        e0 = self.configs.e_margin_factor * math.log(self.configs.num_class)
        eval_after_adapt = getattr(self.configs, 'eval_after_adapt', False)

        for x_batch, y_batch in tqdm(dataloader, desc="EATA-C TTA"):
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            if self.configs.episodic:
                # Reset both model weights and optimizer state (momentum buffers)
                self._reset_model_and_optimizer()
                self.sample_selector.reset()
                is_first_batch = True

            # ---- Optional: adapt BEFORE predict (eval after adaptation) ----
            if eval_after_adapt:
                # Step 1: Full network forward for selection (gradient-free)
                with torch.no_grad():
                    result = self.model(x_batch)
                    full_logits = result["logit"] if isinstance(result, dict) else result
                total_forwards += 1

                # Step 2: Active sample selection
                scores, selected_indices = self.sample_selector.select(
                    full_logits, is_first_batch=is_first_batch
                )
                is_first_batch = False
                batch_size = x_batch.size(0)
                total_samples += batch_size
                total_selected += len(selected_indices)

                # Track entropy pass
                probs = F.softmax(full_logits, dim=1)
                log_probs = F.log_softmax(full_logits, dim=1)
                entropy = -(probs * log_probs).sum(dim=1)
                total_entropy_passed += (entropy < e0).float().sum().item()

                # Step 3: Adapt on this batch
                if len(selected_indices) > 0:
                    reliable_full_logits = full_logits[selected_indices]
                    for _ in range(self.configs.adapt_steps):
                        sub_logits_full = get_subnetwork_prediction(self.model, x_batch)
                        total_forwards += 1
                        sub_logits = sub_logits_full[selected_indices]

                        loss_c = consistency_loss(reliable_full_logits.detach(), sub_logits)
                        conf_signs = prediction_agreement(reliable_full_logits.detach(), sub_logits)
                        loss_ent = min_max_entropy_regularization(sub_logits, conf_signs)

                        loss = (
                            self.configs.consistency_weight * loss_c
                            + self.configs.entropy_reg_weight * loss_ent
                        )
                        if self.fisher_values is not None and self.configs.fisher_alpha > 0:
                            loss_fisher = fisher_regularization(
                                self.model, self.original_params, self.fisher_values
                            )
                            loss = loss + self.configs.fisher_alpha * loss_fisher

                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()
                        total_backwards += 1

                # Step 4: Re-forward after adaptation for evaluation
                with torch.no_grad():
                    result = self.model(x_batch)
                    eval_logits = result["logit"] if isinstance(result, dict) else result
                total_forwards += 1

                all_logits.append(eval_logits.detach().cpu())
                all_labels.append(y_batch.cpu())

            else:
                # ---- Predict BEFORE adapt (original behavior) ----
                # Step 1: Full network forward (gradient-free) for evaluation AND selection
                with torch.no_grad():
                    result = self.model(x_batch)
                    full_logits = result["logit"] if isinstance(result, dict) else result
                total_forwards += 1

                batch_size = x_batch.size(0)
                total_samples += batch_size

                probs = F.softmax(full_logits, dim=1)
                log_probs = F.log_softmax(full_logits, dim=1)
                entropy = -(probs * log_probs).sum(dim=1)
                entropy_mask = (entropy < e0).float()
                total_entropy_passed += entropy_mask.sum().item()

                # Active sample selection
                scores, selected_indices = self.sample_selector.select(
                    full_logits, is_first_batch=is_first_batch
                )
                is_first_batch = False
                total_selected += len(selected_indices)

                if len(selected_indices) > 0:
                    reliable_full_logits = full_logits[selected_indices]

                    for _ in range(self.configs.adapt_steps):
                        sub_logits_full = get_subnetwork_prediction(self.model, x_batch)
                        total_forwards += 1
                        sub_logits = sub_logits_full[selected_indices]

                        loss_c = consistency_loss(reliable_full_logits.detach(), sub_logits)
                        conf_signs = prediction_agreement(reliable_full_logits.detach(), sub_logits)
                        loss_ent = min_max_entropy_regularization(sub_logits, conf_signs)

                        loss = (
                            self.configs.consistency_weight * loss_c
                            + self.configs.entropy_reg_weight * loss_ent
                        )
                        if self.fisher_values is not None and self.configs.fisher_alpha > 0:
                            loss_fisher = fisher_regularization(
                                self.model, self.original_params, self.fisher_values
                            )
                            loss = loss + self.configs.fisher_alpha * loss_fisher

                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()
                        total_backwards += 1

                # Record pre-adapt predictions
                all_logits.append(full_logits.detach().cpu())
                all_labels.append(y_batch.cpu())

        all_logits = torch.cat(all_logits, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        probs = F.softmax(all_logits, dim=1)

        acc = compute_accuracy(all_logits, all_labels)
        ece = compute_ece(probs, all_labels)

        avg_entropy_pass_rate = total_entropy_passed / total_samples if total_samples > 0 else 0.0
        avg_selected_ratio = total_selected / total_samples if total_samples > 0 else 0.0

        result = {
            "accuracy": acc,
            "ece": ece,
            "forwards": total_forwards,
            "backwards": total_backwards,
            "avg_entropy_pass_rate": avg_entropy_pass_rate,
            "avg_selected_ratio": avg_selected_ratio,
        }

        if return_predictions:
            result["logits"] = all_logits
            result["labels"] = all_labels

        return result