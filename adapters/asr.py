"""ASR (Adaptive Sharpness Regularization / Entropy Reset) adapter."""

from copy import deepcopy
import math
from typing import Sequence

import torch
import torch.nn.functional as F
from torch import nn

from sparse_restore import PeriodicSparseRestorer
from adapters.base import (
    ASR_OFFICIAL_A0,
    ASR_OFFICIAL_M0,
    ASR_OFFICIAL_P0,
    ASR_OFFICIAL_R0,
    ASR_OFFICIAL_R1,
    DSR_GRAD_CLIP_NORM,
)
from adapters.dual_reliability import (
    DualReliabilityAdapter,
    top1_top2_probability_margin,
)


class ASRAdapter(DualReliabilityAdapter):
    """Official ASR reset controller paired with the ETA entropy objective.

    ASR does not use FedRAS's sample/region reliability weights or DSR. The
    adapter keeps the matched FEDEATA pre-update prediction rule, while the
    reset state follows YonseiML/asr: an entropy decrease relative to the
    exponential moving threshold triggers a partial restore to the deployment
    anchor and resets the Adam state.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        feature_variance: torch.Tensor,
        restorer: PeriodicSparseRestorer,
        fishers,
        param_names: Sequence[str],
        m0: float = ASR_OFFICIAL_M0,
        p0: float = ASR_OFFICIAL_P0,
        a0: float = ASR_OFFICIAL_A0,
        r0: float = ASR_OFFICIAL_R0,
        r1: float = ASR_OFFICIAL_R1,
        anchor_fisher_alpha: float = 0.0,
        anchor_l2_alpha: float = 0.0,
    ):
        super().__init__(
            model=model,
            optimizer=optimizer,
            feature_variance=feature_variance,
            restorer=restorer,
            fishers=fishers,
            tta_method="asr",
            dsr_grad_clip_norm=DSR_GRAD_CLIP_NORM,
        )
        self.sample_filter = "asr_entropy"
        self.asr_m0 = float(m0)
        self.asr_p0 = float(p0)
        self.asr_a0 = float(a0)
        self.asr_r0 = float(r0)
        self.asr_r1 = float(r1)
        self.anchor_fisher_alpha = float(anchor_fisher_alpha)
        self.anchor_l2_alpha = float(anchor_l2_alpha)
        self.asr_param_names = list(param_names)
        self.asr_step = 0
        self.asr_entropy = math.inf
        self.asr_cum_entropy = math.log(
            float(self.proxy.num_classes) * self.asr_a0
        )
        self.asr_last_scope = None
        self.asr_last_flip_rate = None
        self.asr_last_fisher_scale = None
        self.asr_reset_events = 0

        # Compare adapted prediction with immutable source model.
        self.source_model = deepcopy(model)
        self.source_model.eval()
        self.source_model.requires_grad_(False)
        self.initial_optimizer_state = deepcopy(optimizer.state_dict())
        self.source_parameters = {
            name: parameter.detach().clone()
            for name, parameter in self.source_model.named_parameters()
            if name in self.asr_param_names
        }
        missing = [
            name for name in self.asr_param_names
            if name not in self.source_parameters
        ]
        if missing:
            raise KeyError(
                "ASR source anchor is missing trainable parameters: "
                f"{missing[:10]}"
            )
        if self.anchor_fisher_alpha < 0 or self.anchor_l2_alpha < 0:
            raise ValueError("ASR anchor penalties must be non-negative")
        self.anchor_fisher = {}
        if self.anchor_fisher_alpha > 0:
            for name in self.asr_param_names:
                if name not in fishers:
                    raise KeyError(
                        f"ASR Fisher entries are missing for trainable parameter: {name}"
                    )
                fisher_item = fishers[name]
                if not isinstance(fisher_item, (tuple, list)) or len(fisher_item) < 2:
                    raise TypeError(
                        f"ASR Fisher entry must contain (importance, anchor): {name}"
                    )
                importance = torch.as_tensor(
                    fisher_item[0],
                    device=model.fc.weight.device,
                    dtype=model.fc.weight.dtype,
                )
                anchor = torch.as_tensor(
                    fisher_item[1],
                    device=model.fc.weight.device,
                    dtype=model.fc.weight.dtype,
                )
                parameter = dict(model.named_parameters())[name]
                if importance.shape != parameter.shape or anchor.shape != parameter.shape:
                    raise ValueError(f"ASR Fisher shape mismatch: {name}")
                self.anchor_fisher[name] = (
                    importance.detach(),
                    anchor.detach(),
                )
        if not 0.0 <= self.asr_m0 <= 1.0:
            raise ValueError("ASR M0 must be in [0, 1]")
        if self.asr_p0 < 0.0 or self.asr_a0 <= 0.0:
            raise ValueError("ASR P0 must be non-negative and A0 positive")
        if self.asr_r0 < 0.0 or self.asr_r1 < 0.0:
            raise ValueError("ASR reset coefficients must be non-negative")
        self.counters.update(
            {
                "asr_entropy_sum": 0.0,
                "asr_flip_rate_sum": 0.0,
                "asr_fisher_scale_sum": 0.0,
                "asr_resets": 0,
                "asr_kept_parameters": 0,
                "asr_restored_parameters": 0,
                "asr_anchor_regularization_sum": 0.0,
            }
        )

    @torch.no_grad()
    def anchor_drift(self):
        trainable = dict(self.model.named_parameters())
        diff_sq = 0.0
        anchor_sq = 0.0
        for name in self.asr_param_names:
            current = trainable[name].detach()
            anchor = self.source_parameters[name].to(
                device=current.device, dtype=current.dtype
            )
            diff_sq += float((current - anchor).float().square().sum().cpu())
            anchor_sq += float(anchor.float().square().sum().cpu())
        return math.sqrt(diff_sq / max(anchor_sq, 1e-12))

    def _anchor_regularization(self):
        trainable = dict(self.model.named_parameters())
        regularization = None
        if self.anchor_fisher_alpha > 0 and self.anchor_fisher:
            weighted = None
            weight_sum = 0.0
            for name, (importance, fisher_anchor) in self.anchor_fisher.items():
                parameter = trainable[name]
                anchor = fisher_anchor.to(
                    device=parameter.device, dtype=parameter.dtype
                )
                value = (importance * (parameter - anchor).square()).sum()
                weighted = value if weighted is None else weighted + value
                weight_sum += float(importance.float().sum().cpu())
            regularization = self.anchor_fisher_alpha * weighted / max(
                weight_sum, 1e-12
            )
        if self.anchor_l2_alpha > 0:
            l2 = None
            count = 0
            for name in self.asr_param_names:
                parameter = trainable[name]
                anchor = self.source_parameters[name].to(
                    device=parameter.device, dtype=parameter.dtype
                )
                value = (parameter - anchor).square().sum()
                l2 = value if l2 is None else l2 + value
                count += parameter.numel()
            l2 = self.anchor_l2_alpha * l2 / max(count, 1)
            regularization = l2 if regularization is None else regularization + l2
        return regularization

    @torch.no_grad()
    def _reset_from_anchor(self):
        scope = max(
            self.asr_r0
            - self.asr_r1 * (self.asr_cum_entropy - self.asr_entropy),
            0.0,
        )
        scope = min(1.0, float(scope))
        keep_count = int(math.ceil(scope * len(self.asr_param_names)))
        trainable = dict(self.model.named_parameters())
        for index, name in enumerate(self.asr_param_names):
            parameter = trainable[name]
            if index >= keep_count:
                parameter.copy_(self.source_parameters[name])
        self.optimizer.load_state_dict(deepcopy(self.initial_optimizer_state))
        self.asr_last_scope = scope
        self.counters["asr_resets"] += 1
        self.counters["asr_kept_parameters"] += keep_count
        self.counters["asr_restored_parameters"] += (
            len(self.asr_param_names) - keep_count
        )
        self.asr_reset_events += 1
        self.asr_cum_entropy = math.log(
            float(self.proxy.num_classes) * self.asr_a0
        )
        self.asr_step = 0

    @torch.enable_grad()
    def adapt(self, images: torch.Tensor):
        self.optimizer.zero_grad(set_to_none=True)
        logits = self.model(images)
        log_probability = F.log_softmax(logits, dim=1)
        probability = log_probability.exp()
        sample_entropy = -(probability * log_probability).sum(dim=1)
        entropy_loss = sample_entropy.mean()
        anchor_regularization = self._anchor_regularization()
        loss = entropy_loss
        if anchor_regularization is not None:
            loss = loss + anchor_regularization
        updated = False
        nonfinite = False
        if torch.isfinite(loss):
            loss.backward()
            self.optimizer.step()
            updated = True
        else:
            nonfinite = True
        self.optimizer.zero_grad(set_to_none=True)

        with torch.no_grad():
            mean_logits = logits.detach().mean(dim=0, keepdim=True)
            mean_log_probability = F.log_softmax(mean_logits, dim=1)
            mean_probability = mean_log_probability.exp()
            self.asr_entropy = float(
                -(mean_probability * mean_log_probability).sum().cpu()
            )
            source_logits = self.source_model(images)
            source_prediction = source_logits.argmax(dim=1)
            target_prediction = logits.detach().argmax(dim=1)
            flip_rate = (
                source_prediction.ne(target_prediction).float().mean().item()
            )
        self.asr_last_flip_rate = float(flip_rate)
        self.asr_last_fisher_scale = float(self.asr_p0 * (flip_rate ** 2))
        self.asr_step += 1
        triggered_reset = self.asr_entropy < self.asr_cum_entropy
        if triggered_reset:
            self._reset_from_anchor()
        else:
            momentum = self.asr_m0 * flip_rate + 1.0 - self.asr_m0
            self.asr_cum_entropy = (
                momentum * self.asr_cum_entropy
                + (1.0 - momentum) * self.asr_entropy
            )

        recovered = self.restorer.step()
        margin = top1_top2_probability_margin(probability.detach())
        batch_size = int(images.size(0))
        batch_stats = {
            "samples": batch_size,
            "quality_passed": batch_size,
            "margin_passed": int((margin > 0).sum().item()),
            "margin_sum": float(margin.sum().cpu()),
            "entropy_passed": batch_size,
            "relative_entropy_passed": 0,
            "absolute_entropy_passed": 0,
            "relative_entropy_threshold": None,
            "effective_entropy_threshold": None,
            "nonredundant": batch_size,
            "region_reliable": batch_size,
            "selected": batch_size,
            "recap_soft_weight_sum": 0.0,
            "recap_soft_weight_square_sum": 0.0,
            "recap_soft_weight_count": 0,
            "gradient_norm_before_clip": None,
            "gradient_clipped": False,
            "nonfinite_gradient": False,
            "updated": updated,
            "nonfinite": nonfinite,
            "loss": float(loss.detach().cpu()) if torch.isfinite(loss) else 0.0,
            "prediction_entropy": float(sample_entropy.mean().detach().cpu()),
            "anchor_regularization": float(
                anchor_regularization.detach().cpu()
            )
            if anchor_regularization is not None
            else 0.0,
            "source_agreement": float(1.0 - flip_rate),
            "anchor_drift": self.anchor_drift(),
            "recap_loss": 0.0,
            "eata_regularization": 0.0,
            "recovered": recovered,
            "asr_entropy": self.asr_entropy,
            "asr_flip_rate": float(flip_rate),
            "asr_fisher_scale": self.asr_last_fisher_scale,
            "asr_reset": bool(triggered_reset),
            "asr_scope": self.asr_last_scope,
        }
        self.counters["samples"] += batch_size
        self.counters["quality_passed"] += batch_size
        self.counters["margin_passed"] += batch_stats["margin_passed"]
        self.counters["margin_sum"] += batch_stats["margin_sum"]
        self.counters["entropy_passed"] += batch_size
        self.counters["nonredundant"] += batch_size
        self.counters["region_reliable"] += batch_size
        self.counters["selected"] += batch_size
        self.counters["batches"] += 1
        self.counters["updated_batches"] += int(updated)
        self.counters["skipped_batches"] += int(not updated)
        self.counters["nonfinite_batches"] += int(nonfinite)
        self.counters["loss_sum"] += batch_stats["loss"]
        self.counters["asr_anchor_regularization_sum"] += batch_stats[
            "anchor_regularization"
        ]
        self.counters["asr_entropy_sum"] += self.asr_entropy
        self.counters["asr_flip_rate_sum"] += float(flip_rate)
        self.counters["asr_fisher_scale_sum"] += self.asr_last_fisher_scale
        return logits.detach(), batch_stats

    @torch.inference_mode()
    def predict(self, images: torch.Tensor):
        return self.model(images)

    def finalize(self):
        return False
