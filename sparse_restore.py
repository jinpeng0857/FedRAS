"""Periodic Fisher-guided restoration with drift-tiered budgets, anchor-disagreement events, tiered momentum purging, and healthy-gated Fisher fusion.

Every ``interval`` test batches (or upon emergency sustained event triggers), the restorer
measures the normalized global normalization affine drift from the deployment anchor.
Parameters are ranked by normalized Fisher importance (or online-fused Fisher) times
their current absolute drift. When DLRS supplies batch reliability, entropy, and anchor
disagreement, the restorer tracks the joint condition:
    E_t = [r_bar_t < tau_r or A_t > tau_A] and [z(H_t) < -tau_H] and [D_t > tau_D]
sustained over K consecutive batches to detect genuine semantic collapse / overconfident errors,
while strictly avoiding false alarms on clean domains.
"""

import math
from typing import Dict, Mapping, Optional, Sequence

import torch
from torch import nn


def load_fishers(statistics_path: str) -> Mapping[str, torch.Tensor]:
    """Load the Fisher dictionary stored by ``prepare_main_statistics.py``."""
    payload = torch.load(statistics_path, map_location="cpu", weights_only=False)
    fishers = payload.get("fishers") if isinstance(payload, dict) else None
    if not isinstance(fishers, dict) or not fishers:
        raise ValueError(
            "statistics file must contain a non-empty 'fishers' dictionary"
        )
    return fishers


class PeriodicSparseRestorer:
    """Fixed-period or joint event-triggered Fisher restoration."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        fishers: Mapping[str, torch.Tensor],
        interval: int,
        sparse_ratio: float = 0.01,
        restore_strength: float = 0.7,
        budget_mode: str = "drift_tiered",
        drift_thresholds: Sequence[float] = (0.10, 0.20, 0.35),
        drift_ratios: Sequence[float] = (0.0, 0.02, 0.05, 0.10),
        reliability_threshold: float = 0.55,
        reliability_ema_decay: float = 0.90,
        adam_exp_avg_decay: bool = False,
        adam_exp_avg_sq_decay: bool = False,
        adam_exp_avg_sq_decay_factor: float = None,
        layer_aware: bool = False,
        layer_weights: Optional[Dict[str, float]] = None,
        disable_reliability_gate: bool = False,
        event_trigger: bool = False,
        event_k: int = 3,
        event_tau_r: float = 0.40,
        event_tau_a: float = 0.50,
        event_tau_h: float = 1.50,
        event_tau_d: float = 0.20,
        event_lambda_0: float = 0.50,
        event_lambda_max: float = 0.90,
        event_k_rho: float = 2.0,
        event_rho_max: float = 0.20,
        event_cooldown: int = 20,
        online_fisher_beta: float = 0.0,
        online_fisher_alpha: float = 0.90,
        risk_rho_min: float = 0.0,
        risk_rho_max: float = 0.10,
        risk_gamma_ess: float = 0.50,
        risk_gamma_anchor: float = 0.50,
        risk_gamma_drift: float = 0.0,
        risk_tau_drift: float = 0.35,
        hierarchical_budget: bool = False,
        layer_quota_cap: float = 0.35,
        momentum_discounting: bool = False,
        momentum_away_factor: float = 0.20,
        selection_criterion: str = "fisher_drift",
        selection_seed: Optional[int] = None,
    ):
        if budget_mode not in ("fixed", "drift_tiered", "continuous_risk"):
            raise ValueError(f"unknown budget_mode: {budget_mode}")
        if interval <= 0:
            raise ValueError("interval must be positive")
        if not 0.0 <= sparse_ratio <= 1.0:
            raise ValueError("sparse_ratio must be in [0, 1]")
        if not 0.0 <= restore_strength <= 1.0:
            raise ValueError("restore_strength must be in [0, 1]")
        if len(drift_thresholds) + 1 != len(drift_ratios):
            raise ValueError(
                "drift_ratios must have exactly one more element than drift_thresholds"
            )
        if any(value < 0 for value in drift_thresholds):
            raise ValueError("drift_thresholds must be non-negative")
        if any(
            left >= right
            for left, right in zip(drift_thresholds, drift_thresholds[1:])
        ):
            raise ValueError("drift_thresholds must be strictly increasing")
        if any(not 0.0 <= value <= 1.0 for value in drift_ratios):
            raise ValueError("drift_ratios must be in [0, 1]")
        if any(
            left > right
            for left, right in zip(drift_ratios, drift_ratios[1:])
        ):
            raise ValueError("drift_ratios must be non-decreasing")
        if not 0.0 <= reliability_threshold <= 1.0:
            raise ValueError("reliability_threshold must be in [0, 1]")
        if not 0.0 <= reliability_ema_decay < 1.0:
            raise ValueError("reliability_ema_decay must be in [0, 1)")

        self.optimizer = optimizer
        self.interval = int(interval)
        self.sparse_ratio = float(sparse_ratio)
        self.restore_strength = float(restore_strength)
        self.budget_mode = budget_mode
        self.drift_thresholds = tuple(float(x) for x in drift_thresholds)
        self.drift_ratios = tuple(float(x) for x in drift_ratios)
        self.reliability_threshold = float(reliability_threshold)
        self.reliability_ema_decay = float(reliability_ema_decay)
        self.adam_exp_avg_decay = bool(adam_exp_avg_decay)
        self.adam_exp_avg_sq_decay = bool(adam_exp_avg_sq_decay)
        self.layer_aware = bool(layer_aware)
        self.layer_weights = layer_weights or {}
        self.disable_reliability_gate = bool(disable_reliability_gate)

        # Continuous Risk Parameters (P1)
        self.risk_rho_min = float(risk_rho_min)
        self.risk_rho_max = float(risk_rho_max)
        self.risk_gamma_ess = float(risk_gamma_ess)
        self.risk_gamma_anchor = float(risk_gamma_anchor)
        self.risk_gamma_drift = float(risk_gamma_drift)
        self.risk_tau_drift = float(risk_tau_drift)
        self.last_risk_score: Optional[float] = None
        self.hierarchical_budget = bool(hierarchical_budget)
        self.layer_quota_cap = float(layer_quota_cap)
        if not 0.0 < self.layer_quota_cap <= 1.0:
            raise ValueError("layer_quota_cap must be in (0, 1]")
        self.momentum_discounting = bool(momentum_discounting)
        self.momentum_away_factor = float(momentum_away_factor)
        if not 0.0 <= self.momentum_away_factor <= 1.0:
            raise ValueError("momentum_away_factor must be in [0, 1]")
        if selection_criterion not in ("fisher_drift", "drift_only", "fisher_only", "random"):
            raise ValueError(f"unknown selection_criterion: {selection_criterion}")
        self.selection_criterion = str(selection_criterion)
        self.selection_seed = int(selection_seed) if selection_seed is not None else None

        # Joint Sustained Event & Elastic Restore Parameters
        self.event_trigger = bool(event_trigger)
        self.event_k = int(event_k)
        self.event_tau_r = float(event_tau_r)
        self.event_tau_a = float(event_tau_a)
        self.event_tau_h = float(event_tau_h)
        self.event_tau_d = float(event_tau_d)
        self.event_lambda_0 = float(event_lambda_0)
        self.event_lambda_max = float(event_lambda_max)
        self.event_k_rho = float(event_k_rho)
        self.event_rho_max = float(event_rho_max)
        self.event_cooldown = int(event_cooldown)
        self.online_fisher_beta = float(online_fisher_beta)
        self.online_fisher_alpha = float(online_fisher_alpha)

        default_sq_factor = 1.0 - self.restore_strength
        self.adam_exp_avg_sq_decay_factor = (
            default_sq_factor
            if adam_exp_avg_sq_decay_factor is None
            else float(adam_exp_avg_sq_decay_factor)
        )
        if not 0.0 <= self.adam_exp_avg_sq_decay_factor <= 1.0:
            raise ValueError("adam_exp_avg_sq_decay_factor must be in [0, 1]")
        self.params = self._collect_trainable_norm_parameters(model)
        if not self.params:
            raise RuntimeError(
                "sparse restoration requires trainable normalization affine parameters"
            )

        missing = sorted(set(self.params) - set(fishers))
        if missing:
            raise KeyError(
                f"statistics file is missing Fisher values for: {missing[:10]}"
            )

        self.anchors = {
            name: param.detach().clone() for name, param in self.params.items()
        }
        self.total_trainable_elements = sum(
            param.numel() for param in self.params.values()
        )
        (
            self._parameter_sizes,
            self._flat_fishers,
            self._fisher_ranking,
        ) = self._build_fisher_ranking(fishers)
        self._global_offsets = []
        _off = 0
        for _s in self._parameter_sizes:
            self._global_offsets.append(_off)
            _off += _s

        # Online gradient second-moment tracking for Hybrid Fisher
        self._online_fisher_accumulators = {
            name: torch.zeros(param.numel(), dtype=torch.float32)
            for name, param in self.params.items()
        }

        # Rolling statistics for entropy anomaly detection
        self.entropy_mean_ema: Optional[float] = None
        self.entropy_var_ema: Optional[float] = None
        self.entropy_ema_decay: float = 0.95
        self.event_consecutive_count: int = 0
        self.cooldown_remaining: int = 0
        self.num_event_triggers: int = 0
        self.last_event_triggered: bool = False
        self.last_z_entropy: Optional[float] = None
        self.last_elastic_strength: Optional[float] = None
        self.last_anchor_disagreement: Optional[float] = None
        self.momentum_zero_count: int = 0
        self.momentum_decay_count: int = 0
        self.momentum_away_damped_count: int = 0
        self.momentum_towards_kept_count: int = 0

        initial_ratio = (
            self.sparse_ratio
            if self.budget_mode == "fixed"
            else self.drift_ratios[0]
        )
        self.selected = self._select_high_fisher(initial_ratio)
        self.selected_count = sum(
            indices.numel() for indices in self.selected.values()
        )
        maximum_ratio = (
            self.sparse_ratio
            if self.budget_mode == "fixed"
            else max(self.drift_ratios + ((self.event_rho_max,) if self.event_trigger else ()))
        )
        self.maximum_selected_count = self._count_for_ratio(maximum_ratio)

        self.num_batches = 0
        self.num_scheduled_events = 0
        self.num_triggers = 0
        self.num_recovered_elements = 0
        self.last_recovered_elements = 0
        self.num_optimizer_state_decays = 0
        self.num_skipped_low_reliability = 0
        self.reliability_ema: Optional[float] = None
        self._restored_once = {
            name: torch.zeros(param.numel(), dtype=torch.bool)
            for name, param in self.params.items()
        }
        self._layer_assignments = {name: 0 for name in self.params}
        self.last_drift_before = None
        self.last_drift_after = None
        self.last_trigger_drift = None
        self.last_reliability = None
        self.last_recovery_credit = None
        self.last_base_selected_ratio = initial_ratio
        self.last_selected_ratio = initial_ratio
        self.last_requested_elements = self.selected_count
        self.last_positive_score_elements = None
        self.last_selected_drift_energy_ratio = None
        self.last_selection_jaccard = None
        self.last_fisher_rms = None
        self.last_parameter_drift_rms = None
        self.last_drift_reduction = None
        self.last_relative_drift_reduction = None
        self._previous_restored_global_mask = None
        self._selection_jaccard_sum = 0.0
        self._selection_jaccard_count = 0
        self._budget_event_counts = {
            self._ratio_key(ratio): 0
            for ratio in (
                (self.sparse_ratio,)
                if self.budget_mode == "fixed"
                else self.drift_ratios
            )
        }

    @staticmethod
    def _collect_trainable_norm_parameters(model: nn.Module) -> Dict[str, nn.Parameter]:
        """Collect trainable affine parameters from BatchNorm2d/1d, LayerNorm, and GroupNorm."""
        norm_types = (nn.BatchNorm2d, nn.BatchNorm1d, nn.LayerNorm, nn.GroupNorm)
        params = {}
        for module_name, module in model.named_modules():
            if not isinstance(module, norm_types):
                continue
            for param_name, param in module.named_parameters(recurse=False):
                if (
                    param_name in ("weight", "bias")
                    and param is not None
                    and param.requires_grad
                ):
                    params[f"{module_name}.{param_name}"] = param
        return params

    @staticmethod
    def _collect_trainable_bn_affine(model: nn.Module) -> Dict[str, nn.Parameter]:
        """Backward-compatible alias for _collect_trainable_norm_parameters."""
        return PeriodicSparseRestorer._collect_trainable_norm_parameters(model)

    def _build_fisher_ranking(
        self, fishers: Mapping[str, torch.Tensor]
    ):
        flat_values = []
        sizes = []
        for name, param in self.params.items():
            value = fishers[name]
            if isinstance(value, (tuple, list)):
                value = value[0]
            value = torch.as_tensor(value, dtype=torch.float32).detach().flatten()
            if value.numel() != param.numel():
                raise ValueError(
                    f"Fisher shape mismatch for {name}: "
                    f"fisher={value.numel()}, parameter={param.numel()}"
                )
            if not torch.isfinite(value).all() or (value < 0).any():
                raise ValueError(f"Fisher values must be finite and non-negative: {name}")
            flat_values.append(value)
            sizes.append(param.numel())

        all_values = torch.cat(flat_values)
        ranking = torch.argsort(all_values, descending=True)
        return sizes, all_values, ranking

    def update_online_gradients(self, healthy: bool = True):
        """Update test-time gradient second moments for Hybrid Fisher only on healthy batches."""
        if self.online_fisher_beta <= 0 or not healthy:
            return
        for name, param in self.params.items():
            if param.grad is not None:
                grad_sq = param.grad.detach().flatten().float().cpu().square()
                self._online_fisher_accumulators[name] = (
                    self.online_fisher_alpha * self._online_fisher_accumulators[name]
                    + (1.0 - self.online_fisher_alpha) * grad_sq
                )

    def _get_hybrid_fishers(self) -> torch.Tensor:
        """Blend static source Fisher with online gradient sensitivity."""
        if self.online_fisher_beta <= 0:
            return self._flat_fishers
        online_list = [
            self._online_fisher_accumulators[name] for name in self.params
        ]
        flat_online = torch.cat(online_list)
        online_rms = flat_online.square().mean().sqrt().clamp_min(1e-8)
        norm_online = flat_online / online_rms

        source_rms = self._flat_fishers.square().mean().sqrt().clamp_min(1e-8)
        norm_source = self._flat_fishers / source_rms

        hybrid = (
            (1.0 - self.online_fisher_beta) * norm_source
            + self.online_fisher_beta * norm_online
        )
        return hybrid

    def _count_for_ratio(self, ratio: float) -> int:
        if ratio <= 0:
            return 0
        return min(
            self.total_trainable_elements,
            max(1, math.ceil(round(ratio * self.total_trainable_elements, 6))),
        )

    def _global_indices_to_selection(
        self, global_indices: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        selected = {}
        start = 0
        for (name, _param), size in zip(
            self.params.items(), self._parameter_sizes
        ):
            local = global_indices[
                (global_indices >= start) & (global_indices < start + size)
            ] - start
            selected[name] = local
            start += size
        return selected

    def _select_high_fisher(self, ratio: float) -> Dict[str, torch.Tensor]:
        """Preserve the legacy fixed global High-Fisher prefix."""
        count = self._count_for_ratio(ratio)
        return self._global_indices_to_selection(self._fisher_ranking[:count])

    def _current_absolute_drift(self) -> torch.Tensor:
        drift = []
        for name, param in self.params.items():
            anchor = self.anchors[name].to(
                device=param.device, dtype=param.dtype
            )
            drift.append(
                (param.detach() - anchor).abs().flatten().float().cpu()
            )
        return torch.cat(drift)

    def _select_fisher_drift_indices(
        self, ratio: float, absolute_drift: torch.Tensor
    ) -> torch.Tensor:
        """Rank current candidates by the selected FSLR ablation criterion."""
        count = self._count_for_ratio(ratio)
        if count == 0:
            self.last_positive_score_elements = 0
            self.last_fisher_rms = None
            self.last_parameter_drift_rms = None
            return torch.empty(0, dtype=torch.long)

        if self.selection_criterion == "random":
            g = torch.Generator()
            if self.selection_seed is not None:
                g.manual_seed(int(self.selection_seed + self.num_batches * 1000))
            perm = torch.randperm(self.total_trainable_elements, generator=g)
            selected_count = min(count, self.total_trainable_elements)
            self.last_positive_score_elements = self.total_trainable_elements
            return perm[:selected_count]

        effective_fishers = self._get_hybrid_fishers()
        fisher_rms = float(effective_fishers.square().mean().sqrt())
        drift_rms = float(absolute_drift.square().mean().sqrt())
        self.last_fisher_rms = fisher_rms
        self.last_parameter_drift_rms = drift_rms

        if self.selection_criterion == "drift_only":
            if drift_rms <= 0:
                self.last_positive_score_elements = 0
                return torch.empty(0, dtype=torch.long)
            scores = absolute_drift / (drift_rms + 1e-8)
        elif self.selection_criterion == "fisher_only":
            if fisher_rms <= 0:
                self.last_positive_score_elements = 0
                return torch.empty(0, dtype=torch.long)
            scores = effective_fishers / (fisher_rms + 1e-8)
        elif self.selection_criterion == "fisher_drift":
            if fisher_rms <= 0:
                self.last_positive_score_elements = 0
                return torch.empty(0, dtype=torch.long)
            if drift_rms <= 0:
                scores = effective_fishers / (fisher_rms + 1e-8)
            else:
                scores = (effective_fishers / (fisher_rms + 1e-8)) * (
                    absolute_drift / (drift_rms + 1e-8)
                )
        else:
            raise ValueError(f"unknown selection_criterion: {self.selection_criterion}")

        # Optional Layer-aware score modulation
        if self.layer_aware and self.layer_weights:
            layer_multipliers = []
            for (name, _param), size in zip(self.params.items(), self._parameter_sizes):
                matched_weight = 1.0
                for pattern, weight in self.layer_weights.items():
                    if pattern in name:
                        matched_weight = float(weight)
                        break
                layer_multipliers.append(torch.full((size,), matched_weight, dtype=torch.float32))
            scores = scores * torch.cat(layer_multipliers)

        positive = torch.isfinite(scores) & (scores > 0)
        self.last_positive_score_elements = int(positive.sum().item())
        eligible = positive.nonzero(as_tuple=False).flatten()
        if eligible.numel() == 0:
            return torch.empty(0, dtype=torch.long)

        selected_count = min(count, int(eligible.numel()))
        eligible_scores = scores.index_select(0, eligible)
        ranking = torch.argsort(eligible_scores, descending=True)
        return eligible.index_select(0, ranking[:selected_count])

    def _select_hierarchical_layer_indices(
        self, ratio: float, absolute_drift: torch.Tensor
    ) -> torch.Tensor:
        """Two-Stage Hierarchical Budget Allocation across layers."""
        count = self._count_for_ratio(ratio)
        if count == 0:
            self.last_positive_score_elements = 0
            self.last_fisher_rms = None
            self.last_parameter_drift_rms = None
            return torch.empty(0, dtype=torch.long)

        effective_fishers = self._get_hybrid_fishers()
        drift_rms = absolute_drift.square().mean().sqrt()
        fisher_rms = effective_fishers.square().mean().sqrt()
        self.last_fisher_rms = float(fisher_rms)
        self.last_parameter_drift_rms = float(drift_rms)

        layer_scores = []
        layer_risk_masses = []

        # 1. Compute layer-wise normalized scores s_{l, i} and risk mass M_l
        for (name, param), size, offset in zip(
            self.params.items(), self._parameter_sizes, self._global_offsets
        ):
            f_l = effective_fishers[offset : offset + size]
            d_l = absolute_drift[offset : offset + size]

            f_rms_l = f_l.square().mean().sqrt()
            d_rms_l = d_l.square().mean().sqrt()

            if f_rms_l <= 1e-12 or d_rms_l <= 1e-12:
                s_l = torch.zeros_like(f_l)
                m_l = 0.0
            else:
                s_l = (f_l / (f_rms_l + 1e-8)) * (d_l / (d_rms_l + 1e-8))
                m_l = float(s_l.sum().item())

            layer_scores.append(s_l)
            layer_risk_masses.append(max(0.0, m_l))

        total_risk_mass = sum(layer_risk_masses)
        if total_risk_mass <= 1e-12:
            self.last_positive_score_elements = 0
            return torch.empty(0, dtype=torch.long)

        # 2. Allocate layer quotas K_l based on normalized risk mass
        proportions = [m / total_risk_mass for m in layer_risk_masses]
        quotas_float = [count * p for p in proportions]
        quotas_int = [int(math.floor(q)) for q in quotas_float]

        # Apply layer cap: K_l <= size * layer_quota_cap
        for i, (size, q) in enumerate(zip(self._parameter_sizes, quotas_int)):
            max_allowed = max(1, int(size * self.layer_quota_cap))
            if q > max_allowed:
                quotas_int[i] = max_allowed

        remainder = count - sum(quotas_int)
        if remainder > 0:
            fractional_parts = [
                (q_f - q_i, i)
                for i, (q_f, q_i) in enumerate(zip(quotas_float, quotas_int))
                if quotas_int[i] < max(1, int(self._parameter_sizes[i] * self.layer_quota_cap))
            ]
            fractional_parts.sort(key=lambda x: x[0], reverse=True)
            for _, i in fractional_parts[:remainder]:
                quotas_int[i] += 1

        # 3. Pick top-K_l elements within each layer
        selected_global_indices = []
        total_positive = 0

        for (name, param), size, offset, s_l, k_l in zip(
            self.params.items(), self._parameter_sizes, self._global_offsets, layer_scores, quotas_int
        ):
            pos = torch.isfinite(s_l) & (s_l > 0)
            total_positive += int(pos.sum().item())
            if k_l <= 0:
                continue
            eligible_local = pos.nonzero(as_tuple=False).flatten()
            if eligible_local.numel() == 0:
                continue
            actual_pick = min(k_l, int(eligible_local.numel()))
            scores_eligible = s_l.index_select(0, eligible_local)
            ranking = torch.argsort(scores_eligible, descending=True)
            chosen_local = eligible_local.index_select(0, ranking[:actual_pick])
            selected_global_indices.append(chosen_local + offset)

        self.last_positive_score_elements = total_positive
        if not selected_global_indices:
            return torch.empty(0, dtype=torch.long)
        return torch.cat(selected_global_indices)

    def _record_selection_diagnostics(
        self,
        global_indices: torch.Tensor,
        absolute_drift: torch.Tensor,
    ):
        drift_energy = absolute_drift.square()
        total_energy = float(drift_energy.sum().item())
        if global_indices.numel() == 0 or total_energy <= 0:
            self.last_selected_drift_energy_ratio = 0.0
            self.last_selection_jaccard = None
            return

        selected_energy = float(
            drift_energy.index_select(0, global_indices).sum().item()
        )
        self.last_selected_drift_energy_ratio = float(
            selected_energy / total_energy
        )

        current_mask = torch.zeros(
            self.total_trainable_elements, dtype=torch.bool
        )
        current_mask[global_indices] = True
        if self._previous_restored_global_mask is None:
            self.last_selection_jaccard = None
        else:
            intersection = int(
                (current_mask & self._previous_restored_global_mask).sum().item()
            )
            union = int(
                (current_mask | self._previous_restored_global_mask).sum().item()
            )
            self.last_selection_jaccard = float(intersection / max(1, union))
            self._selection_jaccard_sum += self.last_selection_jaccard
            self._selection_jaccard_count += 1
        self._previous_restored_global_mask = current_mask

    def _ratio_for_drift(self, drift: float) -> float:
        if self.budget_mode == "fixed":
            return self.sparse_ratio
        for threshold, ratio in zip(
            self.drift_thresholds, self.drift_ratios
        ):
            if drift < threshold:
                return ratio
        return self.drift_ratios[-1]

    @staticmethod
    def _ratio_key(ratio: float) -> str:
        return f"{ratio:.6g}"

    @torch.no_grad()
    def step(
        self,
        reliability: Optional[float] = None,
        entropy: Optional[float] = None,
        anchor_disagreement: Optional[float] = None,
        gradient_clipped: bool = False,
        ess_ratio: Optional[float] = None,
        **kwargs,
    ) -> int:
        """Restore Fisher-ranked drift budget with periodic schedule or triple-condition event."""
        # Determine whether current batch is healthy
        is_healthy_batch = (
            (reliability is None or reliability >= self.event_tau_r)
            and (anchor_disagreement is None or anchor_disagreement <= self.event_tau_a)
        )

        # Update online Fisher ONLY on healthy batches
        self.update_online_gradients(healthy=is_healthy_batch)

        if reliability is not None:
            reliability = float(reliability)
            if not math.isfinite(reliability) or not 0.0 <= reliability <= 1.0:
                raise ValueError("reliability must be finite and in [0, 1]")
            self.last_reliability = reliability
            self.reliability_ema = (
                reliability
                if self.reliability_ema is None
                else self.reliability_ema_decay * self.reliability_ema
                + (1.0 - self.reliability_ema_decay) * reliability
            )
            self.last_recovery_credit = reliability
        else:
            self.last_reliability = None
            self.last_recovery_credit = None

        if anchor_disagreement is not None:
            self.last_anchor_disagreement = float(anchor_disagreement)

        # Track entropy EMA and z-score (Freeze baseline during anomalies/events)
        z_entropy = 0.0
        if entropy is not None and math.isfinite(float(entropy)):
            entropy = float(entropy)
            if self.entropy_mean_ema is None:
                self.entropy_mean_ema = entropy
                self.entropy_var_ema = 0.01
            elif is_healthy_batch:
                delta = entropy - self.entropy_mean_ema
                self.entropy_mean_ema = (
                    self.entropy_ema_decay * self.entropy_mean_ema
                    + (1.0 - self.entropy_ema_decay) * entropy
                )
                self.entropy_var_ema = (
                    self.entropy_ema_decay * self.entropy_var_ema
                    + (1.0 - self.entropy_ema_decay) * (delta ** 2)
                )
            entropy_std = math.sqrt(max(self.entropy_var_ema, 1e-6))
            z_entropy = (entropy - self.entropy_mean_ema) / entropy_std
            self.last_z_entropy = z_entropy

        self.num_batches += 1
        self.last_recovered_elements = 0

        # Measure current drift
        self.last_drift_before = self.drift_stats()
        current_drift = self.last_drift_before["total"]
        self.last_trigger_drift = current_drift

        # Evaluate Triple-Condition: [Low Reliability OR High Anchor Disagreement] AND [Abnormally Low Entropy] AND [High Drift]
        event_matched = False
        if self.event_trigger and entropy is not None:
            cond_input = (
                (reliability is not None and reliability < self.event_tau_r)
                or (anchor_disagreement is not None and anchor_disagreement > self.event_tau_a)
            )
            cond_entropy = z_entropy < -self.event_tau_h
            cond_drift = current_drift > self.event_tau_d

            if cond_input and cond_entropy and cond_drift:
                self.event_consecutive_count += 1
            else:
                self.event_consecutive_count = 0

            in_cooldown = (self.cooldown_remaining > 0)
            event_matched = (
                (self.event_consecutive_count >= self.event_k)
                and not in_cooldown
            )

        self.last_event_triggered = event_matched

        is_periodic_step = (self.num_batches % self.interval == 0)
        if not is_periodic_step and not event_matched:
            if self.cooldown_remaining > 0:
                self.cooldown_remaining -= 1
            return 0

        self.num_scheduled_events += 1

        # Check standard low-reliability rejection if NOT in emergency event mode or continuous risk mode
        if (
            not event_matched
            and not self.disable_reliability_gate
            and self.budget_mode != "continuous_risk"
            and reliability is not None
            and reliability < self.reliability_threshold
        ):
            self.num_skipped_low_reliability += 1
            self.last_base_selected_ratio = self._ratio_for_drift(current_drift)
            self.last_selected_ratio = 0.0
            self.last_requested_elements = 0
            self.last_positive_score_elements = 0
            self.last_fisher_rms = None
            self.last_parameter_drift_rms = None
            self.selected = self._global_indices_to_selection(
                torch.empty(0, dtype=torch.long)
            )
            self.selected_count = 0
            self.last_selected_drift_energy_ratio = 0.0
            self.last_selection_jaccard = None
            self.last_drift_after = dict(self.last_drift_before)
            self.last_drift_reduction = 0.0
            self.last_relative_drift_reduction = 0.0
            if self.cooldown_remaining > 0:
                self.cooldown_remaining -= 1
            return 0

        # Determine Restore Ratio and Strength
        if event_matched:
            self.num_event_triggers += 1
            self.event_consecutive_count = 0  # reset consecutive window
            self.cooldown_remaining = self.event_cooldown  # enter cooldown

            # Dual-boosted: simultaneously raise budget ratio and elastic strength
            base_ratio = self._ratio_for_drift(current_drift)
            selected_ratio = min(
                self.event_rho_max,
                self.event_k_rho * max(base_ratio, 0.05)
            )
            elastic_strength = (
                self.event_lambda_0
                + 0.3 * (1.0 - (reliability if reliability is not None else 0.5))
                + 0.1 * (-z_entropy)
            )
            strength = float(
                max(self.event_lambda_0, min(self.event_lambda_max, elastic_strength))
            )
            self.last_elastic_strength = strength
        elif self.budget_mode == "continuous_risk":
            if self.cooldown_remaining > 0:
                self.cooldown_remaining -= 1
            ess_comp = (1.0 - ess_ratio) if ess_ratio is not None else (1.0 - (reliability if reliability is not None else 0.5))
            anchor_comp = anchor_disagreement if anchor_disagreement is not None else 0.0
            hazard_score = float(max(0.0, min(1.0,
                self.risk_gamma_ess * ess_comp +
                self.risk_gamma_anchor * anchor_comp
            )))
            drift_factor = float(max(0.0, min(1.0, current_drift / max(1e-6, self.risk_tau_drift))))
            risk_score = float(max(0.0, min(1.0, hazard_score * drift_factor)))
            selected_ratio = self.risk_rho_min + (self.risk_rho_max - self.risk_rho_min) * risk_score
            base_ratio = selected_ratio
            strength = float(self.restore_strength * (0.5 + 0.5 * risk_score))
            self.last_risk_score = risk_score
            self.last_elastic_strength = strength
        else:
            if self.cooldown_remaining > 0:
                self.cooldown_remaining -= 1
            base_ratio = self._ratio_for_drift(current_drift)
            if self.disable_reliability_gate or reliability is None:
                selected_ratio = base_ratio
            else:
                selected_ratio = base_ratio * reliability
            strength = self.restore_strength
            self.last_elastic_strength = strength

        self.last_base_selected_ratio = base_ratio
        self.last_selected_ratio = selected_ratio
        self.last_requested_elements = self._count_for_ratio(selected_ratio)
        absolute_drift = self._current_absolute_drift()

        if self.hierarchical_budget and not event_matched:
            global_indices = self._select_hierarchical_layer_indices(
                selected_ratio, absolute_drift
            )
        else:
            global_indices = self._select_fisher_drift_indices(
                selected_ratio, absolute_drift
            )

        self.selected = self._global_indices_to_selection(global_indices)
        self.selected_count = sum(
            indices.numel() for indices in self.selected.values()
        )
        ratio_key = self._ratio_key(selected_ratio)
        self._budget_event_counts[ratio_key] = (
            self._budget_event_counts.get(ratio_key, 0) + 1
        )

        if self.selected_count == 0:
            self.last_drift_after = dict(self.last_drift_before)
            self.last_selected_drift_energy_ratio = 0.0
            self.last_selection_jaccard = None
            self.last_drift_reduction = 0.0
            self.last_relative_drift_reduction = 0.0
            return 0

        self._record_selection_diagnostics(global_indices, absolute_drift)
        recovered = 0
        optimizer_state_decays = 0

        # Tiered momentum handling: 2% light restore decays momentum; heavy / event / clipped zeroes momentum
        is_heavy_restore = (selected_ratio > 0.02) or event_matched or gradient_clipped

        for name, param in self.params.items():
            local = self.selected[name].to(param.device)
            if local.numel() == 0:
                continue

            flat_param = param.view(-1)
            flat_anchor = self.anchors[name].to(
                device=param.device, dtype=param.dtype
            ).view(-1)
            flat_delta = flat_param - flat_anchor
            flat_param[local] = (
                (1.0 - strength) * flat_param[local]
                + strength * flat_anchor[local]
            )

            if self.momentum_discounting:
                momentum = self.optimizer.state.get(param, {}).get("momentum_buffer")
                if momentum is not None:
                    flat_momentum = momentum.view(-1)
                    if gradient_clipped:
                        flat_momentum[local] = 0.0
                        self.momentum_zero_count += int(local.numel())
                    elif event_matched:
                        m_loc = flat_momentum[local]
                        d_loc = flat_delta[local]
                        is_pushing_away = (m_loc * d_loc) > 0
                        gamma = torch.where(
                            is_pushing_away,
                            torch.zeros_like(m_loc),
                            torch.full_like(m_loc, max(0.0, 1.0 - strength)),
                        )
                        flat_momentum[local] = gamma * m_loc
                        self.momentum_zero_count += int(is_pushing_away.sum().item())
                        self.momentum_towards_kept_count += int((~is_pushing_away).sum().item())
                    else:
                        m_loc = flat_momentum[local]
                        d_loc = flat_delta[local]
                        is_pushing_away = (m_loc * d_loc) > 0
                        base_gamma = max(0.0, 1.0 - strength)
                        away_gamma = base_gamma * self.momentum_away_factor
                        gamma = torch.where(
                            is_pushing_away,
                            torch.full_like(m_loc, away_gamma),
                            torch.full_like(m_loc, base_gamma),
                        )
                        flat_momentum[local] = gamma * m_loc
                        self.momentum_away_damped_count += int(is_pushing_away.sum().item())
                        self.momentum_towards_kept_count += int((~is_pushing_away).sum().item())
                        self.momentum_decay_count += int(local.numel())
                    optimizer_state_decays += int(local.numel())

                exp_avg = self.optimizer.state.get(param, {}).get("exp_avg")
                if exp_avg is not None:
                    flat_exp_avg = exp_avg.view(-1)
                    if gradient_clipped:
                        flat_exp_avg[local] = 0.0
                    elif event_matched:
                        m_loc = flat_exp_avg[local]
                        d_loc = flat_delta[local]
                        is_pushing_away = (m_loc * d_loc) > 0
                        gamma = torch.where(
                            is_pushing_away,
                            torch.zeros_like(m_loc),
                            torch.full_like(m_loc, max(0.0, 1.0 - strength)),
                        )
                        flat_exp_avg[local] = gamma * m_loc
                    else:
                        m_loc = flat_exp_avg[local]
                        d_loc = flat_delta[local]
                        is_pushing_away = (m_loc * d_loc) > 0
                        base_gamma = max(0.0, 1.0 - strength)
                        away_gamma = base_gamma * self.momentum_away_factor
                        gamma = torch.where(
                            is_pushing_away,
                            torch.full_like(m_loc, away_gamma),
                            torch.full_like(m_loc, base_gamma),
                        )
                        flat_exp_avg[local] = gamma * m_loc
                    optimizer_state_decays += int(local.numel())

                exp_avg_sq = self.optimizer.state.get(param, {}).get("exp_avg_sq")
                if exp_avg_sq is not None:
                    flat_exp_avg_sq = exp_avg_sq.view(-1)
                    if gradient_clipped or event_matched:
                        flat_exp_avg_sq[local] = 0.0
                    elif self.adam_exp_avg_sq_decay:
                        flat_exp_avg_sq[local] = (
                            self.adam_exp_avg_sq_decay_factor * flat_exp_avg_sq[local]
                        )
                    optimizer_state_decays += int(local.numel())
            else:
                momentum = self.optimizer.state.get(param, {}).get("momentum_buffer")
                if momentum is not None:
                    flat_momentum = momentum.view(-1)
                    if is_heavy_restore:
                        flat_momentum[local] = 0.0
                        self.momentum_zero_count += int(local.numel())
                    else:
                        flat_momentum[local] = (1.0 - strength) * flat_momentum[local]
                        self.momentum_decay_count += int(local.numel())
                    optimizer_state_decays += int(local.numel())

                exp_avg = self.optimizer.state.get(param, {}).get("exp_avg")
                if exp_avg is not None:
                    flat_exp_avg = exp_avg.view(-1)
                    if is_heavy_restore:
                        flat_exp_avg[local] = 0.0
                    elif self.adam_exp_avg_decay:
                        flat_exp_avg[local] = (
                            (1.0 - strength) * flat_exp_avg[local]
                        )
                    optimizer_state_decays += int(local.numel())

                exp_avg_sq = self.optimizer.state.get(param, {}).get("exp_avg_sq")
                if exp_avg_sq is not None:
                    flat_exp_avg_sq = exp_avg_sq.view(-1)
                    if is_heavy_restore:
                        flat_exp_avg_sq[local] = 0.0
                    elif self.adam_exp_avg_sq_decay:
                        flat_exp_avg_sq[local] = (
                            self.adam_exp_avg_sq_decay_factor * flat_exp_avg_sq[local]
                        )
                    optimizer_state_decays += int(local.numel())

            local_count = int(local.numel())
            self._restored_once[name][self.selected[name].cpu()] = True
            self._layer_assignments[name] += local_count
            recovered += local_count

        self.num_triggers += 1
        self.num_recovered_elements += recovered
        self.num_optimizer_state_decays += optimizer_state_decays
        self.last_recovered_elements = recovered
        self.last_drift_after = self.drift_stats()
        self.last_drift_reduction = float(
            self.last_drift_before["total"] - self.last_drift_after["total"]
        )
        self.last_relative_drift_reduction = float(
            self.last_drift_reduction
            / max(self.last_drift_before["total"], 1e-12)
        )
        return recovered

    @torch.no_grad()
    def drift_stats(self):
        """Return normalized affine drift from the deployment anchor."""
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
            name: float(math.sqrt(delta_square) / (math.sqrt(anchor_square) + 1e-12))
            for name, (delta_square, anchor_square) in accumulators.items()
        }

    @property
    def stats(self):
        unique_count = sum(
            int(mask.sum().item()) for mask in self._restored_once.values()
        )
        per_layer = {}
        for name, param in self.params.items():
            selected = int(self.selected[name].numel())
            unique = int(self._restored_once[name].sum().item())
            total = int(param.numel())
            assignments = int(self._layer_assignments[name])
            per_layer[name] = {
                "trainable_elements": total,
                "selected_elements": selected,
                "selection_ratio": float(selected / max(1, total)),
                "restored_assignments_total": assignments,
                "restored_unique_parameters": unique,
                "repeat_restore_mean": float(assignments / max(1, unique)),
            }
        return {
            "budget_mode": self.budget_mode,
            "interval": self.interval,
            "sparse_ratio": self.sparse_ratio,
            "selection_criterion": self.selection_criterion,
            "selection_seed": self.selection_seed,
            "reliability_gate_disabled": self.disable_reliability_gate,
            "drift_thresholds": list(self.drift_thresholds),
            "drift_ratios": list(self.drift_ratios),
            "restore_strength": self.restore_strength,
            "total_trainable_elements": self.total_trainable_elements,
            "selected_elements": self.selected_count,
            "maximum_selected_elements": self.maximum_selected_count,
            "batches": self.num_batches,
            "scheduled_restore_events": self.num_scheduled_events,
            "restore_events": self.num_triggers,
            "num_event_triggers": self.num_event_triggers,
            "last_event_triggered": self.last_event_triggered,
            "last_z_entropy": self.last_z_entropy,
            "last_elastic_strength": self.last_elastic_strength,
            "last_anchor_disagreement": self.last_anchor_disagreement,
            "last_risk_score": self.last_risk_score,
            "cooldown_remaining": self.cooldown_remaining,
            "momentum_zero_count": self.momentum_zero_count,
            "momentum_decay_count": self.momentum_decay_count,
            "skipped_low_drift_events": (
                self.num_scheduled_events
                - self.num_triggers
                - self.num_skipped_low_reliability
            ),
            "skipped_low_reliability_events": self.num_skipped_low_reliability,
            "budget_event_counts": dict(self._budget_event_counts),
            "recovered_elements_total": self.num_recovered_elements,
            "restored_assignments_total": self.num_recovered_elements,
            "restored_unique_parameters": unique_count,
            "repeat_restore_mean": float(
                self.num_recovered_elements / max(1, unique_count)
            ),
            "last_recovered_elements": self.last_recovered_elements,
            "last_requested_elements": self.last_requested_elements,
            "last_base_selected_ratio": self.last_base_selected_ratio,
            "last_selected_ratio": self.last_selected_ratio,
            "last_positive_score_elements": self.last_positive_score_elements,
            "last_selected_drift_energy_ratio": (
                self.last_selected_drift_energy_ratio
            ),
            "last_selection_jaccard": self.last_selection_jaccard,
            "mean_selection_jaccard": (
                float(
                    self._selection_jaccard_sum
                    / max(1, self._selection_jaccard_count)
                )
                if self._selection_jaccard_count > 0
                else None
            ),
            "last_fisher_rms": self.last_fisher_rms,
            "last_parameter_drift_rms": self.last_parameter_drift_rms,
            "last_drift_before": self.last_drift_before,
            "last_drift_after": self.last_drift_after,
            "last_drift_reduction": self.last_drift_reduction,
            "last_relative_drift_reduction": (
                self.last_relative_drift_reduction
            ),
            "per_layer": per_layer,
        }
