"""
FedTTA Server: Domain-Isolated Federated TTA Aggregation.

Core design:
  - Each corruption/domain maintains independent adapter and momentum.
  - Different domains NEVER mix — no shared adapter, no cross-domain projection.
  - Same-domain clients collaborate via FedAvg or FedACG-style aggregation.

Two modes:
  - domain_fedavg:  phi_k = phi_k + Delta_k           (no momentum)
  - domain_fedacg:  m_k = lambda * m_k + Delta_k
                     phi_k = phi_k + m_k              (Nesterov momentum)

Formula per round for domain k:
  1. Lookahead:          phi_acc = phi_k + lambda * m_k   (only in fedacg mode)
  2. Client delta:       delta_i = phi_i_after - phi_acc
  3. Aggregation:        Delta_k = weighted_avg(delta_i)
  4. Momentum update:    m_k = lambda * m_k + Delta_k     (only in fedacg mode)
  5. Adapter update:     phi_k = phi_k + Delta_k          (fedavg)
                         phi_k = phi_k + m_k              (fedacg)
"""

import copy
import os
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from .adapter_utils import (
    extract_adapt_state,
    add_state,
    scale_state,
    weighted_average_delta,
    zeros_like_state,
    count_adapt_params,
)
from utils.metrics import compute_client_reliability


class FedTTAServer:
    """
    Domain-Isolated FedTTA Server.

    Each corruption/domain has its own adapter and optional momentum.
    Cross-domain mixing is strictly avoided.

    Args:
        base_model: trained source model (for initial adapter extraction)
        device: computation device
        configs: configuration namespace
    """

    def __init__(
        self,
        base_model: nn.Module,
        device: torch.device,
        configs,
        use_momentum: Optional[bool] = None,
    ):
        """
        Args:
            base_model: trained source model (for initial adapter extraction)
            device: computation device
            configs: configuration namespace
            use_momentum: if True, use FedACG-style momentum.
                          If None, auto-detect from configs.tta_mode.
        """
        self.configs = configs
        self.device = device

        if use_momentum is None:
            self.use_momentum = getattr(self.configs, "tta_mode", "") in (
                "naive_fedacg_tta", "domain_fedacg_eatac"
            )
        else:
            self.use_momentum = use_momentum

        # Source model for adapter initialization
        self.base_model = copy.deepcopy(base_model).to(device)
        self.base_model.eval()

        # Source adapter (initial state for all domains)
        self.source_adapt_state = extract_adapt_state(self.base_model)

        # Domain-isolated adapters and momentums
        # Each domain k: domain_adapters[k] = current phi_k
        #                domain_momentums[k] = current m_k
        self.domain_adapters: Dict[int, Dict] = {}
        self.domain_momentums: Dict[int, Dict] = {}

        # Domain metadata
        self.domain_info: Dict[int, Dict] = {}

    def _get_or_init_domain(self, domain_id: int, corruption: str):
        """Initialize per-domain adapter and momentum if first encounter."""
        if domain_id not in self.domain_adapters:
            self.domain_adapters[domain_id] = copy.deepcopy(self.source_adapt_state)
            self.domain_momentums[domain_id] = zeros_like_state(self.source_adapt_state)
            self.domain_info[domain_id] = {
                "corruption": corruption,
                "num_clients": 0,
                "total_samples": 0,
            }

    def get_accelerated_state(self, domain_id: int) -> Dict:
        """
        Compute accelerated adapter for a domain (FedACG lookahead).

        phi_acc = phi_k + lambda * m_k

        When use_momentum=False, phi_acc = phi_k (no momentum).
        """
        phi_k = self.domain_adapters[domain_id]
        lambda_tta = getattr(self.configs, "lambda_tta", 0.85)

        if self.use_momentum:
            m_k = self.domain_momentums[domain_id]
            return add_state(phi_k, m_k, scale=lambda_tta)
        else:
            return copy.deepcopy(phi_k)

    def run_fedtta_round(
        self,
        target_clients: List[Dict],
        clients_to_run: List[int],
        client_objects: Dict[int, object],
    ) -> Dict:
        """
        Run one round of domain-isolated FedTTA.

        Phase 1: Send accelerated state to each selected client, collect results.
        Phase 2: Group results by domain, aggregate per domain independently.
        Phase 3: Compute round-level metrics.

        Args:
            target_clients: list of target client info dicts
            clients_to_run: list of client_ids to participate this round
            client_objects: dict mapping client_id -> FedTTAClient instance

        Returns:
            dict with round metrics
        """
        lambda_tta = getattr(self.configs, "lambda_tta", 0.85)
        weight_by_num_samples = getattr(self.configs, "weight_by_num_samples", False)
        tau = getattr(self.configs, "reliability_tau", 0.5)

        # Client info lookup
        client_info = {c["client_id"]: c for c in target_clients}

        # Phase 1: Dispatch and collect
        domain_reports = defaultdict(list)
        round_stats = []

        for cid in clients_to_run:
            info = client_info[cid]
            domain_id = info["domain_id"]
            corruption = info["corruption"]

            # Initialize domain if first time
            self._get_or_init_domain(domain_id, corruption)

            # Get accelerated adapter
            accelerated_state = self.get_accelerated_state(domain_id)

            # Run local EATA-C
            client_obj = client_objects[cid]
            result = client_obj.run_local_eatac_tta(
                init_adapt_state=accelerated_state,
                test_loader=info["test_loader"],
                num_batches=getattr(self.configs, "local_tta_batches", None),
                eval_after_adapt=getattr(self.configs, "eval_after_adapt", False),
            )

            domain_reports[domain_id].append(result)
            round_stats.append(result)
            self.domain_info[domain_id]["num_clients"] += 1
            self.domain_info[domain_id]["total_samples"] += result["num_samples"]

        # Phase 2: Domain-isolated aggregation
        domain_metrics = {}
        client_round_stats = []  # per-client stats for this round
        for domain_id, reports in domain_reports.items():
            # Prepare deltas
            deltas = [r["delta"] for r in reports]

            # Compute raw reliability scores
            raw_scores = np.array([
                compute_client_reliability(r["stats"]) for r in reports
            ], dtype=np.float32)

            # Per-domain temperature-controlled softmax
            # tau < 1.0 → sharper (more selective), tau > 1.0 → flatter
            shifted = (raw_scores - raw_scores.max()) / max(tau, 1e-8)
            exp_scores = np.exp(shifted)
            softmax_weights = exp_scores / (exp_scores.sum() + 1e-8)

            # Optionally weight by num_samples
            if weight_by_num_samples:
                num_samples_arr = np.array([r["num_samples"] for r in reports])
                agg_weights = (softmax_weights * num_samples_arr).tolist()
            else:
                agg_weights = softmax_weights.tolist()

            # Save per-client round stats for CSV logging
            for i, r in enumerate(reports):
                s = r["stats"]
                client_round_stats.append({
                    "client_id": r["client_id"],
                    "domain_id": r["domain_id"],
                    "corruption": r["corruption"],
                    "online_acc": s["accuracy"],
                    "online_ece": s["ece"],
                    "selected_ratio": s["selected_ratio"],
                    "mean_confidence": s["mean_confidence"],
                    "mean_entropy_norm": s["mean_entropy_norm"],
                    "collapse_score": s["collapse_score"],
                    "mean_disagreement": s["mean_disagreement"],
                    "reliability_raw": float(raw_scores[i]),
                    "aggregation_weight": float(agg_weights[i]),
                    "delta_norm": s["delta_norm"],
                    "forwards": s["forwards"],
                    "backwards": s["backwards"],
                    "num_samples": r["num_samples"],
                })

            # Weighted average delta for this domain
            delta_k = weighted_average_delta(deltas, agg_weights)

            if self.use_momentum:
                # FedACG: m_k = lambda * m_k + Delta_k
                old_m = self.domain_momentums[domain_id]
                new_m = add_state(scale_state(old_m, lambda_tta), delta_k, scale=1.0)
                self.domain_momentums[domain_id] = new_m

                # phi_k = phi_k + m_k
                old_adapter = self.domain_adapters[domain_id]
                self.domain_adapters[domain_id] = add_state(old_adapter, new_m, scale=1.0)
            else:
                # FedAvg: phi_k = phi_k + Delta_k
                old_adapter = self.domain_adapters[domain_id]
                self.domain_adapters[domain_id] = add_state(old_adapter, delta_k, scale=1.0)

            # Domain-level metrics
            domain_acc = sum(
                r["stats"]["accuracy"] * r["num_samples"] for r in reports
            ) / max(sum(r["num_samples"] for r in reports), 1)
            domain_ece = sum(
                r["stats"]["ece"] * r["num_samples"] for r in reports
            ) / max(sum(r["num_samples"] for r in reports), 1)

            # Average reliability stats for this domain
            avg_collapse = float(np.mean([r["stats"]["collapse_score"] for r in reports]))
            avg_reliability_raw = float(np.mean([r["stats"]["reliability_raw"] for r in reports]))
            avg_delta_norm = float(np.mean([r["stats"]["delta_norm"] for r in reports]))
            avg_selected = float(np.mean([r["stats"]["selected_ratio"] for r in reports]))
            avg_confidence = float(np.mean([r["stats"]["mean_confidence"] for r in reports]))
            avg_entropy_norm = float(np.mean([r["stats"]["mean_entropy_norm"] for r in reports]))

            domain_metrics[domain_id] = {
                "corruption": self.domain_info[domain_id]["corruption"],
                "domain_acc": domain_acc,
                "domain_ece": domain_ece,
                "num_clients": len(reports),
                "num_samples": sum(r["num_samples"] for r in reports),
                "raw_scores": raw_scores.tolist(),
                "agg_weights": agg_weights,
                "avg_selected_ratio": round(avg_selected, 4),
                "avg_confidence": round(avg_confidence, 4),
                "avg_entropy_norm": round(avg_entropy_norm, 4),
                "avg_collapse": round(avg_collapse, 4),
                "avg_reliability_raw": round(avg_reliability_raw, 4),
                "avg_delta_norm": round(avg_delta_norm, 6),
            }

        # Phase 3: Round-level metrics
        total_samples = sum(r["num_samples"] for r in round_stats) or 1
        avg_acc = sum(r["stats"]["accuracy"] * r["num_samples"] for r in round_stats) / total_samples
        avg_ece = sum(r["stats"]["ece"] * r["num_samples"] for r in round_stats) / total_samples
        avg_selected = sum(r["stats"]["selected_ratio"] * r["num_samples"] for r in round_stats) / total_samples
        avg_entropy = sum(r["stats"]["mean_entropy"] * r["num_samples"] for r in round_stats) / total_samples
        avg_disagreement = sum(r["stats"]["mean_disagreement"] * r["num_samples"] for r in round_stats) / total_samples
        avg_consistency = sum(r["stats"]["mean_consistency_loss"] * r["num_samples"] for r in round_stats) / total_samples
        avg_total_consistency = sum(r["stats"].get("total_consistency_loss", 0) for r in round_stats) / max(total_backwards, 1)
        avg_total_minmax = sum(r["stats"].get("total_minmax_entropy_loss", 0) for r in round_stats) / max(total_backwards, 1)
        avg_total_fisher = sum(r["stats"].get("total_fisher_loss", 0) for r in round_stats) / max(total_backwards, 1)
        avg_total_prox = sum(r["stats"].get("total_prox_loss", 0) for r in round_stats) / max(total_backwards, 1)
        avg_total_loss = sum(r["stats"].get("total_loss", 0) for r in round_stats) / max(total_backwards, 1)
        avg_confidence = sum(r["stats"]["mean_confidence"] * r["num_samples"] for r in round_stats) / total_samples
        avg_entropy_norm = sum(r["stats"]["mean_entropy_norm"] * r["num_samples"] for r in round_stats) / total_samples
        avg_collapse = sum(r["stats"]["collapse_score"] * r["num_samples"] for r in round_stats) / total_samples
        avg_reliability = sum(r["stats"]["reliability_raw"] * r["num_samples"] for r in round_stats) / total_samples
        avg_delta_norm = sum(r["stats"]["delta_norm"] * r["num_samples"] for r in round_stats) / total_samples
        total_fw = sum(r["stats"]["forwards"] for r in round_stats)
        total_bw = sum(r["stats"]["backwards"] for r in round_stats)

        # Communication cost = adapt_params * 2 directions * num_participants * 4 bytes/float
        adapt_param_count = count_adapt_params(self.source_adapt_state) if round_stats else 0
        num_participants = len(round_stats)
        comm_params_total = adapt_param_count * 2 * num_participants
        comm_bytes = comm_params_total * 4

        return {
            "avg_acc": avg_acc,
            "avg_ece": avg_ece,
            "avg_selected_ratio": avg_selected,
            "avg_entropy": avg_entropy,
            "avg_disagreement": avg_disagreement,
            "avg_consistency_loss": avg_consistency,
            "avg_total_consistency_loss": avg_total_consistency,
            "avg_total_minmax_loss": avg_total_minmax,
            "avg_total_fisher_loss": avg_total_fisher,
            "avg_total_prox_loss": avg_total_prox,
            "avg_total_loss": avg_total_loss,
            "avg_confidence": avg_confidence,
            "avg_entropy_norm": avg_entropy_norm,
            "avg_collapse": avg_collapse,
            "avg_reliability_raw": avg_reliability,
            "avg_delta_norm": avg_delta_norm,
            "num_forwards": total_fw,
            "num_backwards": total_bw,
            "communication_params": comm_params_total,
            "communication_bytes": comm_bytes,
            "domain_metrics": domain_metrics,
            "client_round_stats": client_round_stats,
            "num_clients": num_participants,
            "num_domains": len(domain_reports),
        }

    def save_adapters(self, path: str):
        """Save all domain adapters to a file."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({
            "domain_adapters": self.domain_adapters,
            "domain_momentums": self.domain_momentums,
            "domain_info": self.domain_info,
            "source_adapt_state": self.source_adapt_state,
        }, path)

    def load_adapters(self, path: str):
        """Load domain adapters from a file."""
        save_dict = torch.load(path, map_location="cpu")
        self.domain_adapters = save_dict["domain_adapters"]
        self.domain_momentums = save_dict["domain_momentums"]
        self.domain_info = save_dict["domain_info"]
        self.source_adapt_state = save_dict["source_adapt_state"]

    def get_domain_adapter(self, domain_id: int) -> Dict:
        return self.domain_adapters.get(domain_id, None)

    def get_domain_momentum(self, domain_id: int) -> Dict:
        return self.domain_momentums.get(domain_id, None)
