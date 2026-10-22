"""Shared server utilities for the FedDR federated supplements."""

import copy

import torch

from train.fedavg.server import FedAvgServer


def normalized_client_weights(counts):
    if not counts:
        raise ValueError("at least one client weight is required")
    total = float(sum(counts))
    if total <= 0:
        raise ValueError("client weights must sum to a positive value")
    return [float(value) / total for value in counts]


def weighted_delta(local_deltas, weights, reference):
    """Average floating-point deltas; exclude BN counters and integer buffers."""
    coefficients = normalized_client_weights(weights)
    result = {}
    for name, value in reference.items():
        if not torch.is_floating_point(value):
            continue
        result[name] = torch.zeros_like(value)
        for delta, coefficient in zip(local_deltas, coefficients):
            result[name].add_(delta[name].to(result[name].dtype), alpha=coefficient)
    return result


class AdaptationAnchorServer(FedAvgServer):
    """FedAvg orchestration with a method-labelled deployment checkpoint."""

    method_name = "baseline"
    source_reference = ""

    def _build_checkpoint(self, round_num, final=False):
        del final
        model_state = copy.deepcopy(self.backbone.state_dict())
        return {
            "round": round_num,
            "epoch": round_num,
            "method": self.method_name,
            "model_state_dict": copy.deepcopy(model_state),
            "adaptation_anchor": model_state,
            "anchor_type": f"{self.method_name}_global_model",
            "best_clean_acc": self.best_clean_acc,
            "best_clean_round": self.best_clean_round,
            "source_reference": self.source_reference,
            "config_info": {
                "method": self.method_name,
                "num_class": getattr(self.configs, "num_class", None),
                "backbone": getattr(self.configs, "backbone", None),
                "dataset": getattr(self.configs, "dataset", None),
                "alpha": getattr(self.configs, "alpha", None),
            },
        }
