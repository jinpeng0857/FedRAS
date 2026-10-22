"""FedAvg server with sample-count-weighted model averaging."""

import copy

import torch

from train.fedacg.base_server import BaseServer
from train.fedacg.server import FedACGServer


class FedAvgServer(FedACGServer):
    """Use the shared training orchestration without FedACG momentum or Lookahead."""

    def __init__(self, device, backbone, configs):
        BaseServer.__init__(self, device, backbone, configs)
        self.global_momentum = None
        self.global_delta = None
        self.ema_decay = 0.0
        self.ema_state_dict = None
        self.best_clean_acc = 0.0
        self.best_clean_round = None
        self.clean_test_loader = None

    def _init_momentum(self, model):
        """FedAvg has no server-momentum state."""

    def _get_lookahead_state(self):
        """Return the current global model sent to every selected client."""
        return copy.deepcopy(self.backbone.state_dict())

    def _build_lookahead_anchor(self):
        """Return the FedAvg global model used as the deployment anchor."""
        return copy.deepcopy(self.backbone.state_dict())

    def aggregate(
        self,
        local_weights_list,
        local_deltas_list,
        client_ids,
        client_weights=None,
    ):
        """Compute the standard sample-count-weighted FedAvg model."""
        del local_deltas_list, client_ids
        if not local_weights_list:
            raise ValueError("No client updates were provided for aggregation")

        if client_weights:
            total_weight = float(sum(client_weights))
            normalized_weights = [
                float(weight) / total_weight for weight in client_weights
            ]
        else:
            normalized_weights = [1.0 / len(local_weights_list)] * len(
                local_weights_list
            )

        first_state = local_weights_list[0]
        averaged = {}
        for name, value in first_state.items():
            if self._is_bn_counter_key(name):
                counter_values = [state[name].item() for state in local_weights_list]
                averaged[name] = local_weights_list[
                    max(range(len(counter_values)), key=counter_values.__getitem__)
                ][name].clone()
            elif self._is_float_key(first_state, name):
                averaged[name] = torch.zeros_like(value)
                for state, weight in zip(local_weights_list, normalized_weights):
                    averaged[name] += state[name].to(averaged[name].dtype) * weight
            else:
                averaged[name] = value.clone()
        return averaged

    def _build_checkpoint(self, round_num, final=False):
        """Save the global model under an explicit non-Lookahead anchor key."""
        del final
        model_state = copy.deepcopy(self.backbone.state_dict())
        return {
            "round": round_num,
            "epoch": round_num,
            "method": "fedavg",
            "model_state_dict": copy.deepcopy(model_state),
            "adaptation_anchor": model_state,
            "anchor_type": "fedavg_global_model",
            "best_clean_acc": self.best_clean_acc,
            "best_clean_round": self.best_clean_round,
            "config_info": {
                "method": "fedavg",
                "num_class": getattr(self.configs, "num_class", None),
                "backbone": getattr(self.configs, "backbone", None),
                "dataset": getattr(self.configs, "dataset", None),
                "alpha": getattr(self.configs, "alpha", None),
            },
        }
