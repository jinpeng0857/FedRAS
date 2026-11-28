"""FedAdam server update for the FedRAS federated supplements."""

import copy

import torch

from train.fedavg.client import FedAvgClient
from train.common import AdaptationAnchorServer, weighted_delta


class FedAdamClient(FedAvgClient):
    """FedAdam uses the standard FedAvg local client objective."""


class FedAdamServer(AdaptationAnchorServer):
    method_name = "fedadam"
    source_reference = "Reddi et al., ICLR 2021; google-research/federated"

    def __init__(self, device, backbone, configs):
        super().__init__(device, backbone, configs)
        self.beta1 = float(configs.fedadam_beta1)
        self.beta2 = float(configs.fedadam_beta2)
        self.tau = float(configs.fedadam_tau)
        self.server_step = float(configs.fedopt_server_lr)
        self.first_moment = {}
        self.second_moment = {}

    def aggregate(self, local_weights_list, local_deltas_list, client_ids, client_weights=None):
        del client_ids
        weights = client_weights or [1.0] * len(local_deltas_list)
        average_model = super().aggregate(
            local_weights_list, local_deltas_list, [], client_weights=weights
        )
        global_state = self.backbone.state_dict()
        delta = weighted_delta(local_deltas_list, weights, global_state)
        result = copy.deepcopy(average_model)
        for name, value in delta.items():
            if name.endswith("running_mean") or name.endswith("running_var"):
                continue
            if name not in self.first_moment:
                self.first_moment[name] = torch.zeros_like(value)
                self.second_moment[name] = torch.zeros_like(value)
            self.first_moment[name].mul_(self.beta1).add_(value, alpha=1.0 - self.beta1)
            self.second_moment[name].mul_(self.beta2).addcmul_(
                value, value, value=1.0 - self.beta2
            )
            update = self.first_moment[name] / (
                self.second_moment[name].sqrt() + self.tau
            )
            result[name] = global_state[name] + self.server_step * update
        return result
