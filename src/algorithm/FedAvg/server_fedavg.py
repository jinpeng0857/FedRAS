"""FedAvg Server: standard weighted averaging."""

import copy
import os
import time

import torch
from src.algorithm.Base.server_base import BaseServer
from utils.logger import logger


class FedAvgServer(BaseServer):
    def __init__(self, device, backbone, configs):
        super().__init__(device, backbone, configs)

    def fit(self):
        # Cross-round weighted average accumulator
        round_avg_sum = None
        round_avg_weight = 0.0

        for r in range(self.global_rounds):
            client_weights, client_net_states, client_accuracy, client_train_time = [], [], [], []
            active_clients = self.select_clients()
            logger.info(f"Round {r}")
            for client in active_clients:
                client_weights.append(len(client.train_dataloader.dataset))
                start_time = time.time()
                report = client.train()
                end_time = time.time()
                client_net_states.append(report["backbone"])
                client_accuracy.append(report["accuracy"])
                client_train_time.append(end_time - start_time)
                logger.info(
                    f"  client{client.cid} time={end_time - start_time:.1f}s "
                    f"acc={report['accuracy']:.2%}"
                )

            global_net_state = self.model_average(client_net_states, client_weights)
            for client in self.clients:
                client.backbone.load_state_dict(global_net_state)
            self.backbone.load_state_dict(global_net_state)

            avg_time = sum(client_train_time) / len(client_train_time)
            avg_acc = sum(
                client_accuracy[i] * client_weights[i] / sum(client_weights)
                for i in range(len(active_clients))
            )
            logger.info(f"  avg_time={avg_time:.1f}s avg_acc={avg_acc:.2%}")

            # Accumulate into cross-round weighted average
            round_data_weight = sum(client_weights) if client_weights else 0
            if round_data_weight > 0:
                if round_avg_sum is None:
                    round_avg_sum = copy.deepcopy(global_net_state)
                    for k in round_avg_sum:
                        if self._is_float_key(round_avg_sum, k):
                            round_avg_sum[k] = round_avg_sum[k] * round_data_weight
                else:
                    for k in round_avg_sum:
                        if self._is_float_key(round_avg_sum, k):
                            round_avg_sum[k] += global_net_state[k] * round_data_weight
                round_avg_weight += round_data_weight

        # Save the cross-round weighted average model
        if round_avg_sum is not None and round_avg_weight > 0:
            final_state = copy.deepcopy(round_avg_sum)
            for k in final_state:
                if self._is_float_key(final_state, k):
                    final_state[k] = round_avg_sum[k] / round_avg_weight

            saved_model = self.backbone.state_dict()
            self.backbone.load_state_dict(final_state)
            self.make_checkpoint(self.global_rounds)
            self.backbone.load_state_dict(saved_model)
