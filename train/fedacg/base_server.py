"""Base Server class for federated learning."""

import copy
import os
import random

import torch


class BaseServer:
    @staticmethod
    def _is_float_key(d, k):
        return d[k].dtype in (torch.float16, torch.float32, torch.float64)

    def __init__(self, device, backbone, configs):
        self.clients = []
        self.join_ratio = configs.join_ratio
        self.global_rounds = configs.global_rounds
        self.checkpoint_path = configs.checkpoint_path
        self.save_freq = configs.save_freq
        self.eval_freq = configs.eval_freq
        self.backbone = copy.deepcopy(backbone)
        self.device = device
        self.debug = configs.debug
        self.configs = configs

    def select_clients(self):
        if self.join_ratio >= 1.0:
            return self.clients
        else:
            return random.sample(
                self.clients,
                int(round(len(self.clients) * self.join_ratio)),
            )

    @staticmethod
    def model_average(client_net_states, client_weights):
        state_avg = copy.deepcopy(client_net_states[0])
        total_w = sum(client_weights)
        for k in state_avg.keys():
            if state_avg[k].dtype not in (torch.float16, torch.float32, torch.float64):
                continue
            state_avg[k] = torch.zeros_like(state_avg[k])
            for i, w in enumerate(client_weights):
                state_avg[k] += client_net_states[i][k] * (w / total_w)
        return state_avg

    def make_checkpoint(self, round_num):
        """Save in ACG-C unified format: {'epoch': N, 'model_state_dict': dict}"""
        os.makedirs(self.checkpoint_path, exist_ok=True)
        path = os.path.join(self.checkpoint_path, f"model_round{round_num}.pt")
        torch.save({
            "epoch": round_num,
            "model_state_dict": self.backbone.state_dict(),
        }, path)

    def load_checkpoint(self, checkpoint_path):
        """Load from ACG-C format (compatible with raw state_dict too)."""
        from utils.helpers import load_checkpoint_weights
        load_checkpoint_weights(self.backbone, checkpoint_path, self.device)
        for client in self.clients:
            client.backbone.load_state_dict(self.backbone.state_dict())
