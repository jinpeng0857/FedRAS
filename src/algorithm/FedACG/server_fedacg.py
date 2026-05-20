"""
FedACG Server: supports multiple aggregation modes.

Modes:
  - fedavg:    Simple weighted average (FedAvg)
  - fedprox:   Same as fedavg (proximal term is client-side)
  - fedlc:     Same as fedavg (LC loss is client-side)
  - feddecorr: Same as fedavg
  - fedmlb:    Same as fedavg (MLB loss is client-side)
  - fedntd:    Same as fedavg
  - feddyn:    FedDyn dynamic regularization aggregation
  - fedacg:    FedACG with lookahead momentum (Nesterov-style)
  - fedadam:   Server-side Adam optimizer for aggregation
"""

import copy
import os
import time
from collections import defaultdict

import torch
from src.algorithm.Base.server_base import BaseServer
from utils.logger import logger


class FedACGServer(BaseServer):
    def __init__(self, device, backbone, configs):
        super().__init__(device, backbone, configs)
        self.mode = configs.fedacg_mode

        # Server-side momentum / optimizer state
        self.global_momentum = None
        self.global_v = None
        self.global_delta = None

    @staticmethod
    def _is_bn_running_stat_key(key):
        """Check if key is a BN running_mean or running_var."""
        return key.endswith("running_mean") or key.endswith("running_var")

    @staticmethod
    def _is_bn_counter_key(key):
        """Check if key is a BN num_batches_tracked counter."""
        return key.endswith("num_batches_tracked")

    @staticmethod
    def _get_client_num_samples(client):
        """Get the real number of training samples for a client.

        Priority: train_set (dataset) > train_dataloader.dataset > 0.
        """
        if hasattr(client, 'train_set') and client.train_set is not None:
            return len(client.train_set)
        if hasattr(client, 'train_dataloader') and client.train_dataloader is not None:
            return len(client.train_dataloader.dataset)
        return 0

    @staticmethod
    def _reset_bn_stats_in_state(state_dict):
        """Reset BN running stats to initial values in-place."""
        for k in state_dict:
            if 'running_mean' in k:
                state_dict[k].zero_()
            elif 'running_var' in k:
                state_dict[k].fill_(1.0)
            elif 'num_batches_tracked' in k:
                state_dict[k].zero_()

    def _init_momentum(self, model):
        """Initialize server momentum buffers (float params only)."""
        sd = copy.deepcopy(model.state_dict())
        for k in sd:
            if not self._is_float_key(sd, k):
                continue
            sd[k] = torch.zeros_like(sd[k])
        self.global_momentum = copy.deepcopy(sd)

        if self.mode == "fedadam":
            self.global_v = copy.deepcopy(sd)
            for k in self.global_v:
                self.global_v[k] = (
                    self.global_v[k] + self.configs.server_momentum ** 2
                )

        if self.mode in ("fedacg", "fedavgm"):
            self.global_delta = copy.deepcopy(sd)

    def _fedacg_lookahead(self, model):
        """FedACG: Send model with Nesterov momentum to clients."""
        sending_dict = copy.deepcopy(model.state_dict())
        for k in self.global_momentum:
            if not self._is_float_key(sending_dict, k):
                continue
            sending_dict[k] = (
                sending_dict[k]
                + self.configs.server_momentum * self.global_momentum[k]
            )
        model.load_state_dict(sending_dict)

    def make_checkpoint(self, round_num):
        """Save checkpoint atomically to prevent pickle truncation."""
        checkpoint = {
            "epoch": round_num,
            "model_state_dict": self.backbone.state_dict(),
            "config_info": {
                "num_class": getattr(self.configs, "num_class", None),
                "backbone": getattr(self.configs, "backbone", None),
                "dataset": getattr(self.configs, "dataset", None),
                "alpha": getattr(self.configs, "alpha", None),
                "fedacg_mode": getattr(self.configs, "fedacg_mode", None),
                "subnet_drop_rate": getattr(self.configs, "subnet_drop_rate", None),
            },
        }
        if self.global_momentum is not None:
            checkpoint["global_momentum"] = self.global_momentum
            la = copy.deepcopy(self.backbone.state_dict())
            for k in la:
                if la[k].dtype not in (torch.float16, torch.float32, torch.float64):
                    continue
                la[k] = la[k] + self.configs.server_momentum * self.global_momentum[k].to(la[k].device)
            checkpoint["lookahead_anchor"] = la

        os.makedirs(self.checkpoint_path, exist_ok=True)
        path = os.path.join(self.checkpoint_path, f"model_round{round_num}.pt")
        tmp_path = path + ".tmp"
        torch.save(checkpoint, tmp_path)
        os.replace(tmp_path, path)  # atomic replace
        logger.info(f"  checkpoint saved: {path}")

        # Auto-clean: keep only last 3 checkpoints
        existing = sorted(
            [f for f in os.listdir(self.checkpoint_path)
             if f.startswith("model_round") and f.endswith(".pt")],
            key=lambda x: int(''.join(c for c in x if c.isdigit()))
        )
        while len(existing) > 3:
            old = os.path.join(self.checkpoint_path, existing.pop(0))
            os.remove(old)
            logger.info(f"  cleaned old checkpoint: {old}")

    def aggregate(self, local_weights_list, local_deltas_list, client_ids, client_weights=None):
        """
        Aggregate client updates based on mode.

        Args:
            local_weights_list: list of client state_dicts
            local_deltas_list: list of state_dicts with (client - global) diffs
            client_ids: list of client IDs
            client_weights: list of data sizes for each client (for weighted avg)

        Returns:
            aggregated state_dict
        """
        C = len(client_ids)

        # Compute normalized weights
        if client_weights is not None and len(client_weights) > 0:
            total_w = sum(client_weights)
            norm_weights = [w / total_w for w in client_weights]
        else:
            norm_weights = [1.0 / C] * C

        # Weighted average: float params (incl. BN running_mean/var)
        # by sample count, BN num_batches_tracked by max, non-float keep first client's value
        avg_weights = copy.deepcopy(local_weights_list[0])

        for k in avg_weights:
            if self._is_bn_counter_key(k):
                # num_batches_tracked: take max across clients (not float average)
                vals = [w[k].item() for w in local_weights_list]
                max_idx = max(range(len(vals)), key=lambda i: vals[i])
                avg_weights[k] = local_weights_list[max_idx][k].clone()
            elif self._is_float_key(avg_weights, k):
                # Float params (including BN running_mean/var): weighted average by sample count
                avg_weights[k] = torch.zeros_like(avg_weights[k])
                for i, w in enumerate(local_weights_list):
                    avg_weights[k] += (w[k] * norm_weights[i]).to(avg_weights[k].dtype)
            # else: non-float, non-BN-counter → keep first client's value (already set by deepcopy)

        if self.mode in ("fedavg", "fedprox", "fedlc", "feddecorr",
                         "fedmlb", "fedntd"):
            return avg_weights

        if self.global_momentum is None:
            self._init_momentum(self.backbone)

        if self.mode == "fedavgm":
            # FedAvg with server momentum
            for k in self.global_delta:
                if not self._is_float_key(self.global_delta, k):
                    continue
                self.global_delta[k] = torch.zeros_like(self.global_delta[k])
                for i, d in enumerate(local_deltas_list):
                    self.global_delta[k] += (d[k] * norm_weights[i]).to(self.global_delta[k].dtype)
                self.global_momentum[k] = (
                    self.configs.server_momentum * self.global_momentum[k]
                    + self.global_delta[k]
                )
            for k in avg_weights:
                if not self._is_float_key(avg_weights, k):
                    continue
                avg_weights[k] = (
                    local_weights_list[0][k] + self.global_momentum[k]
                )
                for i in range(1, C):
                    avg_weights[k] += local_weights_list[i][k] - local_weights_list[0][k]
                avg_weights[k] /= C
            return avg_weights

        elif self.mode == "fedacg":
            # FedACG aggregation
            for k in self.global_delta:
                if not self._is_float_key(self.global_delta, k):
                    continue
                self.global_delta[k] = torch.zeros_like(self.global_delta[k])
                for i, d in enumerate(local_deltas_list):
                    self.global_delta[k] += (d[k] * norm_weights[i]).to(self.global_delta[k].dtype)
                self.global_momentum[k] = (
                    self.configs.server_momentum * self.global_momentum[k]
                    + self.global_delta[k]
                )
            # Weighted averaging (lookahead already applied before sending)
            return avg_weights

        elif self.mode == "fedadam":
            # Server-side Adam
            server_lr = self.configs.server_lr
            for k in self.global_delta:
                if not self._is_float_key(self.global_delta, k):
                    continue
                self.global_delta[k] = torch.zeros_like(self.global_delta[k])
                for i, d in enumerate(local_deltas_list):
                    self.global_delta[k] += (d[k] * norm_weights[i]).to(self.global_delta[k].dtype)
                self.global_momentum[k] = (
                    self.configs.server_momentum * self.global_momentum[k]
                    + (1 - self.configs.server_momentum) * self.global_delta[k]
                )
                self.global_v[k] = (
                    0.999 * self.global_v[k]
                    + 0.001 * (self.global_delta[k] ** 2)
                )
                avg_weights[k] = (
                    avg_weights[k]
                    + server_lr * self.global_momentum[k]
                    / (self.global_v[k].sqrt() + 1e-3)
                )
            return avg_weights

        elif self.mode == "feddyn":
            # FedDyn aggregation
            for k in self.global_momentum:
                if not self._is_float_key(self.global_momentum, k):
                    continue
                self.global_momentum[k] -= (
                    self.configs.dyn_alpha
                    / self.configs.num_client
                    * sum(d[k] for d in local_deltas_list)
                )
                avg_weights[k] = (
                    avg_weights[k]
                    - 1 / self.configs.dyn_alpha * self.global_momentum[k]
                )
            return avg_weights

        return avg_weights

    def fit(self):
        """Main federated training loop. Saves checkpoint every save_freq rounds."""
        if self.mode in ("fedacg", "fedavgm", "fedadam", "feddyn"):
            self._init_momentum(self.backbone)

        for r in range(self.global_rounds):
            active_clients = self.select_clients()
            logger.info(f"Round {r}/{self.global_rounds}, "
                        f"clients={len(active_clients)}")

            # Global state -> clients
            global_state_dict = copy.deepcopy(self.backbone.state_dict())

            # FedACG lookahead momentum
            if self.mode == "fedacg":
                self._fedacg_lookahead(self.backbone)
                global_state_dict = copy.deepcopy(self.backbone.state_dict())

            for client in self.clients:
                client.update_global_model(global_state_dict)

            # CRITICAL: Set selected clients' backbone to lookahead state
            # Ensures local training starts from theta^{t-1}+lambda*m^{t-1},
            # not from stale theta^{t-1}
            for client in active_clients:
                client.backbone.load_state_dict(global_state_dict)

            # Client-side training
            local_weights_list = []
            local_deltas_list = []
            client_accs = []
            client_times = []
            client_weights = []

            for client in active_clients:
                data_len = self._get_client_num_samples(client)
                if data_len == 0:
                    logger.warning(f"  client{client.cid} has no data, skipped")
                    continue
                client_weights.append(data_len)
                start = time.time()
                report = client.train()
                elapsed = time.time() - start
                client_times.append(elapsed)
                client_accs.append(report["accuracy"])
                local_weights_list.append(report["backbone"])
                local_deltas_list.append({
                    k: report["backbone"][k] - global_state_dict[k]
                    for k in report["backbone"]
                })
                logger.info(
                    f"  client{client.cid} time={elapsed:.1f}s "
                    f"acc={report['accuracy']:.2%}"
                )

            # Server aggregation
            if len(local_weights_list) == 0:
                logger.warning(f"  Round {r}: no clients with data, skipping aggregation")
                continue

            global_state = self.aggregate(
                local_weights_list, local_deltas_list,
                [c.cid for c in active_clients],
                client_weights=client_weights,
            )
            self.backbone.load_state_dict(global_state)
            for client in self.clients:
                client.backbone.load_state_dict(global_state)
                client.update_global_model(global_state)

            # Logging
            avg_time = sum(client_times) / len(client_times) if client_times else 0
            avg_acc = sum(
                client_accs[i] * client_weights[i] / sum(client_weights)
                for i in range(len(client_accs))
            ) if client_accs else 0.0
            logger.info(f"  avg_time={avg_time:.1f}s avg_acc={avg_acc:.2%}")
            logger.info(f"  selected client sample counts: {client_weights}")
            logger.info(f"  total selected samples: {sum(client_weights)}")

            # Save checkpoint by save_freq: always save first and last round
            if r % self.save_freq == 0 or r == self.global_rounds - 1:
                self.make_checkpoint(r)
