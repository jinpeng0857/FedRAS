"""Pure FedACG server.

This server keeps only the FedACG training path:
  1. send a lookahead global state to selected clients;
  2. collect client state_dicts and client deltas;
  3. weighted-average client models by local sample count;
  4. update server momentum from the weighted local delta;
  5. save standard checkpoints compatible with the main adaptation test.
"""

import copy
import math
from dataset.tiny_imagenet import load_tiny_imagenet_validation, tiny_imagenet_transform
import os
import time

import torch
from torch.utils.data import DataLoader
import torchvision

from train.fedacg.base_server import BaseServer
from dataset.dataset import DATASET_PATHS, get_transform
from utils.logger import logger


class FedACGServer(BaseServer):
    def __init__(self, device, backbone, configs):
        super().__init__(device, backbone, configs)
        self.mode = configs.fedacg_mode
        if self.mode != "fedacg":
            raise ValueError("This cleaned project only supports --fedacg_mode fedacg")

        self.global_momentum = None
        self.global_delta = None

        self.ema_decay = float(getattr(configs, "ema_decay", 0.0))
        self.ema_state_dict = None
        self.best_clean_acc = 0.0
        self.best_clean_round = None
        self.clean_test_loader = None

    @staticmethod
    def _is_bn_counter_key(key):
        return key.endswith("num_batches_tracked")

    @staticmethod
    def _is_bn_buffer_key(key):
        return (
            key.endswith("running_mean")
            or key.endswith("running_var")
            or key.endswith("num_batches_tracked")
        )

    @staticmethod
    def _get_client_num_samples(client):
        if hasattr(client, "train_set") and client.train_set is not None:
            return len(client.train_set)
        if hasattr(client, "train_dataloader") and client.train_dataloader is not None:
            return len(client.train_dataloader.dataset)
        return 0

    def _init_momentum(self, model):
        """Initialize momentum/delta buffers for float model parameters only."""
        sd = copy.deepcopy(model.state_dict())
        for k in sd:
            if self._is_float_key(sd, k) and not self._is_bn_buffer_key(k):
                sd[k] = torch.zeros_like(sd[k])
        self.global_momentum = copy.deepcopy(sd)
        self.global_delta = copy.deepcopy(sd)

    def _get_lookahead_state(self):
        """Return the state sent to clients without mutating the server model."""
        if self.global_momentum is None:
            self._init_momentum(self.backbone)

        sending_dict = copy.deepcopy(self.backbone.state_dict())
        for k in self.global_momentum:
            if not self._is_float_key(sending_dict, k):
                continue
            if self._is_bn_buffer_key(k):
                continue
            sending_dict[k] = (
                sending_dict[k]
                + self.configs.server_momentum * self.global_momentum[k].to(sending_dict[k].device)
            )
        return sending_dict

    def _update_ema(self, current_state_dict):
        if self.ema_decay <= 0:
            return

        if self.ema_state_dict is None:
            self.ema_state_dict = copy.deepcopy(current_state_dict)
            for k in self.ema_state_dict:
                if self._is_float_key(self.ema_state_dict, k):
                    self.ema_state_dict[k] = self.ema_state_dict[k].detach().cpu()
            return

        with torch.no_grad():
            for k in self.ema_state_dict:
                if k not in current_state_dict:
                    continue
                if not self._is_float_key(self.ema_state_dict, k):
                    continue
                if self._is_bn_buffer_key(k):
                    continue
                self.ema_state_dict[k] = (
                    self.ema_decay * self.ema_state_dict[k]
                    + (1.0 - self.ema_decay) * current_state_dict[k].detach().cpu()
                )

    def _build_lookahead_anchor(self):
        lookahead_anchor = copy.deepcopy(self.backbone.state_dict())
        if self.global_momentum is None:
            return lookahead_anchor
        for k in lookahead_anchor:
            if not self._is_float_key(lookahead_anchor, k):
                continue
            if self._is_bn_buffer_key(k):
                continue
            lookahead_anchor[k] = (
                lookahead_anchor[k]
                + self.configs.server_momentum * self.global_momentum[k].to(lookahead_anchor[k].device)
            )
        return lookahead_anchor

    def _build_checkpoint(self, round_num, final=False):
        model_sd = copy.deepcopy(self.backbone.state_dict())

        if final and self.ema_state_dict is not None and self.ema_decay > 0:
            ema_sd = copy.deepcopy(self.ema_state_dict)
            for k in ema_sd:
                if k in model_sd and self._is_float_key(ema_sd, k):
                    if self._is_bn_buffer_key(k):
                        continue
                    model_sd[k] = ema_sd[k].to(model_sd[k].dtype)

        checkpoint = {
            "round": round_num,
            "epoch": round_num,
            "model_state_dict": model_sd,
            "ema_state_dict": self.ema_state_dict,
            "lookahead_anchor": self._build_lookahead_anchor(),
            "best_clean_acc": self.best_clean_acc,
            "best_clean_round": self.best_clean_round,
            "config_info": {
                "num_class": getattr(self.configs, "num_class", None),
                "backbone": getattr(self.configs, "backbone", None),
                "dataset": getattr(self.configs, "dataset", None),
                "alpha": getattr(self.configs, "alpha", None),
                "fedacg_mode": getattr(self.configs, "fedacg_mode", None),
            },
        }

        if self.ema_state_dict is not None and self.ema_decay > 0:
            checkpoint["ema_decay"] = self.ema_decay

        if self.global_momentum is not None:
            checkpoint["global_momentum"] = self.global_momentum
        return checkpoint

    def make_checkpoint(self, round_num, final=False):
        """Save a checkpoint with raw, EMA, momentum, and lookahead states."""
        checkpoint = self._build_checkpoint(round_num, final=final)

        os.makedirs(self.checkpoint_path, exist_ok=True)
        path = os.path.join(self.checkpoint_path, f"model_round{round_num}.pt")
        tmp_path = path + ".tmp"
        torch.save(checkpoint, tmp_path)
        os.replace(tmp_path, path)
        logger.info(f"  checkpoint saved: {path}")
        logger.info(f"  checkpoint keys: {sorted(checkpoint.keys())}")

        existing = sorted(
            [f for f in os.listdir(self.checkpoint_path)
             if f.startswith("model_round") and f.endswith(".pt")],
            key=lambda x: int("".join(c for c in x if c.isdigit())),
        )
        while len(existing) > 3:
            old = os.path.join(self.checkpoint_path, existing.pop(0))
            os.remove(old)
            logger.info(f"  cleaned old checkpoint: {old}")

    def _save_best_clean_checkpoint(self, round_num):
        checkpoint = self._build_checkpoint(round_num, final=False)
        os.makedirs(self.checkpoint_path, exist_ok=True)
        path = os.path.join(self.checkpoint_path, "best_clean.pt")
        tmp_path = path + ".tmp"
        torch.save(checkpoint, tmp_path)
        os.replace(tmp_path, path)
        logger.info(f"  best clean checkpoint saved: {path}")
        logger.info(f"  checkpoint keys: {sorted(checkpoint.keys())}")

    def _get_round_lr(self, round_num):
        base_lr = self.configs.learning_rate
        scheduler = getattr(self.configs, "lr_scheduler", "none")
        total_rounds = max(int(self.global_rounds), 1)

        if scheduler == "none":
            return base_lr
        if scheduler == "cosine":
            progress = min(round_num, total_rounds - 1) / max(total_rounds - 1, 1)
            return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))
        if scheduler == "multistep":
            if round_num >= int(0.75 * total_rounds):
                return base_lr * 0.01
            if round_num >= int(0.5 * total_rounds):
                return base_lr * 0.1
            return base_lr
        raise ValueError(f"Unknown lr_scheduler: {scheduler}")

    def _get_clean_test_loader(self):
        if self.clean_test_loader is not None:
            return self.clean_test_loader

        dataset_name = getattr(self.configs, "dataset", None)
        if dataset_name not in ("cifar10", "cifar100", "tiny-imagenet"):
            logger.warning("  clean eval skipped: dataset=%s has no clean loader", dataset_name)
            return None

        data_path = self.configs.dataset_path or DATASET_PATHS[dataset_name]
        if dataset_name == "tiny-imagenet":
            clean_dataset = load_tiny_imagenet_validation(
                data_path, transform=tiny_imagenet_transform(train=False)
            )
        else:
            dataset_cls = (
                torchvision.datasets.CIFAR10
                if dataset_name == "cifar10"
                else torchvision.datasets.CIFAR100
            )
            clean_dataset = dataset_cls(
                root=data_path,
                train=False,
                transform=get_transform(dataset_name, train=False),
                download=True,
            )
        self.clean_test_loader = DataLoader(
            clean_dataset,
            batch_size=getattr(self.configs, "test_batch_size", self.configs.batch_size),
            shuffle=False,
            num_workers=self.configs.num_workers,
            pin_memory=self.device.type == "cuda",
        )
        return self.clean_test_loader

    def evaluate_clean(self, state_dict):
        loader = self._get_clean_test_loader()
        if loader is None:
            return None

        model = copy.deepcopy(self.backbone)
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()

        correct = 0
        total = 0
        with torch.inference_mode():
            for data, target in loader:
                data = data.to(self.device)
                target = target.to(self.device)
                logits = model(data)
                if isinstance(logits, dict):
                    logits = logits["logit"]
                pred = logits.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)

        model.cpu()
        return correct / total if total > 0 else 0.0

    def aggregate(self, local_weights_list, local_deltas_list, client_ids, client_weights=None):
        """Weighted model averaging plus FedACG momentum update."""
        num_clients = len(client_ids)
        if num_clients == 0:
            raise ValueError("No client updates were provided for aggregation")

        if client_weights is not None and len(client_weights) > 0:
            total_w = float(sum(client_weights))
            norm_weights = [w / total_w for w in client_weights]
        else:
            norm_weights = [1.0 / num_clients] * num_clients

        first_sd = local_weights_list[0]
        avg_weights = {}

        for k in first_sd:
            if self._is_bn_counter_key(k):
                vals = [w[k].item() for w in local_weights_list]
                max_idx = max(range(len(vals)), key=lambda i: vals[i])
                avg_weights[k] = local_weights_list[max_idx][k].clone()
            elif self._is_float_key(first_sd, k):
                avg_weights[k] = torch.zeros_like(first_sd[k])
                for i, w in enumerate(local_weights_list):
                    avg_weights[k] += (w[k] * norm_weights[i]).to(avg_weights[k].dtype)
            else:
                avg_weights[k] = first_sd[k].clone()

        if self.global_momentum is None or self.global_delta is None:
            self._init_momentum(self.backbone)

        for k in self.global_delta:
            if not self._is_float_key(self.global_delta, k):
                continue
            if self._is_bn_buffer_key(k):
                continue
            self.global_delta[k] = torch.zeros_like(self.global_delta[k])
            for i, delta in enumerate(local_deltas_list):
                self.global_delta[k] += (delta[k] * norm_weights[i]).to(self.global_delta[k].dtype)
            self.global_momentum[k] = (
                self.configs.server_momentum * self.global_momentum[k]
                + self.global_delta[k]
            )

        return avg_weights

    def fit(self):
        """Main federated training loop for pure FedACG."""
        self._init_momentum(self.backbone)
        clients_on_gpu = set()

        for r in range(self.global_rounds):
            active_clients = self.select_clients()
            selected_ids = {c.cid for c in active_clients}
            logger.info(f"Round {r}/{self.global_rounds}, clients={len(active_clients)}")

            global_state_dict = self._get_lookahead_state()
            current_lr = self._get_round_lr(r)
            logger.info(f"  current round={r}, local_lr={current_lr:.6g}, lr_scheduler={getattr(self.configs, 'lr_scheduler', 'none')}")

            if self.device.type != "cpu":
                for cid in clients_on_gpu - selected_ids:
                    self.clients[cid].backbone = self.clients[cid].backbone.cpu()
                clients_on_gpu.intersection_update(selected_ids)

                for client in active_clients:
                    if client.cid not in clients_on_gpu:
                        client.backbone = client.backbone.to(self.device)
                        clients_on_gpu.add(client.cid)

            for client in active_clients:
                client.setup(
                    state_dict=global_state_dict,
                    device=self.device,
                    local_lr=current_lr,
                    global_epoch=r,
                )

            local_weights_list = []
            local_deltas_list = []
            client_accs = []
            client_losses = []
            client_times = []
            client_weights = []
            active_ids_with_data = []

            for client in active_clients:
                data_len = self._get_client_num_samples(client)
                if data_len == 0:
                    logger.warning(f"  client{client.cid} has no data, skipped")
                    continue

                client_weights.append(data_len)
                active_ids_with_data.append(client.cid)

                start = time.time()
                report = client.train()
                elapsed = time.time() - start

                client_times.append(elapsed)
                client_accs.append(report["accuracy"])
                client_losses.append(report.get("loss", 0.0))
                local_weights_list.append(report["backbone"])
                local_deltas_list.append({
                    k: report["backbone"][k] - global_state_dict[k]
                    for k in report["backbone"]
                })

                # Reclaim GPU memory immediately after each client finishes
                if self.device.type != "cpu":
                    client.backbone = client.backbone.cpu()
                    clients_on_gpu.discard(client.cid)

                logger.info(
                    f"  client{client.cid} time={elapsed:.1f}s "
                    f"acc={report['accuracy']:.2%} loss={report.get('loss', 0.0):.4f}"
                )

            if len(local_weights_list) == 0:
                logger.warning(f"  Round {r}: no clients with data, skipping aggregation")
                if self.device.type != "cpu":
                    for cid in selected_ids & clients_on_gpu:
                        self.clients[cid].backbone = self.clients[cid].backbone.cpu()
                    clients_on_gpu.difference_update(selected_ids)
                continue

            global_state = self.aggregate(
                local_weights_list,
                local_deltas_list,
                active_ids_with_data,
                client_weights=client_weights,
            )
            self.backbone.load_state_dict(global_state)

            for client in active_clients:
                client.backbone.load_state_dict(global_state)
                client.update_global_model(global_state)

            if self.ema_decay > 0:
                self._update_ema(global_state)

            total_samples = sum(client_weights)
            avg_time = sum(client_times) / len(client_times) if client_times else 0.0
            avg_acc = (
                sum(client_accs[i] * client_weights[i] / total_samples for i in range(len(client_accs)))
                if client_accs else 0.0
            )
            avg_loss = (
                sum(client_losses[i] * client_weights[i] / total_samples for i in range(len(client_losses)))
                if client_losses else 0.0
            )

            logger.info(f"  avg_time={avg_time:.1f}s avg_acc={avg_acc:.2%} avg_loss={avg_loss:.4f}")
            logger.info(f"  train selected clients avg acc/loss: {avg_acc:.2%}/{avg_loss:.4f}")
            logger.info(f"  selected client sample counts: {client_weights}")
            logger.info(f"  total selected samples: {total_samples}")

            is_final = (r == self.global_rounds - 1)
            should_eval_clean = (
                getattr(self.configs, "eval_clean_every", 0) > 0
                and ((r + 1) % self.configs.eval_clean_every == 0 or is_final)
            )
            if should_eval_clean:
                global_clean_acc = self.evaluate_clean(self.backbone.state_dict())
                clean_acc = self.evaluate_clean(self._build_lookahead_anchor())
                if clean_acc is not None:
                    if clean_acc > self.best_clean_acc:
                        self.best_clean_acc = clean_acc
                        self.best_clean_round = r
                        if getattr(self.configs, "save_best_clean", False):
                            self._save_best_clean_checkpoint(r)
                    logger.info(
                        f"  global_clean_acc={global_clean_acc:.2%} "
                        f"clean_acc={clean_acc:.2%} "
                        f"best_clean_acc={self.best_clean_acc:.2%} "
                        f"best_round={self.best_clean_round}"
                    )

            if r % self.save_freq == 0 or is_final:
                self.make_checkpoint(r, final=is_final)
