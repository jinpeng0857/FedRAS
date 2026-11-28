"""FedDyn dynamic regularization for the FedRAS supplements."""

import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast

from train.fedacg.client import FedACGClient
from train.common import AdaptationAnchorServer


class FedDynClient(FedACGClient):
    def __init__(self, cid, device, backbone, configs):
        super().__init__(cid, device, backbone, configs)
        self.alpha = float(configs.feddyn_alpha)
        self.alpha_scale = 1.0
        self.history = {
            name: torch.zeros_like(value, device="cpu")
            for name, value in backbone.named_parameters()
        }

    def setup(self, state_dict, device, local_lr=None, global_epoch=None):
        super().setup(state_dict, device, local_lr=local_lr, global_epoch=global_epoch)
        lr = local_lr if local_lr is not None else self.learning_rate
        self.optimizer = torch.optim.SGD(
            self.backbone.parameters(),
            lr=lr,
            momentum=self.momentum,
            weight_decay=self.weight_decay + self.alpha * self.alpha_scale,
        )

    def train(self):
        self.backbone.train()
        amp_device = "cuda" if self.device.type == "cuda" and torch.cuda.is_available() else "cpu"
        scaler = GradScaler(amp_device, enabled=getattr(self.configs, "use_amp", False))
        global_parameters = {
            name: self._global_state[name].to(self.device)
            for name, _ in self.backbone.named_parameters()
        }
        history = {name: value.to(self.device) for name, value in self.history.items()}
        correct = total = steps = 0
        running_loss = 0.0
        for _ in range(self.local_epochs):
            for data, target in self.train_dataloader:
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad(set_to_none=True)
                with autocast(amp_device, enabled=getattr(self.configs, "use_amp", False)):
                    logits = self.backbone(data)
                    loss = F.cross_entropy(logits, target, label_smoothing=self.label_smoothing)
                    linear_penalty = sum(
                        (parameter * (-global_parameters[name] + history[name])).sum()
                        for name, parameter in self.backbone.named_parameters()
                    )
                    loss = loss + self.alpha * self.alpha_scale * linear_penalty
                scaler.scale(loss).backward()
                scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.backbone.parameters(), 10.0)
                scaler.step(self.optimizer)
                scaler.update()
                steps += 1
                correct += logits.detach().argmax(1).eq(target).sum().item()
                total += target.numel()
                running_loss += float(loss.detach().cpu())
        if steps == 0:
            raise RuntimeError(f"client {self.cid} completed no local steps")
        local_state = self.backbone.state_dict()
        for name in self.history:
            self.history[name].add_(local_state[name].detach().cpu() - self._global_state[name])
        return {
            "backbone": {k: v.detach().cpu() for k, v in local_state.items()},
            "accuracy": correct / total,
            "loss": running_loss / steps,
        }


class FedDynServer(AdaptationAnchorServer):
    method_name = "feddyn"
    source_reference = "Acar et al., ICLR 2021; alpemreacar/FedDyn"

    def fit(self):
        sample_counts = [len(client.train_set) for client in self.clients]
        mean_count = sum(sample_counts) / len(sample_counts)
        if mean_count <= 0:
            raise RuntimeError("FedDyn requires every participating client to have training data")
        for client, count in zip(self.clients, sample_counts):
            if count <= 0:
                raise RuntimeError("FedDyn does not support an empty client partition")
            client.alpha_scale = mean_count / count
        return super().fit()

    def aggregate(self, local_weights_list, local_deltas_list, client_ids, client_weights=None):
        result = super().aggregate(
            local_weights_list, local_deltas_list, client_ids, client_weights=client_weights
        )
        parameter_names = dict(self.backbone.named_parameters())
        for name in parameter_names:
            mean_history = torch.zeros_like(result[name])
            for client in self.clients:
                mean_history.add_(client.history[name], alpha=1.0 / len(self.clients))
            result[name].add_(mean_history)
        return result
