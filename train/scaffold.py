"""SCAFFOLD Option-II control variates for the FedDR supplements."""

import copy

import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast

from train.fedacg.client import FedACGClient
from train.common import AdaptationAnchorServer


class ScaffoldClient(FedACGClient):
    """PFLlib-compatible client controls adapted to the FEDEATA interface."""

    def __init__(self, cid, device, backbone, configs):
        super().__init__(cid, device, backbone, configs)
        self.client_control = {
            name: torch.zeros_like(value, device="cpu")
            for name, value in backbone.named_parameters()
        }
        self.server_control = None
        self.control_delta = None

    def setup(self, state_dict, device, local_lr=None, global_epoch=None):
        super().setup(state_dict, device, local_lr=local_lr, global_epoch=global_epoch)
        lr = local_lr if local_lr is not None else self.learning_rate
        self.optimizer = torch.optim.SGD(self.backbone.parameters(), lr=lr)
        self.scheduler = None

    def train(self):
        if self.server_control is None:
            raise RuntimeError("SCAFFOLD server control has not been attached")
        self.backbone.train()
        amp_device = "cuda" if self.device.type == "cuda" and torch.cuda.is_available() else "cpu"
        scaler = GradScaler(amp_device, enabled=getattr(self.configs, "use_amp", False))
        old_control = copy.deepcopy(self.client_control)
        initial_state = copy.deepcopy(self._global_state)
        server_control = {key: value.to(self.device) for key, value in self.server_control.items()}
        client_control = {key: value.to(self.device) for key, value in self.client_control.items()}
        correct = total = steps = 0
        running_loss = 0.0

        for _ in range(self.local_epochs):
            for data, target in self.train_dataloader:
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad(set_to_none=True)
                with autocast(amp_device, enabled=getattr(self.configs, "use_amp", False)):
                    logits = self.backbone(data)
                    loss = F.cross_entropy(logits, target, label_smoothing=self.label_smoothing)
                scaler.scale(loss).backward()
                scaler.unscale_(self.optimizer)
                for name, parameter in self.backbone.named_parameters():
                    if parameter.grad is not None:
                        parameter.grad.add_(server_control[name] - client_control[name])
                torch.nn.utils.clip_grad_norm_(self.backbone.parameters(), 10.0)
                scaler.step(self.optimizer)
                scaler.update()
                steps += 1
                correct += logits.detach().argmax(1).eq(target).sum().item()
                total += target.numel()
                running_loss += float(loss.detach().cpu())

        if steps == 0:
            raise RuntimeError(f"client {self.cid} completed no local steps")
        lr = float(self.optimizer.param_groups[0]["lr"])
        local_state = self.backbone.state_dict()
        for name in self.client_control:
            self.client_control[name] = (
                old_control[name]
                - self.server_control[name]
                + (initial_state[name] - local_state[name].detach().cpu()) / (steps * lr)
            )
        self.control_delta = {
            name: self.client_control[name] - old_control[name]
            for name in self.client_control
        }
        return {
            "backbone": {key: value.detach().cpu() for key, value in local_state.items()},
            "accuracy": correct / total,
            "loss": running_loss / steps,
        }


class ScaffoldServer(AdaptationAnchorServer):
    method_name = "scaffold"
    source_reference = "Karimireddy et al., ICML 2020; chenjinpeng031220/PFLlib"

    def __init__(self, device, backbone, configs):
        super().__init__(device, backbone, configs)
        self.server_learning_rate = float(configs.scaffold_server_lr)
        self.server_control = {
            name: torch.zeros_like(value, device="cpu")
            for name, value in backbone.named_parameters()
        }

    def fit(self):
        for client in self.clients:
            client.server_control = self.server_control
        return super().fit()

    def aggregate(self, local_weights_list, local_deltas_list, client_ids, client_weights=None):
        del local_deltas_list, client_weights
        if not local_weights_list:
            raise ValueError("SCAFFOLD received no client updates")
        result = copy.deepcopy(self.backbone.state_dict())
        selected = float(len(client_ids))
        for name, parameter in self.backbone.named_parameters():
            delta = sum(
                local_state[name].detach().cpu() - parameter.detach().cpu()
                for local_state in local_weights_list
            ) / selected
            result[name] = parameter.detach().cpu() + self.server_learning_rate * delta
        total_clients = float(len(self.clients))
        for cid in client_ids:
            delta = self.clients[cid].control_delta
            for name in self.server_control:
                self.server_control[name].add_(delta[name], alpha=1.0 / total_clients)
        return result
