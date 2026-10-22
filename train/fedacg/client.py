"""Pure FedACG client.

The local objective is deliberately kept minimal:
  loss = ce_weight * CrossEntropy + prox_weight * 0.5 * ||theta - theta_global||^2

No extra algorithm branches are kept here. The client always returns a standard
model state_dict and local accuracy.
"""

import copy

import torch
import torch.nn.functional as F
import torch.optim
from torch.amp import autocast, GradScaler

from train.fedacg.base_client import BaseClient


class FedACGClient(BaseClient):
    """Client used by the pure FedACG training pipeline."""

    def __init__(self, cid, device, backbone, configs):
        super().__init__(cid, device, backbone, configs)
        # global_model is NOT deepcopied here — it would double CPU memory.
        # Instead, the global state dict is stored on CPU in setup()
        # and moved to GPU once per train() call if prox_weight > 0.
        self._global_state = None
        self._global_gpu = None

        self.optimizer = None
        self.scheduler = None
        self.ce_weight = float(getattr(configs, "ce_weight", 1.0))
        self.prox_weight = float(getattr(configs, "prox_weight", 0.0))
        self.label_smoothing = float(getattr(configs, "label_smoothing", 0.0))

    def update_global_model(self, state_dict):
        """Update the cached global state dict (CPU)."""
        self._global_state = state_dict

    def setup(self, state_dict, device, local_lr=None, global_epoch=None):
        """Load the state sent by the server and recreate the local optimizer."""
        self.backbone.load_state_dict(state_dict)
        self._global_state = state_dict
        self.device = device

        if self.train_dataloader is None and self.train_set is not None and len(self.train_set) > 0:
            from torch.utils.data import DataLoader
            self.train_dataloader = DataLoader(
                dataset=self.train_set,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.configs.num_workers,
                pin_memory=device.type == "cuda",
                drop_last=False,
            )

        lr = local_lr if local_lr is not None else self.learning_rate
        self.optimizer = torch.optim.SGD(
            params=self.backbone.parameters(),
            lr=lr,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
        )
        local_lr_decay = getattr(self.configs, "local_lr_decay", 1.0)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer=self.optimizer,
            lr_lambda=lambda epoch: local_lr_decay ** epoch,
        )

    def _proximal_loss(self):
        """0.5 * ||theta - theta_global||^2 over trainable parameters.

        Uses _global_gpu (pre-moved to GPU in train()) to avoid per-batch CPU→GPU transfer.
        Falls back to on-demand device move if _global_gpu is not pre-set.
        """
        if self.prox_weight <= 0:
            return None
        if self._global_gpu is None:
            if self._global_state is None:
                return None
            self._global_gpu = {n: v.to(self.device) for n, v in self._global_state.items()}

        prox = 0.0
        for n, p in self.backbone.named_parameters():
            g = self._global_gpu[n]
            prox = prox + ((p - g) ** 2).sum()
        return 0.5 * prox

    def train(self):
        self.backbone.train()

        # Pre-move global state to GPU once (avoids per-batch CPU→GPU transfer)
        if self.prox_weight > 0 and self._global_state is not None:
            self._global_gpu = {n: v.to(self.device) for n, v in self._global_state.items()}
        else:
            self._global_gpu = None

        amp_device = "cuda" if self.device.type == "cuda" and torch.cuda.is_available() else "cpu"
        scaler = GradScaler(amp_device, enabled=getattr(self.configs, "use_amp", False))

        correct, total = 0, 0
        running_loss = 0.0
        num_batches = 0

        for _ in range(self.local_epochs):
            for data, target in self.train_dataloader:
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad(set_to_none=True)

                with autocast(amp_device, enabled=getattr(self.configs, "use_amp", False)):
                    logits = self.backbone(data)
                    ce_loss = F.cross_entropy(
                        logits,
                        target,
                        label_smoothing=self.label_smoothing,
                    )
                    total_loss = self.ce_weight * ce_loss

                    prox_loss = self._proximal_loss()
                    if prox_loss is not None:
                        total_loss = total_loss + self.prox_weight * prox_loss

                scaler.scale(total_loss).backward()
                scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.backbone.parameters(), 10.0)
                scaler.step(self.optimizer)
                scaler.update()

                pred = logits.detach().argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
                running_loss += float(total_loss.detach().cpu())
                num_batches += 1

        if self.scheduler is not None:
            self.scheduler.step()

        # Free GPU-side global state
        self._global_gpu = None

        accuracy = correct / total if total > 0 else 0.0
        avg_loss = running_loss / num_batches if num_batches > 0 else 0.0

        return {
            "backbone": {k: v.detach().cpu() for k, v in self.backbone.state_dict().items()},
            "accuracy": accuracy,
            "loss": avg_loss,
        }
