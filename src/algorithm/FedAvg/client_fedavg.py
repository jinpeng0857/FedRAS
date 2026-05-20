"""FedAvg Client: standard cross-entropy training + TTA test methods."""

import copy
import torch
import torch.nn.functional as F
import torch.optim
from torchvision import transforms
from src.algorithm.Base.client_base import BaseClient


class FedAvgClient(BaseClient):
    def __init__(self, cid, device, backbone, configs):
        super().__init__(cid, device, backbone, configs)
        self.optimizer = torch.optim.SGD(
            params=self.backbone.parameters(),
            lr=self.learning_rate,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
        )

    def train(self):
        self.backbone.to(self.device)
        self.backbone.train()
        correct, total = 0, 0
        for epoch in range(self.local_epochs):
            for data, target in self.train_dataloader:
                data, target = data.to(self.device), target.to(self.device)
                result = self.backbone(data)
                logits = result["logit"]
                pred = logits.data.max(1)[1]
                correct += pred.eq(target).sum().item()
                total += target.size(0)
                self.optimizer.zero_grad()
                loss = F.cross_entropy(logits, target)
                loss.backward()
                self.optimizer.step()
        accuracy = correct / total if total > 0 else 0.0
        self.backbone.cpu()
        return {"backbone": self.backbone.state_dict(), "accuracy": accuracy}

    def test(self):
        from utils.metrics import compute_ece
        self.backbone.to(self.device)
        self.backbone.eval()
        all_targets, all_probs = [], []
        with torch.no_grad():
            for data, target in self.test_dataloader:
                data, target = data.to(self.device), target.to(self.device)
                result = self.backbone(data)
                logits = result["logit"]
                probs = F.softmax(logits, dim=1)
                all_targets.append(target)
                all_probs.append(probs)
        all_targets = torch.cat(all_targets)
        all_probs = torch.cat(all_probs)
        preds = all_probs.argmax(dim=1)
        acc = (preds == all_targets).float().mean().item()
        ece = compute_ece(all_probs, all_targets)
        self.backbone.cpu()
        return {"acc": acc, "ece": ece}

    def tent_test(self, tta_lr=1e-3):
        from utils.metrics import compute_ece
        self.backbone.to(self.device)
        initial_state = copy.deepcopy(self.backbone.state_dict())
        self.backbone.train()
        tent_params = []
        for m in self.backbone.modules():
            if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.LayerNorm)):
                m.requires_grad_(True)
                if m.weight is not None: tent_params.append(m.weight)
                if m.bias is not None: tent_params.append(m.bias)
            else:
                m.requires_grad_(False)
        optimizer = torch.optim.SGD(tent_params, lr=tta_lr, momentum=0.9)
        all_targets, all_probs = [], []
        for data, target in self.test_dataloader:
            data, target = data.to(self.device), target.to(self.device)
            optimizer.zero_grad()
            result = self.backbone(data)
            logits = result["logit"]
            probs = F.softmax(logits, dim=1)
            entropy_loss = -(probs * (probs + 1e-8).log()).sum(dim=1).mean()
            entropy_loss.backward()
            optimizer.step()
            with torch.no_grad():
                all_targets.append(target)
                all_probs.append(probs.detach())
        self.backbone.requires_grad_(True)
        self.backbone.load_state_dict(initial_state)
        self.backbone.cpu()
        all_targets = torch.cat(all_targets)
        all_probs = torch.cat(all_probs)
        preds = all_probs.argmax(dim=1)
        acc = (preds == all_targets).float().mean().item()
        ece = compute_ece(all_probs, all_targets)
        return {"acc": acc, "ece": ece}
