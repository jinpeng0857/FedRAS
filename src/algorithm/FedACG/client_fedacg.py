"""
FedACG Client: multi-loss federated learning client.

Supports configurable losses:
  - CE (cross-entropy, always on)
  - FedProx (L2 proximal term to global model)
  - FedLC (label-calibrated logit calibration)
  - FedDecorr (feature decorrelation)
  - FedMLB (multi-level branch KL distillation)
  - FedNTD (not-true-class distillation)
  - FedDyn (dynamic regularization)

All losses are toggled via configs.fedacg_mode.
"""

import copy
import random
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
from src.algorithm.Base.client_base import BaseClient
from utils.logger import logger


# ---- Custom Losses (from ACG-C utils/loss.py) ----

def fedlc_loss(label_distrib, logit, labels, tau=1.0):
    """
    Label-Calibrated Cross Entropy Loss.
    Adjusts logits by inverse class frequency: logit - tau * freq^(-1/4)
    """
    num_classes = logit.size(1)
    freq = (label_distrib + 1) / (label_distrib.sum() + num_classes)
    inv_freq = freq.pow(-0.25)
    calibrated_logit = logit - tau * inv_freq.unsqueeze(0)
    return F.cross_entropy(calibrated_logit, labels)


def feddecorr_loss(feature):
    """
    Feature Decorrelation Loss.
    Penalizes off-diagonal entries of the feature correlation matrix.
    """
    if feature.dim() > 2:
        feature = feature.view(feature.size(0), -1)
    feature = feature - feature.mean(dim=0, keepdim=True)
    cov = (feature.t() @ feature) / (feature.size(0) - 1)
    diag = torch.diag(cov)
    std = torch.sqrt(diag + 1e-8)
    corr = cov / (std.unsqueeze(0) * std.unsqueeze(1) + 1e-8)
    # Sum of squared off-diagonal elements
    mask = torch.eye(corr.size(0), device=corr.device)
    return (corr * (1 - mask)).pow(2).sum() / 2


def kd_loss(student_logits, teacher_logits, T=4.0):
    """Knowledge Distillation KL divergence loss."""
    student = F.log_softmax(student_logits / T, dim=1)
    teacher = F.softmax(teacher_logits / T, dim=1).detach()
    return F.kl_div(student, teacher, reduction="batchmean") * (T * T)


class FedACGClient(BaseClient):
    """
    Multi-loss FL client supporting all ACG-C algorithms.
    Selects which losses to apply based on configs.fedacg_mode.
    """

    def __init__(self, cid, device, backbone, configs):
        super().__init__(cid, device, backbone, configs)
        # Store a copy of the global model (updated each round via server)
        self.global_model = copy.deepcopy(backbone)
        for p in self.global_model.parameters():
            p.requires_grad = False

        self.optimizer = torch.optim.SGD(
            params=self.backbone.parameters(),
            lr=self.learning_rate,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
        )

        # Control which losses are active
        self.mode = configs.fedacg_mode
        self.use_prox = (self.mode in ["fedprox", "feddyn", "fedacg"])
        self.use_lc = (self.mode == "fedlc")
        self.use_decorr = (self.mode == "feddecorr")
        self.use_mlb = (self.mode == "fedmlb")
        self.use_ntd = (self.mode == "fedntd")
        self.use_dyn = (self.mode == "feddyn")

        # FedDyn state
        if self.use_dyn:
            self.local_delta = None

    def update_global_model(self, state_dict):
        self.global_model.load_state_dict(state_dict)

    def setup_local_stats(self):
        """Compute label distribution for FedLC."""
        if not self.use_lc:
            return
        labels_list = []
        for _, labels in self.train_dataloader:
            labels_list.append(labels)
        all_labels = torch.cat(labels_list)
        num_classes = self.configs.num_class
        self.label_distrib = torch.zeros(num_classes, device=self.device)
        for c in range(num_classes):
            self.label_distrib[c] = (all_labels == c).sum()

    def train(self):
        self.backbone.to(self.device)
        self.global_model.to(self.device)
        self.backbone.train()

        if self.use_lc and not hasattr(self, "label_distrib"):
            self.setup_local_stats()

        correct, total = 0, 0

        for epoch in range(self.local_epochs):
            for data, target in self.train_dataloader:
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()

                use_subnet = (
                    hasattr(self.configs, 'subnet_train_prob')
                    and self.configs.subnet_train_prob > 0
                    and random.random() < self.configs.subnet_train_prob
                )
                result = self.backbone(data, return_sub_network=use_subnet)
                logits = result["logit"]
                feature = result.get("feature", None)

                losses = defaultdict(float)

                # 1. Cross-entropy (always on)
                loss_ce = F.cross_entropy(logits, target)
                losses["cls"] = loss_ce

                # 2. FedProx proximal term
                if self.use_prox:
                    prox_loss = 0
                    fixed_params = {
                        n: p for n, p in self.global_model.named_parameters()
                    }
                    for n, p in self.backbone.named_parameters():
                        prox_loss += ((p - fixed_params[n].detach()) ** 2).sum()
                    losses["prox"] = prox_loss * self.configs.prox_weight

                # 3. FedLC label calibration
                if self.use_lc:
                    lc_loss = fedlc_loss(
                        self.label_distrib, logits, target, self.configs.lc_tau
                    )
                    losses["LC"] = lc_loss * self.configs.lc_weight

                # 4. FedDecorr feature decorrelation
                if self.use_decorr and feature is not None:
                    decorr_loss = feddecorr_loss(feature)
                    losses["decorr"] = decorr_loss * self.configs.decorr_weight

                # 5. FedMLB multi-level branch
                if self.use_mlb:
                    with torch.no_grad():
                        global_result = self.global_model(data)
                    mlb_cls = F.cross_entropy(global_result["logit"], target)
                    mlb_kl = kd_loss(logits, global_result["logit"],
                                     T=self.configs.mlb_temp)
                    losses["MLB_cls"] = mlb_cls * self.configs.mlb_branch_cls_weight
                    losses["MLB_kl"] = mlb_kl * self.configs.mlb_branch_kl_weight

                # 6. FedNTD not-true distillation
                if self.use_ntd:
                    with torch.no_grad():
                        global_result = self.global_model(data)
                    global_logits = global_result["logit"]
                    batch_size = target.size(0)
                    idxs = torch.arange(
                        logits.size(1), device=self.device
                    ).unsqueeze(0).repeat(batch_size, 1)
                    not_true_mask = idxs != target.unsqueeze(1)
                    not_true_logits = logits[not_true_mask].view(
                        batch_size, logits.size(1) - 1
                    )
                    not_true_global = global_logits[not_true_mask].view(
                        batch_size, logits.size(1) - 1
                    )
                    ntd_loss = kd_loss(not_true_logits, not_true_global,
                                       T=self.configs.ntd_temp)
                    losses["NTD"] = ntd_loss * self.configs.ntd_weight
                    # CE weight reduced
                    losses["cls"] = losses["cls"] * (1 - self.configs.ntd_weight)

                # 7. FedDyn dynamic regularization
                if self.use_dyn and self.local_delta is not None:
                    lg_loss = 0
                    for n, p in self.backbone.named_parameters():
                        p_flat = p.view(-1)
                        local_d = self.local_delta[n].detach().clone().to(self.device)
                        local_grad = local_d.view(-1)
                        lg_loss += (p_flat * local_grad).sum()
                    # FedDyn prox term
                    prox_loss = 0
                    fixed_params = {
                        n: p for n, p in self.global_model.named_parameters()
                    }
                    for n, p in self.backbone.named_parameters():
                        prox_loss += ((p - fixed_params[n].detach()) ** 2).sum()
                    losses["Dyn"] = (
                        -lg_loss + 0.5 * self.configs.dyn_alpha * prox_loss
                    ) * self.configs.prox_weight

                total_loss = sum(losses.values())
                total_loss.backward()
                self.optimizer.step()

                # Accuracy tracking
                pred = logits.data.max(1)[1]
                correct += pred.eq(target).sum().item()
                total += target.size(0)

        accuracy = correct / total if total > 0 else 0.0

        # FedDyn: update local deltas
        if self.use_dyn:
            fixed_params = {
                n: p for n, p in self.global_model.named_parameters()
            }
            if self.local_delta is None:
                self.local_delta = copy.deepcopy(
                    {n: torch.zeros_like(p).cpu() for n, p in fixed_params.items()}
                )
            for n, _ in self.backbone.named_parameters():
                if n in self.local_delta:
                    self.local_delta[n] = (
                        self.local_delta[n]
                        - self.configs.dyn_alpha
                        * (self.backbone.state_dict()[n]
                           - fixed_params[n].detach().clone()).cpu()
                    )

        self.backbone.cpu()
        self.global_model.cpu()
        return {"backbone": self.backbone.state_dict(), "accuracy": accuracy}
