"""Base Client class for federated learning."""

import copy

import torch
from torch.utils.data import DataLoader


class BaseClient:
    def __init__(self, cid, device, backbone, configs):
        self.cid = cid
        self.device = device
        self.configs = configs
        self.backbone = copy.deepcopy(backbone)
        # Store a copy of the original backbone for anti-forgetting
        self.original_backbone = copy.deepcopy(backbone)
        self.local_epochs = configs.local_epochs
        self.learning_rate = configs.learning_rate
        self.momentum = configs.momentum
        self.weight_decay = configs.weight_decay
        self.batch_size = configs.batch_size
        self.debug = configs.debug
        self.train_set = None
        self.test_set = None
        self.train_dataloader = None
        self.test_dataloader = None

    def set_train_set(self, train_set):
        self.train_set = train_set
        # Guard: if dataset is empty, leave dataloader as None
        if len(train_set) == 0:
            self.train_dataloader = None
            return
        self.train_dataloader = DataLoader(
            dataset=self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.configs.num_workers,
            pin_memory=self.device.type == "cuda",
            drop_last=False,
        )

    def set_test_set(self, test_set, test_batch_size):
        self.test_set = test_set
        self.test_batch_size = test_batch_size
        self.test_dataloader = DataLoader(
            dataset=self.test_set,
            batch_size=test_batch_size,
            shuffle=False,
            num_workers=self.configs.num_workers,
            pin_memory=self.device.type == "cuda",
        )

    def global_accuracy(self):
        self.backbone.to(self.device)
        self.backbone.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for data, target in self.train_dataloader:
                data, target = data.to(self.device), target.to(self.device)
                result = self.backbone(data)
                logits = result["logit"] if isinstance(result, dict) else result
                pred = logits.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
        self.backbone.cpu()
        return correct / total

