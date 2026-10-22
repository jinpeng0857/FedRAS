"""Parity checks for behavior taken directly from official TTA repositories."""

import copy
import unittest

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from prepare_main_statistics import compute_fishers


class _TinyBNClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 2, 1, bias=False)
        self.bn = nn.BatchNorm2d(2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(2, 2)

    def forward(self, inputs):
        features = self.pool(self.bn(self.conv(inputs))).flatten(1)
        return self.fc(features)


class OfficialEATAFisherTests(unittest.TestCase):
    def test_official_fisher_ignores_dataset_labels_and_uses_pseudo_labels(self):
        torch.manual_seed(7)
        template = _TinyBNClassifier()
        images = torch.randn(8, 1, 4, 4)
        first_labels = torch.zeros(8, dtype=torch.long)
        second_labels = torch.ones(8, dtype=torch.long)
        first_loader = DataLoader(
            TensorDataset(images, first_labels), batch_size=4, shuffle=False
        )
        second_loader = DataLoader(
            TensorDataset(images, second_labels), batch_size=4, shuffle=False
        )

        first, first_samples, first_batches = compute_fishers(
            copy.deepcopy(template),
            first_loader,
            torch.device("cpu"),
            max_batches=-1,
            mode="eata_official",
        )
        second, second_samples, second_batches = compute_fishers(
            copy.deepcopy(template),
            second_loader,
            torch.device("cpu"),
            max_batches=-1,
            mode="eata_official",
        )

        self.assertEqual((first_samples, first_batches), (8, 2))
        self.assertEqual((second_samples, second_batches), (8, 2))
        self.assertEqual(set(first), set(second))
        for name in first:
            self.assertTrue(torch.allclose(first[name][0], second[name][0]))
            self.assertTrue(torch.allclose(first[name][1], second[name][1]))


if __name__ == "__main__":
    unittest.main()
