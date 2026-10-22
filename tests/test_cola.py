import unittest

import torch
from torch import nn

from adapters.cola import (
    CoLAAdapter,
    CoLABatchNorm2d,
    CoLAOptimizer,
    CoLAProxyRestorer,
    configure_cola_model,
)


class _TinyResNetLike(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 4, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(4)
        self.fc = nn.Linear(4, 2)

    def forward(self, images):
        features = torch.relu(self.bn1(self.conv1(images)))
        features = features.mean(dim=(2, 3))
        return self.fc(features)


class CoLAContractTests(unittest.TestCase):
    def test_configure_replaces_bn_and_preserves_fused_function_on_save(self):
        torch.manual_seed(7)
        model = _TinyResNetLike()
        params, names = configure_cola_model(model)
        self.assertIsInstance(model.bn1, CoLABatchNorm2d)
        self.assertIn("bn1.weight", names)
        self.assertIn("bn1.bias", names)
        self.assertIn("cola_state.alpha", names)
        self.assertEqual(model.cola_state.vector_count, 1)

        images = torch.randn(4, 3, 8, 8)
        with torch.no_grad():
            before = model(images)
            model.cola_state.save_current_weight()
            after = model(images)
        self.assertEqual(model.cola_state.vector_count, 2)
        self.assertTrue(torch.allclose(before, after, atol=1e-6, rtol=1e-6))
        self.assertGreater(len(params), 0)

    def test_eta_cola_adapter_runs_pre_update_contract(self):
        torch.manual_seed(11)
        model = _TinyResNetLike()
        configure_cola_model(model)
        optimizer = CoLAOptimizer(
            model.cola_state,
            [
                parameter
                for name, parameter in model.named_parameters()
                if parameter.requires_grad and not name.startswith("cola_state.")
            ],
            epsilon_weight_decay=0.4,
        )
        restorer = CoLAProxyRestorer(model)
        adapter = CoLAAdapter(
            model,
            optimizer,
            restorer,
            num_classes=2,
            entropy_margin_factor=10.0,
        )
        images = torch.randn(4, 3, 8, 8)
        with torch.no_grad():
            expected = model(images)
        logits, stats = adapter.adapt(images)
        self.assertTrue(torch.allclose(logits, expected, atol=1e-6, rtol=1e-6))
        self.assertEqual(tuple(logits.shape), (4, 2))
        self.assertEqual(stats["samples"], 4)
        self.assertEqual(stats["selected"], 4)
        self.assertTrue(stats["updated"])

    def test_vector_cap_keeps_alpha_and_vector_pools_aligned(self):
        model = _TinyResNetLike()
        configure_cola_model(model, max_num_vectors=2)
        images = torch.randn(4, 3, 8, 8)
        with torch.no_grad():
            model.bn1.weight.add_(0.02)
            model.cola_state.mark_updated()
            model.cola_state.save_current_weight()
            model.bn1.weight.add_(0.02)
            model.cola_state.mark_updated()
            model.cola_state.save_current_weight()
            model.bn1.weight.add_(0.02)
            before_cap = model(images)
            model.cola_state.mark_updated()
            model.cola_state.save_current_weight()
            after_cap = model(images)
        self.assertEqual(model.cola_state.vector_count, 2)
        self.assertEqual(
            model.cola_state.alpha.numel(), model.cola_state.vector_count
        )
        self.assertTrue(torch.allclose(before_cap, after_cap, atol=1e-6, rtol=1e-6))


if __name__ == "__main__":
    unittest.main()
