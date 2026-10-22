"""Bounded real-checkpoint acceptance; run from the remote FEDEATA root."""
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path.cwd()))
import main_test as mt
from evaluator.engine import build_main_adapter, load_feature_statistics
from test.tta_baselines import RobustBatchNorm2d


def changed(before, after):
    return sum(not torch.equal(v, after[k]) for k, v in before.items())


def check(run, method):
    directory = 'cifar100_resnet18_run1' if run == 1 else f'run{run}'
    checkpoint = f'train/results/cifar100_100clients_alpha0.3_equal/fedacg_resnet18/{directory}/model_round499.pt'
    statistics = f'test/outputs/fedacg_run{run}_statistics.pt'
    sys.argv = ['check_core', '--model_path', checkpoint, '--statistics_path', statistics,
                '--tta_method', method, '--device', 'cpu', '--output_path', 'unused.json']
    args = mt.parse_args()
    args._adapt_lr_explicit = args.adapt_lr is not None
    args.public_tta_method = method
    args = mt.apply_official_baseline_config(args)
    args = mt.apply_feddr_primary_config(args, method)
    args = mt.apply_cola_primary_config(args)
    torch.manual_seed(0)
    variance, meta = load_feature_statistics(statistics)
    adapter, metadata = build_main_adapter(args, torch.device('cpu'), variance, mt.load_fishers(statistics), meta)
    record = {'run': run, 'method': method, 'checkpoint': checkpoint, 'statistics': statistics,
              'device': 'cpu', 'batch_size': 64, 'augmentation_count': getattr(adapter, 'augmentation_count', None),
              'memory_size': 64 if method == 'petta' else None,
              'update_frequency': getattr(adapter, 'update_frequency', None), 'batches': []}
    for index, corruption in enumerate(['gaussian_noise', 'fog']):
        images = torch.from_numpy(np.array(np.load(f'data/CIFAR-100-C/{corruption}.npy', mmap_mode='r')[40000:40064]))
        images = images.permute(0, 3, 1, 2).float() / 255
        images = (images - images.new_tensor([0.5071, 0.4867, 0.4408]).view(1, 3, 1, 1)) / images.new_tensor([0.2675, 0.2565, 0.2761]).view(1, 3, 1, 1)
        parameters = {n: p.detach().clone() for n, p in adapter.model.named_parameters() if p.requires_grad}
        teacher_parameters = {n: p.detach().clone() for n, p in adapter.model_ema.named_parameters()}
        teacher_state = {n: v.detach().clone() for n, v in adapter.model_ema.state_dict().items()}
        with torch.no_grad():
            prediction_before = adapter.predict(images).clone()
        start = time.perf_counter()
        logits, stats = adapter.adapt(images)
        seconds = time.perf_counter() - start
        with torch.no_grad():
            prediction_after = adapter.predict(images).clone()
        batch = {'corruption': corruption, 'severity': 5, 'samples': 64,
                 'finite_logits': bool(torch.isfinite(logits).all()), 'updated': stats['updated'],
                 'finite_loss': bool(np.isfinite(stats['loss'])), 'loss': stats['loss'],
                 'student_parameter_tensors_changed': changed(parameters, dict(adapter.model.named_parameters())),
                 'teacher_parameter_tensors_changed': changed(teacher_parameters, dict(adapter.model_ema.named_parameters())),
                 'teacher_state_tensors_changed': changed(teacher_state, adapter.model_ema.state_dict()),
                 'same_input_prediction_max_abs_change': float((prediction_after - prediction_before).abs().max()),
                 'seconds': seconds}
        if method == 'petta':
            batch.update(divergence=adapter.last_divergence, teacher_alpha=adapter.nu,
                         robust_bn_eval_after_update=all(not m.training for m in adapter.model_ema.modules() if isinstance(m, RobustBatchNorm2d)))
            if index == 0:
                assert adapter.last_divergence == 0.0
                assert abs(adapter.nu - 0.001) < 1e-8
                assert torch.equal(logits, prediction_before), 'PeTTA must return pre-update teacher logits'
        else:
            batch.update(augmentation_used=stats['teacher_augmentation_used'],
                         stochastic_restored_scalars=stats['stochastic_restored_scalars'])
        assert tuple(logits.shape) == (64, 100)
        assert batch['finite_logits'] and batch['finite_loss'] and batch['updated']
        assert batch['student_parameter_tensors_changed'] > 0
        assert batch['teacher_state_tensors_changed'] > 0
        assert batch['same_input_prediction_max_abs_change'] > 0
        if index == 0:
            assert batch['teacher_parameter_tensors_changed'] > 0
        record['batches'].append(batch)
        print(json.dumps({'run': run, 'method': method, **batch}), flush=True)
    record['passed'] = True
    return record


if __name__ == '__main__':
    results = []
    output = Path('artifacts/petta_cotta_fix_20261008/core_acceptance.json')
    for run in [1, 2, 3]:
        for method in ['petta', 'cotta']:
            results.append(check(run, method))
            output.write_text(json.dumps({'passed': len(results) == 6, 'cases': results}, indent=2), encoding='utf-8')
    print('CORE_ACCEPTANCE_PASSED=6/6', flush=True)
