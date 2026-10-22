"""Run one main_test TTA method with the unified Tiny-ImageNet-C order."""
import sys
import json
import hashlib
import importlib
from pathlib import Path
import numpy as np

method, snapshot, output, *arguments = sys.argv[1:]
sys.path.insert(0, snapshot)
output = Path(output)
from evaluator.dataset import TinyImageNetCDataset

original_init = TinyImageNetCDataset.__init__
def ordered_init(self, *args, **kwargs):
    original_init(self, *args, **kwargs)
    assert len(self.samples) == 10000
    permutation = np.random.RandomState(0).permutation(len(self.samples))
    self.samples = [self.samples[int(i)] for i in permutation]
    records = [[str(Path(p).parent.name) + '/' + Path(p).name, int(y)] for p, y in self.samples]
    digest = hashlib.sha256(json.dumps(records).encode()).hexdigest()
    corruption = args[1] if len(args) > 1 else kwargs['corruption']
    audit = output / 'sample_orders' / (str(corruption) + '.json')
    audit.parent.mkdir(exist_ok=True)
    record = {'corruption': corruption, 'seed': 0, 'count': len(records), 'sha256': digest, 'ordered_samples': records}
    if audit.exists():
        assert json.loads(audit.read_text()) == record, 'Sample order changed'
    else:
        audit.write_text(json.dumps(record))
TinyImageNetCDataset.__init__ = ordered_init

module = importlib.import_module('main_test')
sys.argv = [method] + arguments

original_primary = module.apply_feddr_primary_config
result_path = Path(arguments[arguments.index('--output_path') + 1])
run_name = result_path.stem
def recorded_primary(args, public_method):
    args = original_primary(args, public_method)
    (output / f'{run_name}_resolved_config.json').write_text(
        json.dumps(vars(args), indent=2, default=str)
    )
    return args
module.apply_feddr_primary_config = recorded_primary

module.run()

result = json.loads(result_path.read_text())
result['protocol']['within_domain_order'] = 'RandomState(0) permutation of sorted class/file list, repeated identically'
result['protocol']['batch_size'] = 128
domain = result['domain_wise']['per_corruption']
stream = result['continual']['per_corruption']
assert len(domain) == 15 and len(stream) == 45
assert all(x['total'] == 10000 for x in domain + stream)
post = result['continual']['post_stream']
assert post['clean']['total'] == 10000 and len(post['revisit']) == 4
assert all(x['total'] == 10000 for x in post['revisit'])
result_path.write_text(json.dumps(result, indent=2))
(output / f'{run_name}_validated.json').write_text(json.dumps({
    'complete': True, 'method': method, 'domains': 15,
    'continual_domains': 45, 'samples_per_domain': 10000,
    'clean_samples': 10000, 'revisit_domains': 4,
}))
