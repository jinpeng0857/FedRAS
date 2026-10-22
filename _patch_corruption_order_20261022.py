from pathlib import Path
main = Path('/mnt/sda/PythonProject/CJP_project/FEDEATA_revision_20261018/main_test.py')
s = main.read_text(encoding='utf-8')
old = '    parser.add_argument("--review_continual_only", action="store_true")\n'
new = old + '    parser.add_argument("--corruption_order_seed", type=int, default=None)\n'
if old not in s:
    raise SystemExit('main_test anchor not found')
s = s.replace(old, new, 1)
main.write_text(s, encoding='utf-8', newline='\n')

engine = Path('/mnt/sda/PythonProject/CJP_project/FEDEATA_revision_20261018/evaluator/engine.py')
s = engine.read_text(encoding='utf-8')
old = 'import math\nfrom pathlib import Path\n'
new = 'import math\nimport random\nfrom pathlib import Path\n'
if old not in s:
    raise SystemExit('engine import anchor not found')
s = s.replace(old, new, 1)
old = '\ndef run_domain_wise(args, device, feature_variance, fishers, statistics_meta):\n'
new = '''\ndef _stream_order(args):
    order = list(CORRUPTIONS)
    seed = getattr(args, "corruption_order_seed", None)
    if seed is not None:
        random.Random(int(seed)).shuffle(order)
    return order


def run_domain_wise(args, device, feature_variance, fishers, statistics_meta):
'''
if old not in s:
    raise SystemExit('domain function anchor not found')
s = s.replace(old, new, 1)
old = '''    results = []
    metadata = None
    for corruption in CORRUPTIONS:
'''
new = '''    results = []
    metadata = None
    order = _stream_order(args)
    for corruption in order:
'''
if old not in s:
    raise SystemExit('domain loop anchor not found')
s = s.replace(old, new, 1)
old = '''        "state_continuity": False,
        "summary": summarize_corruptions(results),
'''
new = '''        "state_continuity": False,
        "corruption_order": order,
        "summary": summarize_corruptions(results),
'''
if old not in s:
    raise SystemExit('domain return anchor not found')
s = s.replace(old, new, 1)
old = '''    results = []
    cycle_summaries = []
    for cycle in range(1, CONTINUAL_CYCLES + 1):
'''
new = '''    results = []
    cycle_summaries = []
    order = _stream_order(args)
    for cycle in range(1, CONTINUAL_CYCLES + 1):
'''
if old not in s:
    raise SystemExit('continual loop anchor not found')
s = s.replace(old, new, 1)
old = '''            evaluate_corruption(args, adapter, corruption, device, cycle=cycle)
            for corruption in CORRUPTIONS
'''
new = '''            evaluate_corruption(args, adapter, corruption, device, cycle=cycle)
            for corruption in order
'''
if old not in s:
    raise SystemExit('continual comprehension anchor not found')
s = s.replace(old, new, 1)
old = '''        "state_continuity": True,
        "cycles": CONTINUAL_CYCLES,
'''
new = '''        "state_continuity": True,
        "corruption_order": order,
        "cycles": CONTINUAL_CYCLES,
'''
if old not in s:
    raise SystemExit('continual return anchor not found')
s = s.replace(old, new, 1)
engine.write_text(s, encoding='utf-8', newline='\n')
print('patched main_test.py and engine.py')
