from pathlib import Path
p = Path('/mnt/sda/PythonProject/CJP_project/FEDEATA_revision_20261018/scripts/revision_20261018/runtime.py')
text = p.read_text(encoding='utf-8')
old = '''    if controller != "native" and adapter.rga_ablation_variant == "anchor_only" and abs(getattr(adapter, "anchor_rectify_beta", -1)-beta) > 1e-12:\n        raise ValueError("loss anchor_rectify_beta must match controller beta")\n'''
new = '''    # The disagreement controller forces ESS to one, so its beta is not a\n    # controller decision.  Keep the equality guard for rga_ess, where beta\n    # controls the ESS transform; allow loss-weighting sweeps under\n    # disagreement as an independent factor.\n    if controller == "rga_ess" and adapter.rga_ablation_variant == "anchor_only" and abs(getattr(adapter, "anchor_rectify_beta", -1)-beta) > 1e-12:\n        raise ValueError("loss anchor_rectify_beta must match controller beta")\n'''
if old not in text:
    raise SystemExit('expected guard block not found')
p.with_name('runtime.py.pre_e2_guardfix').write_text(text, encoding='utf-8')
p.write_text(text.replace(old, new), encoding='utf-8')
print('RUNTIME_GUARD_PATCH_APPLIED')