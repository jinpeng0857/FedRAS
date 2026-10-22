from pathlib import Path
path = Path("/mnt/sda/PythonProject/CJP_project/FEDEATA_revision_20261018/scripts/revision_20261018/run_case.py")
s = path.read_text(encoding="utf-8")
old = """            from evaluator.dataset import CORRUPTIONS
            evaluation_args = args[0]
            labels = Path(evaluation_args.cifar_c_path) / \"labels.npy\"
            binding = {
"""
new = """            from evaluator.dataset import CORRUPTIONS
            evaluation_args = args[0]
            if evaluation_args.dataset == \"tiny-imagenet\":
                # Tiny-ImageNet-C is directory-form and has no CIFAR-C labels.npy.
                # Bind the stream to the clean class manifest and per-corruption
                # severity directory inventory instead.
                labels = Path(evaluation_args.clean_data_root) / \"wnids.txt\"
                corruption_root = Path(evaluation_args.cifar_c_path)
                data_files = {}
                for name in CORRUPTIONS:
                    severity_root = corruption_root / name / \"5\"
                    files = [p for p in severity_root.rglob(\"*\") if p.is_file()]
                    data_files[name] = {
                        \"files\": len(files),
                        \"bytes\": sum(p.stat().st_size for p in files),
                        \"max_mtime_ns\": max((p.stat().st_mtime_ns for p in files), default=0),
                    }
            else:
                labels = Path(evaluation_args.cifar_c_path) / \"labels.npy\"
                data_files = {
                    name: {\"bytes\": (labels.parent / f\"{name}.npy\").stat().st_size,
                           \"mtime_ns\": (labels.parent / f\"{name}.npy\").stat().st_mtime_ns}
                    for name in CORRUPTIONS
                }
            binding = {
"""
if old not in s:
    raise SystemExit('anchor block not found')
s = s.replace(old, new, 1)
old2 = """                \"data_files\": {
                    name: {\"bytes\": (labels.parent / f\"{name}.npy\").stat().st_size,
                           \"mtime_ns\": (labels.parent / f\"{name}.npy\").stat().st_mtime_ns}
                    for name in CORRUPTIONS
                },
"""
new2 = """                \"data_files\": data_files,
"""
if old2 not in s:
    raise SystemExit('data_files block not found')
s = s.replace(old2, new2, 1)
path.write_text(s, encoding="utf-8", newline="\n")
print('patched', path)
