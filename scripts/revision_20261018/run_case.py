"""Opt-in revision evaluation entrypoint; the historical entrypoint stays intact."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--revision-controller", choices=["native", "disagreement", "rga_ess"], default="native")
    parser.add_argument("--revision-schedule-record")
    parser.add_argument("--revision-schedule-replay")
    parser.add_argument("--revision-ranking", choices=["random", "fisher_only", "drift_only", "fisher_drift"], default="fisher_drift")
    parser.add_argument("--revision-selection-seed", type=int, default=20261018)
    parser.add_argument("--revision-track", choices=["shared", "reasonable"], default="shared")
    parser.add_argument("--revision-warmup", action="store_true",
                        help="Run one unmeasured evaluation in the same process before measuring.")
    own, remaining = parser.parse_known_args()
    if own.revision_warmup and (
        own.revision_schedule_record or own.revision_schedule_replay
        or any(arg.split("=", 1)[0] == "--review_diagnostics_path" for arg in remaining)
    ):
        parser.error("--revision-warmup cannot be combined with schedules or review diagnostics")
    sys.argv = [str(ROOT / "main_test.py"), *remaining]
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    import main_test
    import evaluator.engine as engine

    # A small fixed CPU thread count prevents GPU evaluation from creating
    # dozens of competing OpenMP workers. This does not change the method.
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    adapters = []
    captured = {}
    original_config = main_test.apply_official_baseline_config

    def configure(args):
        args = original_config(args)
        if own.revision_track == "reasonable":
            from scripts.revision_20261018.baseline_presets import configure_reasonable
            args = configure_reasonable(args)
        captured["args"] = args
        return args

    main_test.apply_official_baseline_config = configure
    original_factory = engine.build_main_adapter

    def factory(*args, **kwargs):
        if own.revision_track == "reasonable":
            from scripts.revision_20261018.baseline_presets import reasonable_factory_scope
            with reasonable_factory_scope(args[0]):
                adapter, metadata = original_factory(*args, **kwargs)
        else:
            adapter, metadata = original_factory(*args, **kwargs)
        if own.revision_track == "reasonable":
            from scripts.revision_20261018.baseline_presets import install_reasonable_adapter
            adapter, metadata = install_reasonable_adapter(adapter, metadata, args[0])
        if own.revision_controller != "native" or own.revision_schedule_record or own.revision_schedule_replay:
            from evaluator.dataset import CORRUPTIONS
            evaluation_args = args[0]
            labels = Path(evaluation_args.cifar_c_path) / "labels.npy"
            binding = {
                "dataset": evaluation_args.dataset,
                "corruptions": list(CORRUPTIONS), "cycles": 3, "severity": 5,
                "batch_size": evaluation_args.batch_size,
                "max_batches": evaluation_args.max_batches,
                "seed": evaluation_args.seed,
                "model_path": str(Path(evaluation_args.model_path).resolve()),
                "statistics_path": str(Path(evaluation_args.statistics_path).resolve()),
                "model_sha256": hashlib.sha256(Path(evaluation_args.model_path).read_bytes()).hexdigest(),
                "statistics_sha256": hashlib.sha256(Path(evaluation_args.statistics_path).read_bytes()).hexdigest(),
                "algorithm_config": {
                    k: v for k, v in vars(evaluation_args).items()
                    if not k.startswith("_")
                    and k not in {"output_path", "review_run_id", "review_diagnostics_path"}
                },
                "executed_sources": {
                    str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in [ROOT / "main_test.py", ROOT / "evaluator/engine.py", ROOT / "evaluator/dataset.py", ROOT / "adapters/base.py", ROOT / "adapters/rga.py", ROOT / "sparse_restore.py", Path(__file__), ROOT / "scripts/revision_20261018/runtime.py"]
                },
                "corruption_root": str(Path(evaluation_args.cifar_c_path).resolve()),
                "labels_sha256": hashlib.sha256(labels.read_bytes()).hexdigest(),
                "data_files": {
                    name: {"bytes": (labels.parent / f"{name}.npy").stat().st_size,
                           "mtime_ns": (labels.parent / f"{name}.npy").stat().st_mtime_ns}
                    for name in CORRUPTIONS
                },
            }
            if own.revision_schedule_replay:
                recorded = json.loads(Path(own.revision_schedule_replay + ".stream.json").read_text(encoding="utf-8"))
                # Compare the JSON representation so argparse tuple defaults agree
                # with the lists stored in the reference manifest.
                current = json.loads(json.dumps(binding))
                mismatches = sorted(
                    key for key in recorded.keys() | current.keys()
                    if recorded.get(key) != current.get(key)
                )
                if mismatches:
                    raise ValueError(
                        "reference schedule evaluation stream identity mismatch: "
                        + ", ".join(mismatches)
                    )
            if own.revision_schedule_record:
                manifest = Path(own.revision_schedule_record + ".stream.json")
                manifest.parent.mkdir(parents=True, exist_ok=True)
                manifest.write_text(json.dumps(binding, indent=2), encoding="utf-8")
            from scripts.revision_20261018.runtime import install_mechanism_controls
            adapter = install_mechanism_controls(
                adapter, controller=own.revision_controller,
                beta=0.3, schedule_record=own.revision_schedule_record,
                schedule_replay=own.revision_schedule_replay,
                ranking=own.revision_ranking,
                selection_seed=own.revision_selection_seed,
            )
        adapters.append(adapter)
        return adapter, metadata

    engine.build_main_adapter = factory
    main_test.build_main_adapter = factory
    if own.revision_warmup:
        measured_argv = sys.argv
        with tempfile.TemporaryDirectory(prefix="fedras-warmup-") as warmup_dir:
            sys.argv = [*measured_argv, "--output_path", str(Path(warmup_dir) / "result.json")]
            try:
                main_test.run()
            finally:
                sys.argv = measured_argv
                adapters.clear()
                captured.clear()
    started = time.time()
    main_test.run()
    if own.revision_controller != "native" or own.revision_schedule_record or own.revision_schedule_replay:
        from scripts.revision_20261018.runtime import finalize_mechanism_controls
        for index, adapter in enumerate(adapters):
            finalize_mechanism_controls(adapter, diagnostics_path=str(captured["args"].output_path) + f".mechanism{index}.audit.json")
    args = captured["args"]
    output = Path(args.output_path)
    result = json.loads(output.read_text(encoding="utf-8"))
    result["revision"] = {
        "testing_only": True, "complete": True, "training_seed_provenance": "unknown",
        "replication_unit": "stored_checkpoint", "controller": own.revision_controller,
        "baseline_track": own.revision_track, "ranking": own.revision_ranking,
        "schedule_record": own.revision_schedule_record,
        "schedule_replay": own.revision_schedule_replay,
        "elapsed_wall_seconds": time.time() - started,
        "in_process_warmup": own.revision_warmup,
        "source_hashes": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [ROOT / "main_test.py", ROOT / "adapters/rga.py", ROOT / "sparse_restore.py", ROOT / "evaluator/engine.py", *sorted((ROOT / "scripts/revision_20261018").glob("*.py"))]
        },
        "mechanism_audits": [str(args.output_path) + f".mechanism{index}.audit.json" for index in range(len(adapters))] if own.revision_controller != "native" or own.revision_schedule_record or own.revision_schedule_replay else [],
    }
    temporary = output.with_suffix(".revision.tmp")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
