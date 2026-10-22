"""Shared launch, validation, and reporting helpers for the six supplements.

The helpers deliberately call the repository's existing ``main_test.py`` and
``test.source_only_test`` entry points.  They do not duplicate model or data
loading code, so every supplement keeps the same 15-domain/3-cycle protocol.
"""

from __future__ import annotations

import csv
import json
import logging
import shlex
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parent.parent
CORRUPTIONS = [
    "gaussian_noise",
    "shot_noise",
    "impulse_noise",
    "defocus_blur",
    "glass_blur",
    "motion_blur",
    "zoom_blur",
    "snow",
    "frost",
    "fog",
    "brightness",
    "contrast",
    "elastic_transform",
    "pixelate",
    "jpeg_compression",
]
CONTINUAL_CYCLES = 3
REVISIT_CORRUPTIONS = CORRUPTIONS[:4]

LOGGER = logging.getLogger("supplement_experiments")


def rooted(path: str | Path) -> Path:
    """Resolve a user path relative to the repository root."""
    value = Path(path).expanduser()
    return value if value.is_absolute() else ROOT / value


@dataclass
class Task:
    """One subprocess invocation and its expected result location."""

    name: str
    command: list[str]
    output: Path
    log: Path
    metadata: dict[str, Any] = field(default_factory=dict)


def add_runtime_args(parser, *, device: str = "cuda", batch_size: int = 64,
                     num_workers: int = 2):
    """Add options shared by all supplement launchers."""
    parser.add_argument("--device", default=device)
    parser.add_argument("--batch_size", type=int, default=batch_size)
    parser.add_argument("--num_workers", type=int, default=num_workers)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--max_batches",
        type=int,
        default=-1,
        help="Development cap per corruption; <=0 means the complete stream.",
    )
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--force_rerun", action="store_true")
    parser.add_argument(
        "--allow_missing",
        action="store_true",
        help="Write the preflight and skip missing model/statistics pairs.",
    )


def add_data_args(parser, *, dataset: str, corruption_default: str,
                  clean_default: str):
    parser.add_argument("--dataset", default=dataset,
                        choices=[dataset])
    parser.add_argument("--cifar_c_path", default=corruption_default)
    parser.add_argument("--clean_data_root", default=clean_default)


def protocol_flags(args) -> list[str]:
    """Return the fixed deployment settings shared by formal supplements."""
    return [
        "--adapt_lr", "0.00025",
        "--batch_size", str(args.batch_size),
        "--num_workers", str(args.num_workers),
        "--recovery_interval", "10",
        "--restore_strength", "0.5",
        "--restore_budget_mode", "drift_tiered",
        "--restore_drift_thresholds", "0.10", "0.20", "0.35",
        "--restore_drift_ratios", "0.0", "0.02", "0.05", "0.10",
        "--fslr_grad_clip_norm", "15.0",
        "--seed", str(args.seed),
        "--max_batches", str(args.max_batches),
    ]


# This is the retained primary FedDR configuration from ``main_test.py`` and
# the same CIFAR-100 FedACG main runs under ``test/outputs/fedacg_run1``.  It
# is kept here as an explicit launcher contract so component ablations cannot
# silently fall back to the older strict-mechanism settings above.
OFFICIAL_FEDDR_BATCH_SIZE = 128
OFFICIAL_FEDDR_LR = "0.001"


def official_protocol_flags(args) -> list[str]:
    """Return the public main-experiment FedDR protocol as CLI flags.

    Ablation rows may change the public TTA method, but they inherit this
    complete set of official controls.  Keeping the values explicit is
    intentional: argparse defaults for a non-``feddr`` method are not the
    primary FedDR settings.
    """
    return [
        "--adapt_lr", OFFICIAL_FEDDR_LR,
        "--batch_size", str(OFFICIAL_FEDDR_BATCH_SIZE),
        "--num_workers", str(args.num_workers),
        "--adapt_bn_scope", "excluding_layer4",
        "--feddr_bn_scope", "excluding_layer4",
        "--dlrs_region_scale_min", "0.6",
        "--dlrs_region_scale_max", "1.2",
        "--dlrs_reliability_alpha", "0.7",
        "--dlrs_sample_reliability_mode", "full",
        "--dlrs_normalize_by_weight",
        "--dlrs_weight_mode", "product",
        "--dlrs_hybrid_threshold", "0.2",
        "--dlrs_skip_update_tau_s", "0.05",
        "--dlrs_skip_update_tau_ess", "0.05",
        "--dlrs_c_max", "2.0",
        "--dlrs_consistency_weight", "0.5",
        "--dlrs_anchor_rectify_beta", "0.3",
        "--dlrs_view_consistency",
        "--dlrs_view_alpha", "0.8",
        "--dlrs_view_js_scale", "1.0",
        "--disable_fslr_reliability_gate",
        "--recovery_interval", "10",
        "--restore_strength", "0.7",
        "--restore_budget_mode", "drift_tiered",
        "--restore_drift_thresholds", "0.10", "0.20", "0.35",
        "--restore_drift_ratios", "0.0", "0.02", "0.05", "0.10",
        "--restore_reliability_threshold", "0.55",
        "--restore_reliability_ema_decay", "0.90",
        "--fslr_grad_clip_norm", "15.0",
        "--fslr_event_trigger",
        "--fslr_event_k", "3",
        "--fslr_event_tau_r", "0.40",
        "--fslr_event_tau_h", "1.50",
        "--fslr_event_tau_d", "0.20",
        "--fslr_event_lambda_0", "0.50",
        "--fslr_event_lambda_max", "0.90",
        "--fslr_event_tau_a", "0.50",
        "--fslr_event_k_rho", "2.0",
        "--fslr_event_rho_max", "0.20",
        "--fslr_event_cooldown", "20",
        "--fslr_continuous_risk",
        "--fslr_risk_rho_min", "0.0",
        "--fslr_risk_rho_max", "0.10",
        "--fslr_risk_gamma_ess", "0.50",
        "--fslr_risk_gamma_anchor", "0.50",
        "--fslr_risk_gamma_drift", "0.0",
        "--fslr_risk_tau_drift", "0.35",
        "--fslr_hierarchical_budget",
        "--fslr_layer_quota_cap", "0.35",
        "--fslr_selection_criterion", "fisher_drift",
        "--seed", str(args.seed),
        "--max_batches", str(args.max_batches),
    ]


def build_main_task(*, name: str, dataset: str, federated_method: str,
                    model_path: Path, anchor_key: str, statistics_path: Path,
                    tta_method: str, corruption_path: str,
                    clean_root: str, device: str, batch_size: int,
                    num_workers: int, seed: int, max_batches: int,
                    output: Path, log: Path,
                    extra_flags: Sequence[str] = (),
                    metadata: Mapping[str, Any] | None = None) -> Task:
    command = [
        sys.executable,
        str(ROOT / "main_test.py"),
        "--dataset", dataset,
        "--federated_method", federated_method,
        "--model_path", str(model_path),
        "--anchor_key", anchor_key,
        "--tta_method", tta_method,
        "--statistics_path", str(statistics_path),
        "--cifar_c_path", corruption_path,
        "--clean_data_root", clean_root,
        "--device", device,
        "--batch_size", str(batch_size),
        "--num_workers", str(num_workers),
        "--seed", str(seed),
        "--max_batches", str(max_batches),
        *extra_flags,
        "--output_path", str(output),
    ]
    return Task(name, command, output, log, dict(metadata or {}))


def build_source_task(*, name: str, dataset: str, model_path: Path,
                      anchor_key: str, corruption_path: str, clean_root: str,
                      device: str, batch_size: int, num_workers: int, seed: int,
                      max_batches: int, output: Path, log: Path,
                      metadata: Mapping[str, Any] | None = None) -> Task:
    command = [
        sys.executable,
        "-m", "test.source_only_test",
        "--dataset", dataset,
        "--model_path", str(model_path),
        "--anchor_key", anchor_key,
        "--cifar_c_path", corruption_path,
        "--clean_data_root", clean_root,
        "--device", device,
        "--batch_size", str(batch_size),
        "--num_workers", str(num_workers),
        "--seed", str(seed),
        "--max_batches", str(max_batches),
        "--output_path", str(output),
    ]
    return Task(name, command, output, log, dict(metadata or {}))


def _result_sections(payload: Mapping[str, Any]):
    domain = payload.get("domain_wise", {})
    continual = payload.get("continual", {})
    return domain, continual


def valid_result(path: Path, *, dataset: str | None = None,
                 backbone: str | None = None) -> bool:
    """Accept schema 1 Source output and schema 9 adaptation output only when complete."""
    if not path.is_file() or path.stat().st_size < 200:
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    if payload.get("schema_version") not in (1, 9):
        return False
    if dataset is not None and payload.get("dataset") != dataset:
        return False
    if backbone is not None and payload.get("backbone") != backbone:
        return False
    max_batches = payload.get("protocol", {}).get("max_batches")
    if max_batches is not None and int(max_batches) > 0:
        return False
    domain, continual = _result_sections(payload)
    if len(domain.get("per_corruption", [])) != len(CORRUPTIONS):
        return False
    if len(continual.get("per_corruption", [])) != len(CORRUPTIONS) * CONTINUAL_CYCLES:
        return False
    if [item.get("corruption") for item in domain["per_corruption"]] != CORRUPTIONS:
        return False
    expected_stream = CORRUPTIONS * CONTINUAL_CYCLES
    if [item.get("corruption") for item in continual["per_corruption"]] != expected_stream:
        return False
    cycle_summaries = continual.get("summary", {}).get("cycle_summaries", [])
    if len(cycle_summaries) != CONTINUAL_CYCLES:
        return False
    return "post_stream" in continual and "clean_retention" in payload


def preflight(entries: Iterable[Mapping[str, Any]], output: Path,
              *, allow_missing: bool = False) -> list[dict[str, Any]]:
    """Record path existence before launching any model process."""
    records = []
    missing_required = []
    for entry in entries:
        path = rooted(entry["path"])
        required = bool(entry.get("required", True))
        record = {
            "role": entry.get("role", "input"),
            "path": str(path),
            "exists": path.is_file() if entry.get("kind", "file") == "file" else path.is_dir(),
            "bytes": path.stat().st_size if path.is_file() else None,
            "required": required,
        }
        records.append(record)
        if required and not record["exists"]:
            missing_required.append(str(path))
    payload = {"schema_version": 1, "root": str(ROOT), "inputs": records}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if missing_required and not allow_missing:
        formatted = "\n".join(f"  - {item}" for item in missing_required)
        raise FileNotFoundError(
            "Preflight failed; missing required inputs:\n" + formatted
            + "\nPass --allow_missing only to generate a partial manifest."
        )
    return records


def run_tasks(tasks: Sequence[Task], *, dry_run: bool = False,
              force_rerun: bool = False, allow_missing: bool = False) -> list[str]:
    """Run tasks sequentially, preserving one log per task and resumability."""
    failures: list[str] = []
    for index, task in enumerate(tasks, start=1):
        if dry_run:
            print(shlex.join(task.command))
            continue
        missing_inputs = [
            str(item) for item in task.metadata.get("required_inputs", [])
            if not Path(item).is_file()
        ]
        if missing_inputs:
            message = "%s missing inputs: %s" % (task.name, ", ".join(missing_inputs))
            if allow_missing:
                LOGGER.warning("[%d/%d] skipping %s", index, len(tasks), message)
            else:
                LOGGER.error("[%d/%d] %s", index, len(tasks), message)
                failures.append(task.name)
            continue
        if not force_rerun and valid_result(
            task.output,
            dataset=task.metadata.get("dataset"),
            backbone=task.metadata.get("backbone"),
        ):
            LOGGER.info("[%d/%d] validated result exists: %s", index, len(tasks), task.output)
            continue
        task.output.parent.mkdir(parents=True, exist_ok=True)
        task.log.parent.mkdir(parents=True, exist_ok=True)
        LOGGER.info("[%d/%d] running %s", index, len(tasks), task.name)
        with task.log.open("w", encoding="utf-8") as handle:
            result = subprocess.run(
                task.command,
                cwd=str(ROOT),
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        if result.returncode:
            LOGGER.error("%s failed with exit code %d; see %s", task.name,
                         result.returncode, task.log)
            failures.append(task.name)
            continue
        if not valid_result(task.output,
                           dataset=task.metadata.get("dataset"),
                           backbone=task.metadata.get("backbone")):
            LOGGER.error("%s produced an incomplete result: %s", task.name, task.output)
            failures.append(task.name)
    return failures


def _fraction(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def result_metrics(path: Path) -> dict[str, Any]:
    """Extract paper-facing metrics without assuming one historical JSON layout."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    domain, continual = _result_sections(payload)
    domain_summary = domain.get("summary", domain)
    continual_summary = continual.get("summary", continual)
    cycles = continual_summary.get("cycle_summaries", [])
    c1 = _fraction(cycles[0].get("mean_accuracy")) if len(cycles) > 0 else None
    c3 = _fraction(cycles[2].get("mean_accuracy")) if len(cycles) > 2 else None
    post = continual.get("post_stream", {})
    revisit = post.get("mean_revisit_accuracy")
    if revisit is None:
        revisit = post.get("revisit_mean_accuracy")
    clean = post.get("clean", {}).get("accuracy")
    if clean is None:
        retention = payload.get("clean_retention", {})
        clean = retention.get("final_continual_clean_accuracy")
        if clean is None:
            clean = retention.get("final_clean", {}).get("accuracy")
    values = {
        "domain": _fraction(domain_summary.get("mean_accuracy")),
        "cycle1": c1,
        "cycle3": c3,
        "delta3_minus_1": None if c1 is None or c3 is None else c3 - c1,
        "revisit": _fraction(revisit),
        "clean": _fraction(clean),
    }
    for key, value in list(values.items()):
        values[key] = None if value is None else round(100.0 * value, 6)
    return values


def write_metrics_summary(rows: Sequence[Mapping[str, Any]], output: Path) -> None:
    """Write a JSON and a spreadsheet-friendly CSV beside the requested output."""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(list(rows), indent=2, ensure_ascii=False) + "\n",
                      encoding="utf-8")
    csv_path = output.with_suffix(".csv")
    columns = ["label", "source", "status", "domain", "cycle1", "cycle3",
               "delta3_minus_1", "revisit", "clean"]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def summarize_tasks(tasks: Sequence[Task], *, output: Path,
                   aliases: Mapping[str, str] | None = None) -> list[dict[str, Any]]:
    rows = []
    aliases = aliases or {}
    for task in tasks:
        if not task.output.is_file():
            rows.append({"label": aliases.get(task.name, task.name),
                         "source": task.name, "status": "missing"})
            continue
        try:
            values = result_metrics(task.output)
            rows.append({"label": aliases.get(task.name, task.name),
                         "source": task.name, "status": "ok", **values})
        except (OSError, ValueError, TypeError, KeyError) as exc:
            rows.append({"label": aliases.get(task.name, task.name),
                         "source": task.name, "status": f"invalid: {exc}"})
    write_metrics_summary(rows, output)
    return rows


def extract_45_block_curve(result_path: Path, output_json: Path,
                           output_csv: Path, plot_path: Path | None = None) -> list[dict[str, Any]]:
    """Export the 15-corruption x 3-cycle continual stream as block rows."""
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    stream = payload.get("continual", {}).get("per_corruption", [])
    if len(stream) != len(CORRUPTIONS) * CONTINUAL_CYCLES:
        raise ValueError(
            f"expected 45 continual blocks, found {len(stream)} in {result_path}"
        )
    rows = []
    for index, item in enumerate(stream):
        cycle = int(item.get("cycle", index // len(CORRUPTIONS) + 1))
        within_cycle = index % len(CORRUPTIONS)
        corruption = item.get("corruption", CORRUPTIONS[within_cycle])
        accuracy = _fraction(item.get("accuracy"))
        if accuracy is None:
            raise ValueError(f"block {index + 1} has no accuracy")
        rows.append({
            "block": index + 1,
            "cycle": cycle,
            "within_cycle": within_cycle + 1,
            "corruption": corruption,
            "accuracy": round(100.0 * accuracy, 6),
        })
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n",
                            encoding="utf-8")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    if plot_path is not None:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            LOGGER.warning("matplotlib is unavailable; curve CSV/JSON were still written")
        else:
            figure, axis = plt.subplots(figsize=(12, 4.5))
            axis.plot([row["block"] for row in rows], [row["accuracy"] for row in rows],
                      marker=".", linewidth=1.4, markersize=3)
            for boundary in (15.5, 30.5):
                axis.axvline(boundary, color="0.65", linestyle="--", linewidth=0.8)
            axis.set(xlabel="Continual block (15 corruptions × 3 cycles)",
                     ylabel="Accuracy (%)", xlim=(1, len(rows)),
                     title="FedDR 45-block continual accuracy")
            axis.grid(alpha=0.2)
            figure.tight_layout()
            plot_path.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(plot_path, dpi=180)
            plt.close(figure)
    return rows


def raise_on_failures(failures: Sequence[str]) -> None:
    if failures:
        raise RuntimeError("The following supplement tasks failed: "
                           + ", ".join(failures))
