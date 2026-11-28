#!/usr/bin/env python3
"""Unified launcher and table summarizer for all remaining FedRAS paper TODO experiments.

Covers:
  - components : Tables 7, 8, 13, and 14 (DSR only + FedRAS across run1..run3 with diagnostics)
  - random     : Table 9 matched random coordinate selection control (run1..run3)
  - fixed      : Table 9 fixed restoration ratios rho in {0.02, 0.05, 0.10} (run1..run3)
  - sensitivity: Table 12 one-factor-at-a-time sensitivity on run1 (R, tau_D, rho_max)
  - cost       : Table 10 stream-level cost on run1 (warm-up + 3 repeats; optional baselines)
  - all        : All stages above
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import statistics
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "revision_20261018" / "run_case.py"
DEFAULT_OUTPUT_ROOT = ROOT / "outputs" / "paper_todo_experiments"

CORRUPTIONS = (
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
)

CHECKPOINT_RELPATHS: Dict[int, Path] = {
    1: Path("train/results/cifar100_100clients_alpha0.3_equal/fedacg_resnet18/cifar100_resnet18_run1/model_round499.pt"),
    2: Path("train/results/cifar100_100clients_alpha0.3_equal/fedacg_resnet18/run2/model_round499.pt"),
    3: Path("train/results/cifar100_100clients_alpha0.3_equal/fedacg_resnet18/run3/model_round499.pt"),
}

STATISTICS_RELPATHS: Dict[int, Path] = {
    1: Path("tests/outputs/fedacg_run1_statistics.pt"),
    2: Path("tests/outputs/fedacg_run2_statistics.pt"),
    3: Path("tests/outputs/fedacg_run3_statistics.pt"),
}


def resolve_assets(args: argparse.Namespace) -> Tuple[Dict[int, Path], Dict[int, Path]]:
    asset_root = Path(args.asset_root)
    checkpoints = {
        idx: Path(getattr(args, f"ckpt{idx}") or (asset_root / rel))
        for idx, rel in CHECKPOINT_RELPATHS.items()
    }
    statistics_map = {
        idx: Path(getattr(args, f"stats{idx}") or (asset_root / rel))
        for idx, rel in STATISTICS_RELPATHS.items()
    }
    return checkpoints, statistics_map


@dataclass(frozen=True)
class TrialSpec:
    name: str
    run_index: int
    extra_args: Tuple[str, ...]
    needs_cuda: bool = False
    diagnostics_relpath: Optional[str] = None


def _common_args(
    *,
    device: str,
    data_root: Path,
    clean_root: Path,
    num_workers: int,
    model_path: Path,
    stats_path: Path,
    output_path: Path,
) -> List[str]:
    return [
        "--dataset", "cifar100",
        "--federated_method", "fedacg",
        "--anchor_key", "lookahead_anchor",
        "--tta_method", "dlrs_fslr",
        "--revision-ranking", "drift_only",
        "--review_continual_only",
        "--device", device,
        "--cifar_c_path", str(data_root),
        "--clean_data_root", str(clean_root),
        "--batch_size", "128",
        "--seed", "0",
        "--num_workers", str(num_workers),
        "--max_batches", "-1",
        "--adapt_lr", "0.001",
        "--dlrs_sample_reliability_mode", "entropy",
        "--dlrs_weight_mode", "sample_only",
        "--dlrs_hybrid_threshold", "0",
        "--dlrs_anchor_rectify_beta", "0.3",
        "--disable_dlrs_view_consistency",
        "--fslr_grad_clip_norm", "15",
        "--disable_fslr_reliability_gate",
        "--fslr_hierarchical_budget",
        "--fslr_layer_quota_cap", "0.35",
        "--fslr_selection_criterion", "drift_only",
        "--model_path", str(model_path),
        "--statistics_path", str(stats_path),
        "--output_path", str(output_path),
    ]


def _adaptive_args() -> Tuple[str, ...]:
    return (
        "--fslr_continuous_risk",
        "--fslr_risk_gamma_anchor", "0.5",
        "--fslr_risk_rho_min", "0",
        "--fslr_risk_rho_max", "0.10",
        "--fslr_risk_tau_drift", "0.35",
        "--recovery_interval", "10",
        "--restore_strength", "0.7",
        "--fslr_event_trigger",
    )


def _baseline_cost_args(
    *,
    method: str,
    device: str,
    data_root: Path,
    clean_root: Path,
    num_workers: int,
    model_path: Path,
    stats_path: Path,
    output_path: Path,
) -> List[str]:
    return [
        "--dataset", "cifar100",
        "--federated_method", "fedacg",
        "--anchor_key", "lookahead_anchor",
        "--tta_method", method,
        "--revision-track", "reasonable",
        "--revision-warmup",
        "--review_continual_only",
        "--device", device,
        "--cifar_c_path", str(data_root),
        "--clean_data_root", str(clean_root),
        "--batch_size", "128",
        "--seed", "0",
        "--num_workers", str(num_workers),
        "--max_batches", "-1",
        "--model_path", str(model_path),
        "--statistics_path", str(stats_path),
        "--output_path", str(output_path),
    ]


def build_stage_trials(
    stage: str,
    *,
    output_root: Path,
    runs: Sequence[int],
    selection_seed: int,
    include_cost_baselines: bool,
    record_schedule_in_components: bool,
) -> List[TrialSpec]:
    trials: List[TrialSpec] = []
    adaptive = _adaptive_args()

    if stage in ("components", "all"):
        for idx in runs:
            dsr_name = f"components/run{idx}/dsr_only"
            fedras_name = f"components/run{idx}/fedras"
            trials.append(
                TrialSpec(
                    name=dsr_name,
                    run_index=idx,
                    extra_args=(
                        *adaptive,
                        "--dlrs_ablation_variant", "uniform_recap",
                        "--anchor_disagreement_diagnostic",
                        "--review_diagnostics_path", str(output_root / dsr_name / "diagnostics.jsonl"),
                        "--review_run_id", dsr_name,
                        "--review_beta_a", "0",
                    ),
                    diagnostics_relpath="diagnostics.jsonl",
                )
            )
            fedras_extra: List[str] = [
                *adaptive,
                "--dlrs_ablation_variant", "anchor_only",
                "--review_diagnostics_path", str(output_root / fedras_name / "diagnostics.jsonl"),
                "--review_run_id", fedras_name,
                "--review_beta_a", "0.3",
            ]
            if record_schedule_in_components:
                fedras_extra.extend(["--revision-schedule-record", str(output_root / fedras_name / "schedule.pt")])
            trials.append(
                TrialSpec(
                    name=fedras_name,
                    run_index=idx,
                    extra_args=tuple(fedras_extra),
                    diagnostics_relpath="diagnostics.jsonl",
                )
            )

    if stage in ("random", "all"):
        for idx in runs:
            comp_schedule = output_root / f"components/run{idx}/fedras/schedule.pt"
            ref_name = f"random/run{idx}/reference"
            ref_schedule = output_root / ref_name / "schedule.pt"
            use_comp = (stage == "all" and record_schedule_in_components) or comp_schedule.is_file()
            schedule_path = comp_schedule if use_comp else ref_schedule
            if not use_comp:
                trials.append(
                    TrialSpec(
                        name=ref_name,
                        run_index=idx,
                        extra_args=(
                            *adaptive,
                            "--dlrs_ablation_variant", "anchor_only",
                            "--revision-schedule-record", str(ref_schedule),
                        ),
                    )
                )
            trials.append(
                TrialSpec(
                    name=f"random/run{idx}/matched",
                    run_index=idx,
                    extra_args=(
                        *adaptive,
                        "--dlrs_ablation_variant", "anchor_only",
                        "--revision-ranking", "random",
                        "--revision-selection-seed", str(selection_seed),
                        "--revision-schedule-replay", str(schedule_path),
                    ),
                )
            )

    if stage in ("fixed", "all"):
        for idx in runs:
            for rho in ("0.02", "0.05", "0.10"):
                trials.append(
                    TrialSpec(
                        name=f"fixed/run{idx}/rho_{rho}",
                        run_index=idx,
                        extra_args=(
                            "--dlrs_ablation_variant", "anchor_only",
                            "--disable_fslr_event_trigger",
                            "--disable_fslr_continuous_risk",
                            "--recovery_interval", "10",
                            "--restore_budget_mode", "fixed",
                            "--sparse_ratio", rho,
                            "--restore_strength", "0.7",
                        ),
                    )
                )

    if stage in ("sensitivity", "all"):
        for interval in ("5", "20"):
            trials.append(
                TrialSpec(
                    name=f"sensitivity/R_{interval}",
                    run_index=1,
                    extra_args=(
                        "--fslr_continuous_risk",
                        "--fslr_risk_gamma_anchor", "0.5",
                        "--fslr_risk_rho_min", "0",
                        "--fslr_risk_rho_max", "0.10",
                        "--fslr_risk_tau_drift", "0.35",
                        "--recovery_interval", interval,
                        "--restore_strength", "0.7",
                        "--fslr_event_trigger",
                        "--dlrs_ablation_variant", "anchor_only",
                    ),
                )
            )
        for tau in ("0.25", "0.50"):
            trials.append(
                TrialSpec(
                    name=f"sensitivity/tauD_{tau}",
                    run_index=1,
                    extra_args=(
                        "--fslr_continuous_risk",
                        "--fslr_risk_gamma_anchor", "0.5",
                        "--fslr_risk_rho_min", "0",
                        "--fslr_risk_rho_max", "0.10",
                        "--fslr_risk_tau_drift", tau,
                        "--recovery_interval", "10",
                        "--restore_strength", "0.7",
                        "--fslr_event_trigger",
                        "--dlrs_ablation_variant", "anchor_only",
                    ),
                )
            )
        for rho_max in ("0.05", "0.20"):
            trials.append(
                TrialSpec(
                    name=f"sensitivity/rhoMax_{rho_max}",
                    run_index=1,
                    extra_args=(
                        "--fslr_continuous_risk",
                        "--fslr_risk_gamma_anchor", "0.5",
                        "--fslr_risk_rho_min", "0",
                        "--fslr_risk_rho_max", rho_max,
                        "--fslr_risk_tau_drift", "0.35",
                        "--recovery_interval", "10",
                        "--restore_strength", "0.7",
                        "--fslr_event_trigger",
                        "--dlrs_ablation_variant", "anchor_only",
                    ),
                )
            )

    if stage in ("cost", "all"):
        for rep in ("repeat1", "repeat2", "repeat3"):
            trials.append(
                TrialSpec(
                    name=f"cost/fedras/{rep}",
                    run_index=1,
                    extra_args=(
                        *adaptive,
                        "--dlrs_ablation_variant", "anchor_only",
                        "--revision-warmup",
                    ),
                    needs_cuda=True,
                )
            )
        if include_cost_baselines:
            for method in ("asr", "recap", "tent"):
                for rep in ("repeat1", "repeat2", "repeat3"):
                    trials.append(
                        TrialSpec(
                            name=f"cost/{method}/{rep}",
                            run_index=1,
                            extra_args=("__BASELINE_COST__", method),
                            needs_cuda=True,
                        )
                    )

    return trials


def validate_inputs(
    trials: Sequence[TrialSpec],
    *,
    data_root: Path,
    clean_root: Path,
    checkpoints: Dict[int, Path],
    statistics_map: Dict[int, Path],
    device: str,
) -> None:
    missing: List[Path] = []
    if not RUNNER.is_file():
        missing.append(RUNNER)
    for split in ("train", "test", "meta"):
        p = clean_root / "cifar-100-python" / split
        if not p.is_file():
            missing.append(p)
    labels = data_root / "labels.npy"
    if not labels.is_file():
        missing.append(labels)
    for corr in CORRUPTIONS:
        p = data_root / f"{corr}.npy"
        if not p.is_file():
            missing.append(p)
    for idx in sorted({t.run_index for t in trials}):
        for asset in (checkpoints[idx], statistics_map[idx]):
            if not asset.is_file():
                missing.append(asset)
    if missing:
        raise FileNotFoundError("Missing required input files:\n" + "\n".join(f"  - {m}" for m in missing))

    if any(t.needs_cuda for t in trials):
        if not (device == "cuda" or device.startswith("cuda:")):
            raise ValueError(f"Cost measurements require a CUDA device, got device={device}")


def _build_command(
    spec: TrialSpec,
    *,
    python_bin: str,
    device: str,
    data_root: Path,
    clean_root: Path,
    checkpoints: Dict[int, Path],
    statistics_map: Dict[int, Path],
    num_workers: int,
    output_root: Path,
) -> List[str]:
    trial_dir = output_root / spec.name
    result_path = trial_dir / "result.json"
    ckpt = checkpoints[spec.run_index]
    stats = statistics_map[spec.run_index]
    if len(spec.extra_args) == 2 and spec.extra_args[0] == "__BASELINE_COST__":
        method = spec.extra_args[1]
        args = _baseline_cost_args(
            method=method,
            device=device,
            data_root=data_root,
            clean_root=clean_root,
            num_workers=num_workers,
            model_path=ckpt,
            stats_path=stats,
            output_path=result_path,
        )
    else:
        args = _common_args(
            device=device,
            data_root=data_root,
            clean_root=clean_root,
            num_workers=num_workers,
            model_path=ckpt,
            stats_path=stats,
            output_path=result_path,
        )
        args.extend(spec.extra_args)
    return [python_bin, "-u", str(RUNNER), *args]


def run_trial(
    spec: TrialSpec,
    *,
    python_bin: str,
    device: str,
    data_root: Path,
    clean_root: Path,
    checkpoints: Dict[int, Path],
    statistics_map: Dict[int, Path],
    num_workers: int,
    output_root: Path,
    dry_run: bool,
    force_rerun: bool,
) -> None:
    cmd = _build_command(
        spec,
        python_bin=python_bin,
        device=device,
        data_root=data_root,
        clean_root=clean_root,
        checkpoints=checkpoints,
        statistics_map=statistics_map,
        num_workers=num_workers,
        output_root=output_root,
    )
    if dry_run:
        print(subprocess.list2cmdline(cmd) if os.name == "nt" else shlex.join(cmd))
        return

    trial_dir = output_root / spec.name
    complete_marker = trial_dir / "complete.json"
    result_path = trial_dir / "result.json"
    signature = {"argv": cmd[2:]}

    if not force_rerun and complete_marker.is_file() and result_path.is_file() and result_path.stat().st_size > 0:
        recorded = json.loads(complete_marker.read_text(encoding="utf-8"))
        if recorded == signature:
            print(f"[skip] {spec.name} (already complete)")
            return
        raise RuntimeError(
            f"Completed trial {spec.name} has different arguments; use --force-rerun or another --output-root."
        )

    if trial_dir.exists():
        archive_dir = output_root / "_incomplete_archive" / f"{spec.name.replace('/', '__')}_{int(time.time())}"
        archive_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(trial_dir), str(archive_dir))
        print(f"[archive] moved incomplete {spec.name} -> {archive_dir}")

    trial_dir.mkdir(parents=True, exist_ok=True)
    log_path = trial_dir / "run.log"
    env = os.environ.copy()
    env.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    started = time.perf_counter()
    print(f"[{time.strftime('%H:%M:%S')}] Starting {spec.name}")
    with log_path.open("w", encoding="utf-8") as log_file:
        proc = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            log_file.write(line)
        rc = proc.wait()

    elapsed = time.perf_counter() - started
    if rc != 0:
        raise RuntimeError(f"Trial {spec.name} failed with exit code {rc} (see {log_path})")
    if not result_path.is_file() or result_path.stat().st_size == 0:
        raise RuntimeError(f"Trial {spec.name} finished without writing {result_path}")

    complete_marker.write_text(json.dumps(signature, indent=2), encoding="utf-8")
    print(f"[{time.strftime('%H:%M:%S')}] Finished {spec.name} in {elapsed:.1f}s")


# ---------------------------------------------------------------------------
# Summary extraction for Tables 7, 8, 9, 10, 12, 13, 14
# ---------------------------------------------------------------------------

def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _mean_sd(values: Sequence[float]) -> Tuple[float, Optional[float]]:
    if not values:
        raise ValueError("empty values")
    mean_val = statistics.mean(values)
    sd_val = statistics.stdev(values) if len(values) >= 2 else None
    return mean_val, sd_val


def _fmt_mean_sd(values: Sequence[Optional[float]], digits: int = 2) -> str:
    clean = [v for v in values if v is not None]
    if not clean:
        return "[TODO]"
    m, sd = _mean_sd(clean)
    if sd is None:
        return f"{m:.{digits}f} (n={len(clean)})"
    return f"{m:.{digits}f} ± {sd:.{digits}f}"


def _extract_result_metrics(payload: Dict[str, Any]) -> Dict[str, Any]:
    cont = payload["continual"]
    cs = cont["summary"]["cycle_summaries"]
    return {
        "c1": 100.0 * cs[0]["mean_accuracy"],
        "c2": 100.0 * cs[1]["mean_accuracy"],
        "c3": 100.0 * cs[2]["mean_accuracy"],
        "clean": 100.0 * payload["clean_retention"]["final_clean"]["accuracy"],
        "revisit": 100.0 * cont["post_stream"]["mean_revisit_accuracy"],
        "restored": int(cont["restoration"]["restored_assignments_total"]),
        "drift_c1": cs[0].get("final_bn_affine_drift", {}).get("total"),
        "drift_c2": cs[1].get("final_bn_affine_drift", {}).get("total"),
        "drift_c3": cs[2].get("final_bn_affine_drift", {}).get("total"),
        "wall_time": cont["resource"].get("wall_time_seconds"),
        "adapt_ms": cont["resource"].get("mean_adaptation_time_ms"),
        "peak_mem_mib": cont["resource"].get("peak_cuda_memory_mb"),
    }


def _aggregate_diagnostics(jsonl_paths: Sequence[Path]) -> Optional[Dict[str, Any]]:
    existing = [p for p in jsonl_paths if p.is_file()]
    if not existing:
        return None

    total_n = 0
    total_e = 0
    total_c = 0
    total_e_dis = 0
    total_c_dis = 0
    total_w = 0.0
    total_err_w = 0.0
    valid_batches = 0
    de_gt_dc_batches = 0
    cat_cw_rc = 0
    cat_cc_rw = 0
    cat_both_wrong = 0
    cat_same = 0

    for path in existing:
        with path.open("r", encoding="utf-8") as stream:
            for raw in stream:
                raw = raw.strip()
                if not raw:
                    continue
                row = json.loads(raw)
                n = int(row["n"])
                n_e = int(row["n_E"])
                n_c = int(row["n_C"])
                total_n += n
                total_e += n_e
                total_c += n_c
                total_e_dis += int(row["n_E_disagree"])
                total_c_dis += int(row["n_C_disagree"])
                total_w += float(row["weight_sum"])
                total_err_w += float(row["error_weight_sum"])
                cat_cw_rc += int(row["current_wrong_anchor_correct"])
                cat_cc_rw += int(row["current_correct_anchor_wrong"])
                cat_both_wrong += int(row["both_wrong_different_class"])
                cat_same += int(row["both_same_class"])
                if n_e > 0 and n_c > 0:
                    valid_batches += 1
                    if float(row["d_E"]) > float(row["d_C"]):
                        de_gt_dc_batches += 1

    if total_n == 0:
        return None
    eps = 100.0 * total_e / total_n
    eps_rga = 100.0 * total_err_w / total_w if total_w > 0 else 0.0
    return {
        "files_used": len(existing),
        "d_E": 100.0 * total_e_dis / max(1, total_e),
        "d_C": 100.0 * total_c_dis / max(1, total_c),
        "de_gt_dc_batches": de_gt_dc_batches,
        "valid_batches": valid_batches,
        "epsilon": eps,
        "epsilon_rga": eps_rga,
        "reduction_pp": eps - eps_rga,
        "cat_cw_rc": cat_cw_rc,
        "cat_cc_rw": cat_cc_rw,
        "cat_both_wrong": cat_both_wrong,
        "cat_same": cat_same,
        "total_samples": total_n,
    }


def summarize_tables(output_root: Path, runs: Sequence[int]) -> Path:
    lines: List[str] = [
        "# FedRAS Paper Missing Tables Summary",
        "",
        f"Source directory: `{output_root}`",
        "",
    ]

    # Table 7
    lines.extend([
        "## Table 7: Component Ablation (CIFAR-100-C, 3 cycles)",
        "",
        "| Method | Cycle 1 | Cycle 2 | Cycle 3 | Clean | Revisit |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    comp_metrics: Dict[str, Dict[int, Dict[str, Any]]] = {"dsr_only": {}, "fedras": {}}
    for arm, label in (("dsr_only", "DSR only"), ("fedras", "FedRAS")):
        for idx in runs:
            payload = _load_json(output_root / f"components/run{idx}/{arm}/result.json")
            if payload is not None:
                comp_metrics[arm][idx] = _extract_result_metrics(payload)
        rows = list(comp_metrics[arm].values())
        lines.append(
            f"| {label} | {_fmt_mean_sd([r['c1'] for r in rows])} | "
            f"{_fmt_mean_sd([r['c2'] for r in rows])} | "
            f"{_fmt_mean_sd([r['c3'] for r in rows])} | "
            f"{_fmt_mean_sd([r['clean'] for r in rows])} | "
            f"{_fmt_mean_sd([r['revisit'] for r in rows])} |"
        )
    lines.append("")

    # Table 8 & Table 13
    lines.extend([
        "## Table 8: RGA Diagnostics (Pooled)",
        "",
        "| Method | d_E (%) | d_C (%) | Batches (d_E > d_C) | epsilon (%) | epsilon_RGA (%) | Reduction (pp) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    diag_summaries: Dict[str, Optional[Dict[str, Any]]] = {}
    for arm, label in (("dsr_only", "DSR only"), ("fedras", "FedRAS")):
        diag_paths = [output_root / f"components/run{idx}/{arm}/diagnostics.jsonl" for idx in runs]
        d = _aggregate_diagnostics(diag_paths)
        diag_summaries[arm] = d
        if d is None:
            lines.append(f"| {label} | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] |")
        else:
            note = "" if d["files_used"] == 3 else f" (n={d['files_used']})"
            lines.append(
                f"| {label}{note} | {d['d_E']:.2f} | {d['d_C']:.2f} | "
                f"{d['de_gt_dc_batches']:,}/{d['valid_batches']:,} | "
                f"{d['epsilon']:.2f} | {d['epsilon_rga']:.2f} | {d['reduction_pp']:.2f} |"
            )
    lines.append("")

    lines.extend([
        "## Table 13 (Appendix A): Prediction Categories (Pooled)",
        "",
        "| Method | Current wrong / Ref. correct | Current correct / Ref. wrong | Both wrong (diff. class) | Same class | Total |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for arm, label in (("dsr_only", "DSR only"), ("fedras", "FedRAS")):
        d = diag_summaries[arm]
        if d is None:
            lines.append(f"| {label} | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] |")
        else:
            lines.append(
                f"| {label} | {d['cat_cw_rc']:,} | {d['cat_cc_rw']:,} | "
                f"{d['cat_both_wrong']:,} | {d['cat_same']:,} | {d['total_samples']:,} |"
            )
    lines.append("")

    # Table 14 (Appendix B, run1)
    lines.extend([
        "## Table 14 (Appendix B): Single-Checkpoint DSR Drift Diagnostics (run1)",
        "",
        "| Method | Drift C1 | Drift C2 | Drift C3 | Acc. C3 | Clean |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for arm, label in (("dsr_only", "DSR only"), ("fedras", "FedRAS")):
        m = comp_metrics[arm].get(1)
        if m is None:
            lines.append(f"| {label} | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] |")
        else:
            lines.append(
                f"| {label} | {m['drift_c1']:.2f} | {m['drift_c2']:.2f} | "
                f"{m['drift_c3']:.2f} | {m['c3']:.2f} | {m['clean']:.2f} |"
            )
    lines.append("")

    # Table 9
    lines.extend([
        "## Table 9: DSR Selection and Budget Controls",
        "",
        "| Setting | Cycle 3 | Clean | Revisit | Restored assignments |",
        "|---|---:|---:|---:|---:|",
    ])
    fedras_rows = list(comp_metrics["fedras"].values())
    lines.append(
        f"| FedRAS (adaptive) | {_fmt_mean_sd([r['c3'] for r in fedras_rows])} | "
        f"{_fmt_mean_sd([r['clean'] for r in fedras_rows])} | "
        f"{_fmt_mean_sd([r['revisit'] for r in fedras_rows])} | "
        f"{_fmt_mean_sd([float(r['restored']) for r in fedras_rows], digits=0)} |"
    )
    control_specs = [
        ("Random selection", [output_root / f"random/run{idx}/matched/result.json" for idx in runs]),
        ("Fixed rho = 0.02", [output_root / f"fixed/run{idx}/rho_0.02/result.json" for idx in runs]),
        ("Fixed rho = 0.05", [output_root / f"fixed/run{idx}/rho_0.05/result.json" for idx in runs]),
        ("Fixed rho = 0.10", [output_root / f"fixed/run{idx}/rho_0.10/result.json" for idx in runs]),
    ]
    for label, paths in control_specs:
        rows = []
        for p in paths:
            payload = _load_json(p)
            if payload is not None:
                rows.append(_extract_result_metrics(payload))
        lines.append(
            f"| {label} | {_fmt_mean_sd([r['c3'] for r in rows])} | "
            f"{_fmt_mean_sd([r['clean'] for r in rows])} | "
            f"{_fmt_mean_sd([r['revisit'] for r in rows])} | "
            f"{_fmt_mean_sd([float(r['restored']) for r in rows], digits=0)} |"
        )
    lines.append("")

    # Table 10
    lines.extend([
        "## Table 10: Stream-Level Computational Cost (run1, 3 repeats after warm-up)",
        "",
        "| Method | Wall time (s) | Adapt (ms/batch) | Peak Mem. (MiB) | Cycle 3 | Clean | Revisit |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for method, label in (("fedras", "FedRAS"), ("asr", "ASR"), ("recap", "ReCAP"), ("tent", "Tent")):
        rows = []
        for rep in ("repeat1", "repeat2", "repeat3"):
            payload = _load_json(output_root / f"cost/{method}/{rep}/result.json")
            if payload is not None:
                rows.append(_extract_result_metrics(payload))
        if not rows and method != "fedras":
            continue
        lines.append(
            f"| {label} | {_fmt_mean_sd([r['wall_time'] for r in rows])} | "
            f"{_fmt_mean_sd([r['adapt_ms'] for r in rows])} | "
            f"{_fmt_mean_sd([r['peak_mem_mib'] for r in rows], digits=1)} | "
            f"{_fmt_mean_sd([r['c3'] for r in rows])} | "
            f"{_fmt_mean_sd([r['clean'] for r in rows])} | "
            f"{_fmt_mean_sd([r['revisit'] for r in rows])} |"
        )
    lines.append("")

    # Table 12
    lines.extend([
        "## Table 12: Hyperparameter Sensitivity (run1)",
        "",
        "| Param. | Value | C1 (%) | C3 (%) | Rev. (%) | Clean (%) |",
        "|---|---|---:|---:|---:|---:|",
    ])
    sens_rows = [
        ("R", "5", "sensitivity/R_5"),
        ("R", "20", "sensitivity/R_20"),
        ("tau_D", "0.25", "sensitivity/tauD_0.25"),
        ("tau_D", "0.50", "sensitivity/tauD_0.50"),
        ("rho_max", "0.05", "sensitivity/rhoMax_0.05"),
        ("rho_max", "0.20", "sensitivity/rhoMax_0.20"),
    ]
    for param_name, val, rel in sens_rows:
        payload = _load_json(output_root / rel / "result.json")
        if payload is None:
            lines.append(f"| {param_name} | {val} | [TODO] | [TODO] | [TODO] | [TODO] |")
        else:
            m = _extract_result_metrics(payload)
            lines.append(
                f"| {param_name} | {val} | {m['c1']:.2f} | {m['c3']:.2f} | "
                f"{m['revisit']:.2f} | {m['clean']:.2f} |"
            )

    summary_path = output_root / "SUMMARY_TABLES.md"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Summary written to: {summary_path}")
    return summary_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run and summarize missing FedRAS paper TODO experiments.")
    parser.add_argument(
        "--stage",
        choices=["components", "random", "fixed", "sensitivity", "cost", "all"],
        default="all",
        help="Experiment group to run.",
    )
    parser.add_argument("--runs", type=int, nargs="+", default=[1, 2, 3], choices=[1, 2, 3])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data" / "CIFAR-100-C")
    parser.add_argument("--clean-root", type=Path, default=ROOT / "data" / "cifar100")
    parser.add_argument("--asset-root", type=Path, default=ROOT,
                        help="Root directory holding train/results/... and tests/outputs/... (useful on Colab Drive).")
    parser.add_argument("--ckpt1", type=Path, default=None)
    parser.add_argument("--ckpt2", type=Path, default=None)
    parser.add_argument("--ckpt3", type=Path, default=None)
    parser.add_argument("--stats1", type=Path, default=None)
    parser.add_argument("--stats2", type=Path, default=None)
    parser.add_argument("--stats3", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--selection-seed", type=int, default=2026)
    parser.add_argument("--include-cost-baselines", action="store_true",
                        help="Also re-run ASR/ReCAP/Tent in the cost stage on the same hardware.")
    parser.add_argument("--no-record-schedule-in-components", action="store_true",
                        help="Do not record schedule.pt during components/fedras.")
    parser.add_argument("--dry-run", action="store_true", help="Print resolved commands without executing.")
    parser.add_argument("--force-rerun", action="store_true", help="Re-run even if complete.json matches.")
    parser.add_argument("--summarize-only", action="store_true", help="Only generate SUMMARY_TABLES.md.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.summarize_only:
        summarize_tables(args.output_root, args.runs)
        return 0

    checkpoints, statistics_map = resolve_assets(args)
    trials = build_stage_trials(
        args.stage,
        output_root=args.output_root,
        runs=args.runs,
        selection_seed=args.selection_seed,
        include_cost_baselines=args.include_cost_baselines,
        record_schedule_in_components=not args.no_record_schedule_in_components,
    )
    if not args.dry_run:
        validate_inputs(
            trials,
            data_root=args.data_root,
            clean_root=args.clean_root,
            checkpoints=checkpoints,
            statistics_map=statistics_map,
            device=args.device,
        )

    for trial in trials:
        run_trial(
            trial,
            python_bin=args.python_bin,
            device=args.device,
            data_root=args.data_root,
            clean_root=args.clean_root,
            checkpoints=checkpoints,
            statistics_map=statistics_map,
            num_workers=args.num_workers,
            output_root=args.output_root,
            dry_run=args.dry_run,
            force_rerun=args.force_rerun,
        )

    if not args.dry_run:
        summarize_tables(args.output_root, args.runs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
