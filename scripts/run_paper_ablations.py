#!/usr/bin/env python3
"""Run Paper Ablation Experiments for FedRAS.

Implements the three core ablation blocks defined for the main paper:
  - Block A: Core Module Decomposition (Base, RGA-only, DSR-only, FedRAS-Core, FedRAS-Full)
  - Block B: RGA Joint Weighting Ablation (Uniform, Sample-only, Region-only, Product)
  - Block C: DSR Coordinate Selection Ablation (Random, Drift-only, Fisher-only, Fisher*Drift)

Evaluates on CIFAR-100-C (severity 5) across multiple FedACG checkpoints.
"""

import argparse
import json
import logging
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Dict, List, Optional

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s [%(name)s]: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOGGER = logging.getLogger("ablation_runner")

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Default Checkpoint and Statistics Paths for 3 Federated Runs
DEFAULT_RUNS = [
    {
        "name": "run1",
        "checkpoint": "train/results/cifar100_100clients_alpha0.3_equal/fedacg_resnet18/cifar100_resnet18_run1/model_round499.pt",
        "statistics": "test/outputs/fedacg_run1_statistics.pt",
    },
    {
        "name": "run2",
        "checkpoint": "train/results/cifar100_100clients_alpha0.3_equal/fedacg_resnet18/run2/model_round499.pt",
        "statistics": "test/outputs/fedacg_run2_statistics.pt",
    },
    {
        "name": "run3",
        "checkpoint": "train/results/cifar100_100clients_alpha0.3_equal/fedacg_resnet18/run3/model_round499.pt",
        "statistics": "test/outputs/fedacg_run3_statistics.pt",
    },
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run FedRAS Paper Ablation Suite (Blocks A, B, C)"
    )
    parser.add_argument(
        "--blocks",
        nargs="+",
        choices=["A", "B", "C", "all"],
        default=["all"],
        help="Which ablation blocks to execute (A, B, C, or all).",
    )
    parser.add_argument(
        "--runs",
        nargs="+",
        default=["run1", "run2", "run3"],
        help="List of run identifiers to evaluate (e.g. run1 run2 run3).",
    )
    parser.add_argument(
        "--output_dir",
        default="test/outputs/paper_ablations",
        help="Root directory for saving ablation outputs and logs.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Computing device (e.g. cuda, cuda:0, cpu).",
    )
    parser.add_argument(
        "--cifar_c_path",
        default="data/CIFAR-100-C",
        help="Path to CIFAR-100-C dataset.",
    )
    parser.add_argument(
        "--clean_data_root",
        default="data",
        help="Root path for clean CIFAR-100 evaluation.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=2,
        help="DataLoader worker count.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print command lines without actually executing them.",
    )
    parser.add_argument(
        "--force_rerun",
        action="store_true",
        help="Re-execute even if the target output JSON already exists.",
    )
    parser.add_argument(
        "--random_seeds",
        type=int,
        nargs="+",
        default=[42, 43, 44],
        help="Random seeds for Block C Random coordinate selection ablation.",
    )
    return parser.parse_args()


def validate_existing_result(output_path: Path, expected_spec: Dict[str, Any]) -> bool:
    """Validate that an existing JSON file is complete, valid, and matches the expected spec."""
    if not output_path.exists() or output_path.stat().st_size == 0:
        return False

    try:
        with open(output_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 1. Check top-level schema sections
        if "domain_wise" not in data or "continual" not in data or "clean_retention" not in data:
            LOGGER.warning("Existing JSON missing required sections: %s", output_path)
            return False

        # 2. Check domain_wise completion
        dom_summary = data.get("domain_wise", {}).get("summary", {})
        if "mean_accuracy" not in dom_summary or not math.isfinite(float(dom_summary["mean_accuracy"])):
            LOGGER.warning("Existing JSON incomplete domain_wise: %s", output_path)
            return False

        # 3. Check continual completion (must have 3 full cycles)
        cont_summary = data.get("continual", {}).get("summary", {})
        cycle_summaries = cont_summary.get("cycle_summaries", [])
        if len(cycle_summaries) < 3:
            LOGGER.warning("Existing JSON has incomplete continual cycles (%d < 3): %s", len(cycle_summaries), output_path)
            return False

        # 4. Check clean retention completion
        clean_ret = data.get("clean_retention", {})
        final_clean = clean_ret.get("final_continual_clean_accuracy")
        if final_clean is None:
            final_clean = clean_ret.get("final_clean", {}).get("accuracy")
        if final_clean is None or not math.isfinite(float(final_clean)):
            LOGGER.warning("Existing JSON missing clean retention metric: %s", output_path)
            return False

        # 5. Check metadata consistency
        if data.get("dataset") != "cifar100":
            LOGGER.warning("Existing JSON dataset mismatch: %s", output_path)
            return False

        expected_tta = expected_spec.get("tta_method")
        actual_tta = data.get("adaptation", {}).get("tta_method")
        if expected_tta and actual_tta:
            # Normalize tta_method names (e.g. fedras -> rga_dsr)
            if expected_tta == "fedras" and actual_tta != "rga_dsr" and actual_tta != "fedras":
                LOGGER.warning("Existing JSON tta_method mismatch: %s != %s in %s", actual_tta, expected_tta, output_path)
                return False
            elif expected_tta != "fedras" and actual_tta != expected_tta:
                LOGGER.warning("Existing JSON tta_method mismatch: %s != %s in %s", actual_tta, expected_tta, output_path)
                return False

        expected_criterion = expected_spec.get("selection_criterion")
        if expected_criterion:
            actual_criterion = data.get("restoration", {}).get("selection_criterion")
            if actual_criterion and actual_criterion != expected_criterion:
                LOGGER.warning("Existing JSON selection_criterion mismatch: %s != %s in %s", actual_criterion, expected_criterion, output_path)
                return False

        return True
    except Exception as e:
        LOGGER.warning("Existing JSON failed validation with error (%s): %s", e, output_path)
        return False


def run_experiment(
    cmd_args: List[str],
    output_path: Path,
    log_path: Path,
    expected_spec: Dict[str, Any],
    dry_run: bool = False,
    force_rerun: bool = False,
) -> bool:
    """Run a single test execution with logging and resume caching."""
    if not force_rerun and validate_existing_result(output_path, expected_spec):
        LOGGER.info("Skipping already validated task: %s", output_path)
        return True

    output_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    full_cmd = [sys.executable, str(PROJECT_ROOT / "main_test.py")] + cmd_args + [
        "--output_path", str(output_path)
    ]

    cmd_str = " ".join(full_cmd)
    if dry_run:
        LOGGER.info("[DRY-RUN] %s", cmd_str)
        return True

    LOGGER.info("Executing: %s", cmd_str)
    with open(log_path, "w", encoding="utf-8") as log_file:
        proc = subprocess.run(
            full_cmd,
            cwd=str(PROJECT_ROOT),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )

    if proc.returncode != 0:
        LOGGER.error(
            "Execution failed with code %d. Log file: %s",
            proc.returncode,
            log_path,
        )
        return False

    if not validate_existing_result(output_path, expected_spec):
        LOGGER.error("Execution finished but output validation failed: %s", output_path)
        return False

    LOGGER.info("Successfully finished task: %s", output_path)
    return True


def build_block_a_experiments(
    run_info: Dict[str, str],
    base_args: List[str],
    block_dir: Path,
    log_dir: Path,
) -> List[Dict[str, Any]]:
    """Block A: Core Module Decomposition (Base, RGA-only, DSR-only, FedRAS-Core, FedRAS-Full).

    Parameter Scope: All 5 rows uniformly use the final non-layer4 BatchNorm affine parameters.
    First 4 rows: Standardized Factorial Decomposition (dsr_grad_clip_norm=0.0, drift_tiered DSR, no continuous risk).
    5th row: Full Method (Continuous risk, hierarchical layer quotas, grad clip 15.0).
    """
    tasks = []
    run_name = run_info["name"]

    # Common standard controls for first 4 factorial rows:
    # Disable gradient clipping (0.0), fixed drift-tiered DSR, no reliability gate, no event trigger,
    # uniform non-layer4 BN parameter scope (no strict_mechanism_ablation flag).
    std_dsr_flags = [
        "--dsr_grad_clip_norm", "0.0",
        "--recovery_interval", "10",
        "--restore_strength", "0.7",
        "--restore_budget_mode", "drift_tiered",
        "--restore_drift_thresholds", "0.10", "0.20", "0.35",
        "--restore_drift_ratios", "0.0", "0.02", "0.05", "0.10",
        "--disable_dsr_reliability_gate",
        "--disable_dsr_event_trigger",
        "--dsr_selection_criterion", "fisher_drift",
    ]

    # 1. Base (Unweighted regional adaptation baseline)
    tasks.append({
        "name": f"block_a_base_{run_name}",
        "args": base_args + ["--tta_method", "base"] + std_dsr_flags,
        "output": block_dir / f"block_a_base_{run_name}.json",
        "log": log_dir / f"block_a_base_{run_name}.log",
        "spec": {"tta_method": "base"},
    })

    # 2. RGA-only (Batch adaptation without DSR)
    tasks.append({
        "name": f"block_a_rga_only_{run_name}",
        "args": base_args + ["--tta_method", "rga"] + std_dsr_flags,
        "output": block_dir / f"block_a_rga_only_{run_name}.json",
        "log": log_dir / f"block_a_rga_only_{run_name}.log",
        "spec": {"tta_method": "rga"},
    })

    # 3. DSR-only (Periodic restoration on Base without RGA weighting)
    tasks.append({
        "name": f"block_a_dsr_only_{run_name}",
        "args": base_args + ["--tta_method", "base_dsr"] + std_dsr_flags,
        "output": block_dir / f"block_a_dsr_only_{run_name}.json",
        "log": log_dir / f"block_a_dsr_only_{run_name}.log",
        "spec": {"tta_method": "base_dsr"},
    })

    # 4. FedRAS-Core (RGA + Standard Drift-Tiered DSR)
    tasks.append({
        "name": f"block_a_fedras_core_{run_name}",
        "args": base_args + ["--tta_method", "rga_dsr"] + std_dsr_flags,
        "output": block_dir / f"block_a_fedras_core_{run_name}.json",
        "log": log_dir / f"block_a_fedras_core_{run_name}.log",
        "spec": {"tta_method": "rga_dsr"},
    })

    # 5. FedRAS-Full (Primary Final Configuration: Continuous Risk + Hierarchical Budget)
    tasks.append({
        "name": f"block_a_fedras_full_{run_name}",
        "args": base_args + ["--tta_method", "fedras"],
        "output": block_dir / f"block_a_fedras_full_{run_name}.json",
        "log": log_dir / f"block_a_fedras_full_{run_name}.log",
        "spec": {"tta_method": "fedras"},
    })

    return tasks


def build_block_b_experiments(
    run_info: Dict[str, str],
    base_args: List[str],
    block_dir: Path,
    log_dir: Path,
) -> List[Dict[str, Any]]:
    """Block B: RGA Joint Weighting Ablation (Uniform, Sample-only, Region-only, Product).

    Critical control:
      - Fixed search radius eta_min = eta_max = 1.2;
      - Pure weighting comparison without sample filtering:
        rga_hybrid_threshold = 0.0, skip_update_tau_s = 0.0, skip_update_tau_ess = 0.0.
      - Bounded batch normalization enabled (c_max = 2.0).
    """
    tasks = []
    run_name = run_info["name"]

    weighting_modes = [
        ("uniform", "uniform"),
        ("sample_only", "sample_only"),
        ("region_only", "region_only"),
        ("product", "product"),
    ]

    for label, mode in weighting_modes:
        cmd_flags = base_args + [
            "--tta_method", "rga_dsr",
            "--rga_weight_mode", mode,
            "--rga_region_scale_min", "1.2",
            "--rga_region_scale_max", "1.2",
            "--rga_hybrid_threshold", "0.0",
            "--rga_skip_update_tau_s", "0.0",
            "--rga_skip_update_tau_ess", "0.0",
            "--rga_normalize_by_weight",
            "--rga_c_max", "2.0",
            "--rga_anchor_rectify_beta", "0.3",
            "--rga_view_consistency",
            "--rga_view_alpha", "0.8",
            "--rga_view_js_scale", "1.0",
            "--recovery_interval", "10",
            "--restore_strength", "0.7",
            "--restore_budget_mode", "drift_tiered",
            "--disable_dsr_reliability_gate",
            "--disable_dsr_event_trigger",
            "--dsr_grad_clip_norm", "0.0",
        ]
        tasks.append({
            "name": f"block_b_weight_{label}_{run_name}",
            "args": cmd_flags,
            "output": block_dir / f"block_b_weight_{label}_{run_name}.json",
            "log": log_dir / f"block_b_weight_{label}_{run_name}.log",
            "spec": {"tta_method": "rga_dsr"},
        })

    return tasks


def build_block_c_experiments(
    run_info: Dict[str, str],
    base_args: List[str],
    block_dir: Path,
    log_dir: Path,
    random_seeds: List[int],
) -> List[Dict[str, Any]]:
    """Block C: DSR Coordinate Selection Ablation (Random, Drift-only, Fisher-only, Fisher*Drift).

    Common fair control:
      - Fixed ratio 5% (restore_budget_mode fixed, sparse_ratio 0.05);
      - Fixed strength 0.7, interval 10;
      - No hierarchical budget, no continuous risk, no gate;
      - Disabling gradient clipping (0.0).
    """
    tasks = []
    run_name = run_info["name"]

    common_c_flags = [
        "--tta_method", "rga_dsr",
        "--restore_budget_mode", "fixed",
        "--sparse_ratio", "0.05",
        "--restore_strength", "0.7",
        "--recovery_interval", "10",
        "--disable_dsr_reliability_gate",
        "--disable_dsr_event_trigger",
        "--disable_dsr_continuous_risk",
        "--disable_dsr_hierarchical_budget",
        "--dsr_grad_clip_norm", "0.0",
    ]

    # 1. Deterministic Selection Modes
    deterministic_modes = [
        ("drift_only", "drift_only"),
        ("fisher_only", "fisher_only"),
        ("fisher_drift", "fisher_drift"),
    ]
    for label, criterion in deterministic_modes:
        tasks.append({
            "name": f"block_c_select_{label}_{run_name}",
            "args": base_args + common_c_flags + ["--dsr_selection_criterion", criterion],
            "output": block_dir / f"block_c_select_{label}_{run_name}.json",
            "log": log_dir / f"block_c_select_{label}_{run_name}.log",
            "spec": {"tta_method": "rga_dsr", "selection_criterion": criterion},
        })

    # 2. Random Selection with Multi-Seed Replications
    for seed in random_seeds:
        tasks.append({
            "name": f"block_c_select_random_s{seed}_{run_name}",
            "args": base_args + common_c_flags + [
                "--dsr_selection_criterion", "random",
                "--dsr_selection_seed", str(seed),
            ],
            "output": block_dir / f"block_c_select_random_s{seed}_{run_name}.json",
            "log": log_dir / f"block_c_select_random_s{seed}_{run_name}.log",
            "spec": {"tta_method": "rga_dsr", "selection_criterion": "random"},
        })

    return tasks


def main():
    args = parse_args()
    output_root = Path(args.output_dir)
    log_root = output_root / "logs"

    blocks = set(args.blocks)
    if "all" in blocks:
        blocks = {"A", "B", "C"}

    LOGGER.info("Starting Paper Ablation Suite. Active Blocks: %s", sorted(list(blocks)))
    LOGGER.info("Output root: %s", output_root)

    # Filter active runs
    active_runs = [r for r in DEFAULT_RUNS if r["name"] in args.runs]
    if not active_runs:
        LOGGER.error("No valid runs selected from: %s", args.runs)
        sys.exit(1)

    all_tasks = []

    for run_info in active_runs:
        checkpoint_path = PROJECT_ROOT / run_info["checkpoint"]
        statistics_path = PROJECT_ROOT / run_info["statistics"]

        if not args.dry_run:
            if not checkpoint_path.exists():
                LOGGER.warning("Checkpoint not found for %s: %s", run_info["name"], checkpoint_path)
            if not statistics_path.exists():
                LOGGER.warning("Statistics not found for %s: %s", run_info["name"], statistics_path)

        base_cmd_args = [
            "--dataset", "cifar100",
            "--federated_method", "fedacg",
            "--model_path", str(checkpoint_path),
            "--anchor_key", "lookahead_anchor",
            "--statistics_path", str(statistics_path),
            "--cifar_c_path", args.cifar_c_path,
            "--clean_data_root", args.clean_data_root,
            "--device", args.device,
            "--num_workers", str(args.num_workers),
        ]

        if "A" in blocks:
            b_dir = output_root / "block_a_core"
            l_dir = log_root / "block_a_core"
            all_tasks.extend(build_block_a_experiments(run_info, base_cmd_args, b_dir, l_dir))

        if "B" in blocks:
            b_dir = output_root / "block_b_weighting"
            l_dir = log_root / "block_b_weighting"
            all_tasks.extend(build_block_b_experiments(run_info, base_cmd_args, b_dir, l_dir))

        if "C" in blocks:
            b_dir = output_root / "block_c_selection"
            l_dir = log_root / "block_c_selection"
            all_tasks.extend(build_block_c_experiments(run_info, base_cmd_args, b_dir, l_dir, args.random_seeds))

    LOGGER.info("Total ablation experiment tasks generated: %d", len(all_tasks))

    success_count = 0
    fail_count = 0

    for i, task in enumerate(all_tasks, 1):
        LOGGER.info("[%d/%d] Processing %s", i, len(all_tasks), task["name"])
        ok = run_experiment(
            cmd_args=task["args"],
            output_path=task["output"],
            log_path=task["log"],
            expected_spec=task.get("spec", {}),
            dry_run=args.dry_run,
            force_rerun=args.force_rerun,
        )
        if ok:
            success_count += 1
        else:
            fail_count += 1
            LOGGER.warning("Task %s failed. Continuing to next task...", task["name"])

    LOGGER.info("=" * 60)
    LOGGER.info(
        "Ablation Suite Finished. Total: %d, Success: %d, Failed: %d",
        len(all_tasks),
        success_count,
        fail_count,
    )
    LOGGER.info("=" * 60)

    if fail_count > 0 and not args.dry_run:
        LOGGER.error("Ablation suite completed with %d failed tasks! Exiting with code 1.", fail_count)
        sys.exit(1)


if __name__ == "__main__":
    main()
