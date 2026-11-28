"""Schema Version 9 result summary and JSON generator."""

import math
from typing import Any, Dict, List, Optional


def summarize_rga_reliability(results: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Aggregate correctness-conditioned RGA weights over a result block.

    The correctness label is the pre-update prediction returned by the adapter.
    Weight means are pooled by sample count rather than averaged per batch, so
    the summary remains valid when the final batch is smaller than the rest.
    """
    items = [item.get("rga_reliability") for item in results]
    items = [item for item in items if isinstance(item, dict)]
    if not items:
        return None

    def total(key: str) -> float:
        return float(sum(float(item.get(key, 0.0)) for item in items))

    correct_samples = int(total("correct_samples"))
    wrong_samples = int(total("wrong_samples"))
    total_samples = int(total("total_samples"))
    selected_samples = int(total("selected_samples"))
    applied_samples = int(total("applied_samples"))
    correct_weight_sum = total("correct_weight_sum")
    wrong_weight_sum = total("wrong_weight_sum")
    total_weight_sum = total("total_weight_sum")
    applied_weight_sum = sum(
        float(item.get("update_weight_mass_retention", 0.0))
        * float(item.get("total_samples", 0.0))
        for item in items
    )
    batches = int(total("batches"))
    updated_batches = int(total("updated_batches"))
    modes = {item.get("sample_reliability_mode") for item in items}
    weight_modes = {item.get("weight_mode") for item in items}

    correct_avg = correct_weight_sum / max(1, correct_samples)
    wrong_avg = wrong_weight_sum / max(1, wrong_samples)
    return {
        "sample_reliability_mode": (
            next(iter(modes)) if len(modes) == 1 else "mixed"
        ),
        "weight_mode": (
            next(iter(weight_modes)) if len(weight_modes) == 1 else "mixed"
        ),
        "correct_samples": correct_samples,
        "wrong_samples": wrong_samples,
        "total_samples": total_samples,
        "selected_samples": selected_samples,
        "applied_samples": applied_samples,
        "correct_weight_sum": correct_weight_sum,
        "wrong_weight_sum": wrong_weight_sum,
        "total_weight_sum": total_weight_sum,
        "correct_avg_weight": correct_avg,
        "wrong_avg_weight": wrong_avg,
        "separation_gap": correct_avg - wrong_avg,
        "wrong_sample_weight_ratio": wrong_weight_sum
        / max(total_weight_sum, 1e-12),
        "selected_sample_ratio": selected_samples / max(1, total_samples),
        "update_retention": applied_samples / max(1, total_samples),
        "update_weight_mass_retention": applied_weight_sum
        / max(1, total_samples),
        "update_batch_retention": updated_batches / max(1, batches),
        "batches": batches,
    }


def summarize_corruptions(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not results:
        return {
            "mean_accuracy": 0.0,
            "micro_accuracy": 0.0,
            "total_correct": 0,
            "total_samples": 0,
            "worst_corruption": None,
            "second_half_mean_accuracy": 0.0,
        }
    total_correct = sum(item["correct"] for item in results)
    total_samples = sum(item["total"] for item in results)
    worst = min(results, key=lambda item: item["accuracy"])
    second_half = results[len(results) // 2 :]
    worst_summary = {
        "corruption": worst["corruption"],
        "accuracy": float(worst["accuracy"]),
    }
    if "cycle" in worst:
        worst_summary["cycle"] = int(worst["cycle"])
    summary = {
        "mean_accuracy": float(
            sum(item["accuracy"] for item in results) / len(results)
        ),
        "micro_accuracy": float(total_correct / max(1, total_samples)),
        "total_correct": int(total_correct),
        "total_samples": int(total_samples),
        "worst_corruption": worst_summary,
        "second_half_definition": (
            f"last_{len(second_half)}_entries_in_this_result_block"
        ),
        "second_half_mean_accuracy": float(
            sum(item["accuracy"] for item in second_half) / len(second_half)
        ),
    }
    rga_summary = summarize_rga_reliability(results)
    if rga_summary is not None:
        summary["rga_reliability"] = rga_summary
    return summary


def restoration_stats_for_output(args, restorer) -> Dict[str, Any]:
    """Return restoration diagnostics with appropriate gate metadata."""
    stats = dict(restorer.stats)
    if (
        getattr(args, "tta_method", "").endswith("_dsr")
        and getattr(args, "disable_dsr_reliability_gate", False)
    ):
        stats["reliability_gate_disabled"] = True
    return stats
