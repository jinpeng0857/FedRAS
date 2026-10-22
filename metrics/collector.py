"""Resource monitoring and counter arithmetic utilities."""

import time
from typing import Any, Dict, List, Optional
import torch


def _counter_delta(after: Dict[str, Any], before: Dict[str, Any], key: str) -> float:
    return float(after.get(key, 0) - before.get(key, 0))


def _mapping_delta(after: Dict[str, Any], before: Dict[str, Any], key: str) -> Dict[str, int]:
    after_values = after.get(key, {})
    before_values = before.get(key, {})
    return {
        item: int(after_values.get(item, 0) - before_values.get(item, 0))
        for item in sorted(set(after_values) | set(before_values))
    }


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def resource_summary(results: List[Dict[str, Any]], wall_seconds: float, device: torch.device) -> Dict[str, Any]:
    batches = sum(item["batches"] for item in results)
    weighted_adaptation_ms = sum(
        item["mean_adaptation_time_ms"] * item["batches"] for item in results
    )
    peak_memory = (
        float(torch.cuda.max_memory_allocated(device) / (1024.0**2))
        if device.type == "cuda"
        else None
    )
    return {
        "wall_time_seconds": float(wall_seconds),
        "mean_adaptation_time_ms": float(
            weighted_adaptation_ms / max(1, batches)
        ),
        "peak_cuda_memory_mb": peak_memory,
        "relative_overhead_to_pure_recap": None,
        "relative_overhead_note": (
            "Not computed because comparison implementations are intentionally "
            "excluded from the main-only codebase."
        ),
    }
