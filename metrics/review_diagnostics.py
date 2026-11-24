"""Sample-level, label-side diagnostics for the 2026-09-26 review matrix."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Mapping, Sequence


def summarize_review_batch(
    *,
    sample_ids: Sequence[int],
    current_predictions: Sequence[int],
    anchor_predictions: Sequence[int],
    labels: Sequence[int],
    weights: Sequence[float],
) -> dict:
    """Return reconstructable batch records and fixed-denominator diagnostics.

    Call this only after the adaptation interface has returned its pre-update
    predictions and reliability weights. Labels are consumed here on the
    evaluator side and are never passed into the adapter.
    """
    fields = {
        "sample_ids": list(sample_ids),
        "current_predictions": list(current_predictions),
        "anchor_predictions": list(anchor_predictions),
        "labels": list(labels),
        "weights": [float(value) for value in weights],
    }
    sizes = {name: len(value) for name, value in fields.items()}
    if not sizes or len(set(sizes.values())) != 1:
        raise ValueError(f"review batch vectors must have equal lengths: {sizes}")
    if not fields["sample_ids"]:
        raise ValueError("review batch must contain at least one sample")
    if len(set(fields["sample_ids"])) != len(fields["sample_ids"]):
        raise ValueError("sample_ids must be unique within a batch")
    if any(not math.isfinite(value) or value < 0 for value in fields["weights"]):
        raise ValueError("weights must be finite and non-negative")

    current = [int(value) for value in fields["current_predictions"]]
    anchor = [int(value) for value in fields["anchor_predictions"]]
    labels_int = [int(value) for value in fields["labels"]]
    wrong = [prediction != label for prediction, label in zip(current, labels_int)]
    correct = [not value for value in wrong]
    a_i = [int(prediction != other) for prediction, other in zip(current, anchor)]
    disagree = [bool(value) for value in a_i]
    n = len(current)
    n_error = sum(wrong)
    n_correct = n - n_error
    n_error_disagree = sum(error and diff for error, diff in zip(wrong, disagree))
    n_correct_disagree = sum(ok and diff for ok, diff in zip(correct, disagree))
    weight_sum = math.fsum(fields["weights"])
    error_weight_sum = math.fsum(
        weight for weight, error in zip(fields["weights"], wrong) if error
    )

    return {
        **fields,
        "a_i": a_i,
        "prediction_timing": "pre_update",
        "N": n,
        "n": n,
        "n_E": n_error,
        "n_C": n_correct,
        "n_error": n_error,
        "n_correct": n_correct,
        "n_E_disagree": n_error_disagree,
        "n_C_disagree": n_correct_disagree,
        "n_error_disagree": n_error_disagree,
        "n_correct_disagree": n_correct_disagree,
        "d_E": n_error_disagree / n_error if n_error else None,
        "d_C": n_correct_disagree / n_correct if n_correct else None,
        "epsilon": n_error / n,
        "weight_sum": weight_sum,
        "error_weight_sum": error_weight_sum,
        "epsilon_RGA": error_weight_sum / weight_sum if weight_sum else None,
        "current_wrong_anchor_correct": sum(
            err and pred_a == label
            for err, pred_a, label in zip(wrong, anchor, labels_int)
        ),
        "current_correct_anchor_wrong": sum(
            ok and pred_a != label
            for ok, pred_a, label in zip(correct, anchor, labels_int)
        ),
        "both_wrong_different_class": sum(
            err and pred_a != label and pred != pred_a
            for err, pred_a, label, pred in zip(wrong, anchor, labels_int, current)
        ),
        "both_same_class": sum(pred == pred_a for pred, pred_a in zip(current, anchor)),
    }


def write_review_batch_jsonl(
    path: str | Path, metadata: Mapping[str, object], record: Mapping[str, object]
) -> None:
    """Append one complete, reconstructable review record as a JSONL row."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    row = {**metadata, **record}
    with target.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()


def anchor_rectification_weights(
    current_predictions: Sequence[int],
    anchor_predictions: Sequence[int],
    *,
    beta: float,
) -> list[float]:
    """Compute w_i = 1 - beta * 1[p_i != p_i^anchor]."""
    if not math.isfinite(beta) or not 0.0 <= beta < 1.0:
        raise ValueError("beta must be finite and in [0, 1)")
    if len(current_predictions) != len(anchor_predictions):
        raise ValueError("current and anchor prediction vectors must have equal lengths")
    return [
        1.0 - beta * int(int(current) != int(anchor))
        for current, anchor in zip(current_predictions, anchor_predictions)
    ]
