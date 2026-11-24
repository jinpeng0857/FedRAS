"""Single corruption, clean, and post-stream evaluation routines."""

import json
import math
from copy import deepcopy
from pathlib import Path
import time
from typing import Any, Dict, List, Tuple

import torch
from torch.utils.data import DataLoader

from adapters.base import DLRS_METHODS
from evaluator.dataset import (
    PROJECT_ROOT,
    REVISIT_CORRUPTIONS,
    SEVERITY,
    build_clean_dataset,
    build_transform,
    make_corruption_loader,
)
from metrics.collector import _counter_delta, _mapping_delta, _synchronize
from metrics.review_diagnostics import (
    anchor_rectification_weights,
    summarize_review_batch,
    write_review_batch_jsonl,
)


def evaluate_corruption(
    args,
    adapter,
    corruption: str,
    device: torch.device,
    cycle=None,
    flush_adapter: bool = False,
):
    dataset, loader = make_corruption_loader(args, corruption)
    review_path = getattr(args, "review_diagnostics_path", None)
    write_review_diagnostics = bool(review_path and cycle is not None)
    review_anchor_model = None
    if write_review_diagnostics:
        review_anchor_model = getattr(adapter, "anchor_model", None)
        if review_anchor_model is None:
            review_anchor_model = deepcopy(adapter.model)
            review_anchor_model.eval()
            review_anchor_model.requires_grad_(False)
        adapter.review_anchor_model = review_anchor_model
    before = dict(adapter.counters)
    restore_before = dict(adapter.restorer.stats)
    correct = 0
    total = 0
    adaptation_seconds = 0.0
    selected_counts = []
    anchor_diag_samples = 0
    anchor_diag_agreements = 0
    anchor_diag_correct_agreements = 0
    anchor_diag_correct_conflicts = 0
    anchor_diag_js_sum = 0.0
    local_gradient_norm_max = 0.0
    rga_mode = None
    rga_correct_weight_sum = 0.0
    rga_wrong_weight_sum = 0.0
    rga_weight_sum = 0.0
    rga_applied_weight_sum = 0.0
    rga_correct_count = 0
    rga_wrong_count = 0
    rga_total_samples = 0
    rga_positive_samples = 0
    rga_applied_samples = 0
    rga_batches = 0
    rga_updated_batches = 0
    diagnostic_sums = {
        "prediction_entropy": 0.0,
        "source_agreement": 0.0,
        "source_margin": 0.0,
        "anchor_drift": 0.0,
        "asr_entropy": 0.0,
        "asr_flip_rate": 0.0,
        "anchor_regularization": 0.0,
        "batch_mean_reliability": 0.0,
        "anchor_disagreement": 0.0,
        "sb_ratio": 0.0,
        "ess_ratio": 0.0,
    }
    diagnostic_counts = {key: 0 for key in diagnostic_sums}
    asr_reset_events = 0
    last_anchor_drift = None
    sample_offset = 0
    try:
        for batch_index, (images, labels) in enumerate(loader):
            if args.max_batches > 0 and batch_index >= args.max_batches:
                break
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            _synchronize(device)
            start = time.perf_counter()
            logits, batch_stats = adapter.adapt(images)
            _synchronize(device)
            adaptation_seconds += time.perf_counter() - start
            if write_review_diagnostics:
                current_predictions = logits.detach().argmax(dim=1).cpu().tolist()
                with torch.no_grad():
                    anchor_predictions = (
                        review_anchor_model(images).argmax(dim=1).cpu().tolist()
                    )
                beta_a = float(getattr(args, "review_beta_a", 0.3))
                weights = anchor_rectification_weights(
                    current_predictions, anchor_predictions, beta=beta_a
                )
                record = summarize_review_batch(
                    sample_ids=list(range(sample_offset, sample_offset + len(current_predictions))),
                    current_predictions=current_predictions,
                    anchor_predictions=anchor_predictions,
                    labels=labels.detach().cpu().tolist(),
                    weights=weights,
                )
                scalar_stats = {
                    key: value
                    for key, value in batch_stats.items()
                    if not torch.is_tensor(value)
                    and (
                        value is None
                        or isinstance(value, (str, bool, int))
                        or (isinstance(value, float) and math.isfinite(value))
                    )
                }
                restorer = adapter.restorer
                restoration = {}
                for key, value in vars(restorer).items():
                    if not key.startswith("last_"):
                        continue
                    if isinstance(value, dict):
                        scalar_value = {
                            name: item
                            for name, item in value.items()
                            if item is None
                            or isinstance(item, (str, bool, int))
                            or (isinstance(item, float) and math.isfinite(item))
                        }
                        if len(scalar_value) == len(value):
                            restoration[key] = scalar_value
                    elif (
                        value is None
                        or isinstance(value, (str, bool, int))
                        or (isinstance(value, float) and math.isfinite(value))
                    ):
                        restoration[key] = value
                record["adaptation"] = scalar_stats
                record["dsr"] = restoration
                for key, output_key in (
                    ("rga_joint_weight_vector", "actual_rga_joint_weight_vector"),
                    ("rga_applied_weight_vector", "applied_rga_weight_vector"),
                ):
                    vector = batch_stats.get(key)
                    if torch.is_tensor(vector):
                        record[output_key] = vector.detach().cpu().tolist()
                write_review_batch_jsonl(
                    review_path,
                    {
                        "run_id": getattr(args, "review_run_id", None),
                        "model_path": args.model_path,
                        "seed": args.seed,
                        "corruption": corruption,
                        "severity": SEVERITY,
                        "cycle": cycle,
                        "batch_index": batch_index,
                        "sample_id_start": sample_offset,
                        "beta_a": beta_a,
                    },
                    record,
                )
                sample_offset += len(current_predictions)
            selected_counts.append(batch_stats["selected"])
            for key in diagnostic_sums:
                value = batch_stats.get(key)
                if value is not None and math.isfinite(float(value)):
                    diagnostic_sums[key] += float(value)
                    diagnostic_counts[key] += 1
                    if key == "anchor_drift":
                        last_anchor_drift = float(value)
            asr_reset_events += int(bool(batch_stats.get("asr_reset", False)))
            correct_mask = logits.argmax(dim=1).eq(labels)
            correct += correct_mask.sum().item()
            total += int(labels.size(0))
            rga_weights = batch_stats.get("rga_joint_weight_vector")
            if rga_weights is not None:
                if not torch.is_tensor(rga_weights):
                    raise TypeError("rga_joint_weight_vector must be a tensor")
                rga_weights = rga_weights.detach().to(
                    device=labels.device, dtype=torch.float64
                )
                if rga_weights.ndim != 1 or rga_weights.numel() != labels.numel():
                    raise ValueError(
                        "RGA weight vector must have one finite value per sample"
                    )
                if not torch.isfinite(rga_weights).all() or (rga_weights < 0).any():
                    raise ValueError("RGA weights must be finite and non-negative")
                rga_mode = getattr(
                    adapter, "sample_reliability_mode", rga_mode
                )
                correct_weights = rga_weights[correct_mask]
                wrong_weights = rga_weights[~correct_mask]
                rga_correct_weight_sum += float(correct_weights.sum().item())
                rga_wrong_weight_sum += float(wrong_weights.sum().item())
                rga_weight_sum += float(rga_weights.sum().item())
                rga_correct_count += int(correct_mask.sum().item())
                rga_wrong_count += int((~correct_mask).sum().item())
                rga_total_samples += int(labels.numel())
                rga_positive_samples += int((rga_weights > 0).sum().item())
                applied_weights = batch_stats.get("rga_applied_weight_vector")
                if applied_weights is None:
                    applied_weights = (
                        rga_weights
                        if bool(batch_stats.get("updated", False))
                        else torch.zeros_like(rga_weights)
                    )
                else:
                    applied_weights = applied_weights.detach().to(
                        device=labels.device, dtype=torch.float64
                    )
                if (
                    applied_weights.ndim != 1
                    or applied_weights.numel() != labels.numel()
                    or not torch.isfinite(applied_weights).all()
                    or (applied_weights < 0).any()
                ):
                    raise ValueError(
                        "applied RGA weights must be finite and non-negative"
                    )
                rga_applied_weight_sum += float(applied_weights.sum().item())
                rga_applied_samples += int((applied_weights > 0).sum().item())
                rga_batches += 1
                rga_updated_batches += int(bool(batch_stats.get("updated", False)))
            if batch_stats.get("gradient_norm_before_clip") is not None:
                local_gradient_norm_max = max(
                    local_gradient_norm_max,
                    float(batch_stats["gradient_norm_before_clip"]),
                )
            if "anchor_agreement_vector" in batch_stats:
                agreement = batch_stats["anchor_agreement_vector"].to(
                    labels.device
                )
                js_values = batch_stats["anchor_js_vector"].to(labels.device)
                anchor_diag_samples += int(agreement.numel())
                anchor_diag_agreements += int(agreement.sum().item())
                anchor_diag_correct_agreements += int(
                    (agreement & correct_mask).sum().item()
                )
                anchor_diag_correct_conflicts += int(
                    ((~agreement) & correct_mask).sum().item()
                )
                anchor_diag_js_sum += float(js_values.sum().item())
    finally:
        dataset.close()
    if flush_adapter:
        adapter.finalize()

    after = dict(adapter.counters)
    restore_after = dict(adapter.restorer.stats)
    samples = int(_counter_delta(after, before, "samples"))
    selected = int(_counter_delta(after, before, "selected"))
    relative_entropy_quality_batches = int(
        _counter_delta(
            after,
            before,
            "relative_entropy_quality_batches",
        )
    )
    gradient_norm_batches = int(
        after.get("gradient_norm_batches", 0)
        - before.get("gradient_norm_batches", 0)
    )
    histogram = {
        str(value): selected_counts.count(value)
        for value in sorted(set(selected_counts))
    }
    rga_reliability = None
    if rga_total_samples > 0:
        rga_reliability = {
            "sample_reliability_mode": rga_mode,
            "weight_mode": getattr(adapter, "weight_mode", None),
            "correct_samples": int(rga_correct_count),
            "wrong_samples": int(rga_wrong_count),
            "total_samples": int(rga_total_samples),
            "selected_samples": int(rga_positive_samples),
            "applied_samples": int(rga_applied_samples),
            "correct_weight_sum": float(rga_correct_weight_sum),
            "wrong_weight_sum": float(rga_wrong_weight_sum),
            "total_weight_sum": float(rga_weight_sum),
            "correct_avg_weight": float(
                rga_correct_weight_sum / max(1, rga_correct_count)
            ),
            "wrong_avg_weight": float(
                rga_wrong_weight_sum / max(1, rga_wrong_count)
            ),
            "separation_gap": float(
                rga_correct_weight_sum / max(1, rga_correct_count)
                - rga_wrong_weight_sum / max(1, rga_wrong_count)
            ),
            "wrong_sample_weight_ratio": float(
                rga_wrong_weight_sum / max(rga_weight_sum, 1e-12)
            ),
            "selected_sample_ratio": float(
                rga_positive_samples / max(1, rga_total_samples)
            ),
            "update_retention": float(
                rga_applied_samples / max(1, rga_total_samples)
            ),
            "update_weight_mass_retention": float(
                rga_applied_weight_sum / max(1, rga_total_samples)
            ),
            "update_batch_retention": float(
                rga_updated_batches / max(1, rga_batches)
            ),
            "batches": int(rga_batches),
            "updated_batches": int(rga_updated_batches),
        }
    result = {
        "corruption": corruption,
        "severity": SEVERITY,
        "accuracy": float(correct / total if total else 0.0),
        "correct": int(correct),
        "total": int(total),
        "quality_gate": getattr(adapter, "sample_filter", None),
        "quality_pass_ratio": float(
            _counter_delta(after, before, "quality_passed")
            / max(1, samples)
        ),
        "margin_pass_ratio": float(
            _counter_delta(after, before, "margin_passed")
            / max(1, samples)
        ),
        "mean_prediction_margin": float(
            _counter_delta(after, before, "margin_sum")
            / max(1, samples)
        ),
        "entropy_pass_ratio": float(
            _counter_delta(after, before, "entropy_passed") / max(1, samples)
        ),
        "entropy_pass_role": (
            "legacy_alias_of_quality_pass_ratio"
            if getattr(adapter, "sample_filter", None) == "margin"
            else "method_specific_entropy_filter"
        ),
        "relative_entropy_pass_ratio": (
            None
            if relative_entropy_quality_batches == 0
            else float(
                _counter_delta(
                    after,
                    before,
                    "relative_entropy_passed",
                )
                / max(1, samples)
            )
        ),
        "absolute_entropy_pass_ratio": (
            None
            if relative_entropy_quality_batches == 0
            else float(
                _counter_delta(
                    after,
                    before,
                    "absolute_entropy_passed",
                )
                / max(1, samples)
            )
        ),
        "absolute_entropy_threshold": (
            getattr(adapter, "relative_entropy_absolute_margin", None)
            if relative_entropy_quality_batches > 0
            else None
        ),
        "mean_relative_entropy_threshold": (
            None
            if relative_entropy_quality_batches == 0
            else float(
                _counter_delta(
                    after,
                    before,
                    "relative_entropy_threshold_sum",
                )
                / relative_entropy_quality_batches
            )
        ),
        "mean_effective_entropy_threshold": (
            None
            if relative_entropy_quality_batches == 0
            else float(
                _counter_delta(
                    after,
                    before,
                    "effective_entropy_threshold_sum",
                )
                / relative_entropy_quality_batches
            )
        ),
        "nonredundant_ratio": float(
            _counter_delta(after, before, "nonredundant") / max(1, samples)
        ),
        "region_reliable_ratio": float(
            _counter_delta(after, before, "region_reliable")
            / max(1, samples)
        ),
        "selected_ratio": float(selected / max(1, samples)),
        "mean_recap_soft_weight": float(
            _counter_delta(after, before, "recap_soft_weight_sum")
            / max(
                1,
                _counter_delta(after, before, "recap_soft_weight_count"),
            )
        ),
        "rms_recap_soft_weight": float(
            math.sqrt(
                _counter_delta(
                    after,
                    before,
                    "recap_soft_weight_square_sum",
                )
                / max(
                    1,
                    _counter_delta(
                        after,
                        before,
                        "recap_soft_weight_count",
                    ),
                )
            )
        ),
        "fslr_gradient_clip_enabled": bool(
            getattr(adapter, "fslr_grad_clip_norm", None) is not None
        ),
        "fslr_gradient_clip_norm": getattr(
            adapter, "fslr_grad_clip_norm", None
        ),
        "mean_gradient_norm_before_clip": (
            None
            if gradient_norm_batches == 0
            else float(
                (
                    after.get("gradient_norm_sum", 0.0)
                    - before.get("gradient_norm_sum", 0.0)
                )
                / gradient_norm_batches
            )
        ),
        "max_gradient_norm_before_clip": (
            None if gradient_norm_batches == 0 else local_gradient_norm_max
        ),
        "gradient_clipped_batches": int(
            after.get("gradient_clipped_batches", 0)
            - before.get("gradient_clipped_batches", 0)
        ),
        "gradient_clip_ratio": float(
            (
                after.get("gradient_clipped_batches", 0)
                - before.get("gradient_clipped_batches", 0)
            )
            / max(1, gradient_norm_batches)
        ),
        "nonfinite_gradient_batches": int(
            after.get("nonfinite_gradient_batches", 0)
            - before.get("nonfinite_gradient_batches", 0)
        ),
        "batches": int(_counter_delta(after, before, "batches")),
        "updated_batches": int(
            _counter_delta(after, before, "updated_batches")
        ),
        "skipped_batches": int(
            _counter_delta(after, before, "skipped_batches")
        ),
        "nonfinite_batches": int(
            _counter_delta(after, before, "nonfinite_batches")
        ),
        "diagnostics": {
            "mean_prediction_entropy": (
                diagnostic_sums["prediction_entropy"]
                / max(1, diagnostic_counts["prediction_entropy"])
            ),
            "mean_source_agreement": (
                None
                if diagnostic_counts["source_agreement"] == 0
                else diagnostic_sums["source_agreement"]
                / diagnostic_counts["source_agreement"]
            ),
            "mean_source_margin": (
                None
                if diagnostic_counts["source_margin"] == 0
                else diagnostic_sums["source_margin"]
                / diagnostic_counts["source_margin"]
            ),
            "mean_anchor_drift": (
                None
                if diagnostic_counts["anchor_drift"] == 0
                else diagnostic_sums["anchor_drift"]
                / diagnostic_counts["anchor_drift"]
            ),
            "final_anchor_drift": (
                last_anchor_drift
            ),
            "mean_asr_entropy": (
                None
                if diagnostic_counts["asr_entropy"] == 0
                else diagnostic_sums["asr_entropy"]
                / diagnostic_counts["asr_entropy"]
            ),
            "mean_asr_flip_rate": (
                None
                if diagnostic_counts["asr_flip_rate"] == 0
                else diagnostic_sums["asr_flip_rate"]
                / diagnostic_counts["asr_flip_rate"]
            ),
            "mean_anchor_regularization": (
                None
                if diagnostic_counts["anchor_regularization"] == 0
                else diagnostic_sums["anchor_regularization"]
                / diagnostic_counts["anchor_regularization"]
            ),
            "mean_batch_reliability": (
                None
                if diagnostic_counts["batch_mean_reliability"] == 0
                else diagnostic_sums["batch_mean_reliability"]
                / diagnostic_counts["batch_mean_reliability"]
            ),
            "mean_anchor_disagreement": (
                None
                if diagnostic_counts["anchor_disagreement"] == 0
                else diagnostic_sums["anchor_disagreement"]
                / diagnostic_counts["anchor_disagreement"]
            ),
            "mean_sb_ratio": (
                None
                if diagnostic_counts["sb_ratio"] == 0
                else diagnostic_sums["sb_ratio"]
                / diagnostic_counts["sb_ratio"]
            ),
            "mean_ess_ratio": (
                None
                if diagnostic_counts["ess_ratio"] == 0
                else diagnostic_sums["ess_ratio"]
                / diagnostic_counts["ess_ratio"]
            ),
            "asr_reset_events": int(asr_reset_events),
            "diagnostic_batches": int(
                max(diagnostic_counts.values()) if diagnostic_counts else 0
            ),
        },
        "mean_adaptation_loss": float(
            _counter_delta(after, before, "loss_sum")
            / max(1, _counter_delta(after, before, "updated_batches"))
        ),
        "mean_recap_loss": float(
            _counter_delta(after, before, "recap_loss_sum")
            / max(1, _counter_delta(after, before, "updated_batches"))
        ),
        "mean_eata_regularization": float(
            _counter_delta(after, before, "eata_regularization_sum")
            / max(1, _counter_delta(after, before, "updated_batches"))
        ),
        "restore_events": int(
            restore_after["restore_events"] - restore_before["restore_events"]
        ),
        "scheduled_restore_events": int(
            restore_after.get("scheduled_restore_events", 0)
            - restore_before.get("scheduled_restore_events", 0)
        ),
        "skipped_low_drift_events": int(
            restore_after.get("skipped_low_drift_events", 0)
            - restore_before.get("skipped_low_drift_events", 0)
        ),
        "skipped_low_reliability_events": int(
            restore_after.get("skipped_low_reliability_events", 0)
            - restore_before.get("skipped_low_reliability_events", 0)
        ),
        "restore_budget_event_counts": _mapping_delta(
            restore_after, restore_before, "budget_event_counts"
        ),
        "restored_assignments_total": int(
            restore_after["restored_assignments_total"]
            - restore_before["restored_assignments_total"]
        ),
        "selected_count_per_batch": {
            "min": int(min(selected_counts, default=0)),
            "max": int(max(selected_counts, default=0)),
            "mean": float(
                sum(selected_counts) / max(1, len(selected_counts))
            ),
            "histogram": histogram,
        },
        "mean_adaptation_time_ms": float(
            1000.0 * adaptation_seconds / max(1, len(selected_counts))
        ),
    }
    if rga_reliability is not None:
        result["rga_reliability"] = rga_reliability
    if getattr(adapter, "tta_method", None) in DLRS_METHODS:
        batches_delta = max(1, int(_counter_delta(after, before, "batches")))
        ess_batches = max(1, int(_counter_delta(after, before, "effective_sample_size_batches")))
        mean_ess = float(_counter_delta(after, before, "effective_sample_size_sum") / ess_batches)
        skipped_low_quality = int(_counter_delta(after, before, "skipped_low_quality_batches"))
        anchor_batches = int(_counter_delta(after, before, "anchor_disagreement_batches"))
        mean_anchor_disagree = (
            float(_counter_delta(after, before, "anchor_disagreement_sum") / anchor_batches)
            if anchor_batches > 0
            else None
        )
        rec_count = max(1, int(_counter_delta(after, before, "recovery_credit_count")))
        mean_rec_credit = float(_counter_delta(after, before, "recovery_credit_sum") / rec_count)
        event_triggers = int(restore_after.get("num_event_triggers", 0) - restore_before.get("num_event_triggers", 0))

        result.update(
            {
                "dlrs_entropy_reference_ratio": float(
                    _counter_delta(
                        after, before, "deyo_entropy_passed"
                    )
                    / max(1, samples)
                ),
                "dlrs_plpd_reference_ratio": float(
                    _counter_delta(after, before, "plpd_passed")
                    / max(1, samples)
                ),
                "dlrs_sample_selection": (
                    f"hybrid_soft_hard_tau{float(getattr(adapter, 'hybrid_threshold', 0.0)):g}"
                    if getattr(adapter, "hybrid_threshold", None) is not None
                    and float(getattr(adapter, "hybrid_threshold", 0.0) or 0.0) > 0.0
                    else "soft_all_samples"
                ),
                "dlrs_hybrid_threshold": getattr(adapter, "hybrid_threshold", None),
                "dlrs_normalize_by_weight": bool(
                    getattr(adapter, "normalize_by_weight", False)
                ),
                "dlrs_weight_mode": getattr(adapter, "weight_mode", "product"),
                "dlrs_mean_sample_reliability": float(
                    _counter_delta(
                        after, before, "dlrs_reliability_sum"
                    )
                    / max(
                        1,
                        _counter_delta(
                            after, before, "dlrs_reliability_count"
                        ),
                    )
                ),
                "dlrs_mean_region_scale": float(
                    _counter_delta(
                        after, before, "dlrs_region_scale_sum"
                    )
                    / max(
                        1,
                        _counter_delta(
                            after, before, "dlrs_reliability_count"
                        ),
                    )
                ),
                "dlrs_mean_joint_weight": float(
                    _counter_delta(after, before, "dlrs_joint_weight_sum")
                    / max(
                        1,
                        _counter_delta(
                            after,
                            before,
                            "dlrs_joint_weight_count",
                        ),
                    )
                ),
                "dlrs_rms_joint_weight": float(
                    math.sqrt(
                        _counter_delta(
                            after,
                            before,
                            "dlrs_joint_weight_square_sum",
                        )
                        / max(
                            1,
                            _counter_delta(
                                after,
                                before,
                                "dlrs_joint_weight_count",
                            ),
                        )
                    )
                ),
                "dlrs_mean_effective_sample_size": mean_ess,
                "dlrs_mean_ess_ratio": float(mean_ess / max(1.0, float(samples) / max(1, batches_delta))),
                "dlrs_skipped_low_quality_batches": skipped_low_quality,
                "dlrs_skip_update_ratio": float(skipped_low_quality / batches_delta),
                "dlrs_mean_anchor_disagreement": mean_anchor_disagree,
                "dlrs_mean_recovery_credit": mean_rec_credit,
                "fslr_num_event_triggers": event_triggers,
            }
        )
    if cycle is not None:
        result["cycle"] = int(cycle)
    print(
        f"{'cycle ' + str(cycle) + ' ' if cycle is not None else ''}"
        f"{corruption}: accuracy={100.0 * result['accuracy']:.2f}% "
        f"quality_pass={100.0 * result['quality_pass_ratio']:.2f}% "
        f"selected={100.0 * result['selected_ratio']:.2f}% "
        f"recovery={result['restored_assignments_total']}"
    )
    return result


@torch.inference_mode()
def evaluate_frozen(adapter, loader, device, max_batches=-1):
    correct = 0
    total = 0
    for batch_index, (images, labels) in enumerate(loader):
        if max_batches > 0 and batch_index >= max_batches:
            break
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = adapter.predict(images)
        correct += logits.argmax(dim=1).eq(labels).sum().item()
        total += int(labels.size(0))
    return correct, total


@torch.inference_mode()
def evaluate_initial_clean(args, device):
    """Evaluate the untouched deployment anchor with source BN statistics."""
    from utils.model_utils import build_model, infer_backbone_from_state_dict

    anchor_key = getattr(args, "anchor_key", "lookahead_anchor")
    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    state_dict = (
        checkpoint[anchor_key] if isinstance(checkpoint, dict) and anchor_key in checkpoint
        else (checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint)
    )
    backbone = infer_backbone_from_state_dict(state_dict)
    num_classes = int(state_dict["fc.weight"].shape[0])
    model = build_model(backbone, num_classes=num_classes)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    clean_root = (
        Path(args.clean_data_root)
        if args.clean_data_root
        else PROJECT_ROOT / "data"
    )
    dataset = build_clean_dataset(
        args.dataset, clean_root, build_transform(args.dataset)
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    correct = 0
    total = 0
    for batch_index, (images, labels) in enumerate(loader):
        if args.max_batches > 0 and batch_index >= args.max_batches:
            break
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        correct += logits.argmax(dim=1).eq(labels).sum().item()
        total += int(labels.size(0))
    return {
        "protocol": f"untouched_{anchor_key}_with_source_bn_statistics",
        "accuracy": float(correct / total if total else 0.0),
        "correct": int(correct),
        "total": int(total),
    }


def evaluate_post_stream(args, adapter, online_results, device):
    counters_before = dict(adapter.counters)
    restorer_before = dict(adapter.restorer.stats)
    params_before = {
        name: param.detach().clone()
        for name, param in adapter.model.named_parameters()
    }
    was_training = adapter.model.training
    adapter.model.eval()

    clean_root = (
        Path(args.clean_data_root)
        if args.clean_data_root
        else PROJECT_ROOT / "data"
    )
    clean_dataset = build_clean_dataset(
        args.dataset, clean_root, build_transform(args.dataset)
    )
    clean_loader = DataLoader(
        clean_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    clean_correct, clean_total = evaluate_frozen(
        adapter, clean_loader, device, args.max_batches
    )

    first_pass = {}
    for item in online_results:
        first_pass.setdefault(item["corruption"], item)
    revisits = []
    for corruption in REVISIT_CORRUPTIONS:
        dataset, loader = make_corruption_loader(args, corruption)
        try:
            correct, total = evaluate_frozen(
                adapter, loader, device, args.max_batches
            )
        finally:
            dataset.close()
        first_accuracy = first_pass[corruption]["accuracy"]
        accuracy = correct / total if total else 0.0
        revisits.append(
            {
                "corruption": corruption,
                "accuracy": float(accuracy),
                "correct": int(correct),
                "total": int(total),
                "first_pass_accuracy": float(first_accuracy),
                "forgetting": float(first_accuracy - accuracy),
            }
        )

    adapter.model.train(was_training)
    counters_unchanged = adapter.counters == counters_before
    restorer_unchanged = adapter.restorer.stats == restorer_before
    parameters_unchanged = all(
        torch.equal(adapter.model.get_parameter(name).detach(), value)
        for name, value in params_before.items()
    )
    if not (counters_unchanged and restorer_unchanged and parameters_unchanged):
        raise RuntimeError("frozen post-stream evaluation mutated method state")

    mean_revisit = sum(item["accuracy"] for item in revisits) / len(revisits)
    mean_forgetting = sum(item["forgetting"] for item in revisits) / len(revisits)
    return {
        "protocol": "frozen_final_model_no_adaptation",
        "parameters_unchanged": parameters_unchanged,
        "adapter_counters_unchanged": counters_unchanged,
        "restorer_counters_unchanged": restorer_unchanged,
        "clean": {
            "accuracy": float(clean_correct / clean_total if clean_total else 0.0),
            "correct": int(clean_correct),
            "total": int(clean_total),
        },
        "revisit": revisits,
        "mean_revisit_accuracy": float(mean_revisit),
        "mean_forgetting": float(mean_forgetting),
    }
