"""Evaluator package for TTA evaluation."""

from evaluator.dataset import (
    CIFARCDataset,
    CONTINUAL_CYCLES,
    CORRUPTIONS,
    DATASET_STATS,
    REVISIT_CORRUPTIONS,
    SEVERITY,
    build_transform,
    make_corruption_loader,
)
from evaluator.corruption_runner import (
    evaluate_corruption,
    evaluate_frozen,
    evaluate_initial_clean,
    evaluate_post_stream,
)
from evaluator.engine import (
    build_main_adapter,
    configure_bn_adaptation,
    configure_fedras_bn_adaptation,
    configure_official_bn_adaptation,
    load_deployment_model,
    load_feature_statistics,
    load_petta_statistics,
    run_continual,
    run_domain_wise,
    select_anchor_state,
    select_lookahead_state,
    validate_statistics,
)

__all__ = [
    "CIFARCDataset",
    "CONTINUAL_CYCLES",
    "CORRUPTIONS",
    "DATASET_STATS",
    "REVISIT_CORRUPTIONS",
    "SEVERITY",
    "build_transform",
    "make_corruption_loader",
    "evaluate_corruption",
    "evaluate_frozen",
    "evaluate_initial_clean",
    "evaluate_post_stream",
    "build_main_adapter",
    "configure_bn_adaptation",
    "configure_fedras_bn_adaptation",
    "configure_official_bn_adaptation",
    "load_deployment_model",
    "load_feature_statistics",
    "load_petta_statistics",
    "run_continual",
    "run_domain_wise",
    "select_anchor_state",
    "select_lookahead_state",
    "validate_statistics",
]
