"""Run the FedDR experiment family under one matched CIFAR-C protocol.

FedDR is the primary method defined as ``DLRS + FSLR``.
FedACG supplies the upstream federated initialization/anchor; DLRS performs
soft joint sample/region reliability weighting, and FSLR controls long-term
parameter drift. The primary comparison roster is Source, Tent, EATA, ASR,
SAR, CoTTA, RoTTA, PeTTA, ReCAP, and FedDR.
"""

import argparse
from copy import deepcopy
import json
import logging
import math
import os
from pathlib import Path
import sys
import time
from typing import Dict, Tuple

import numpy as np
import torch
from torch import nn

from sparse_restore import PeriodicSparseRestorer, load_fishers

# Modularized sub-packages
from adapters import (
    ASR_OFFICIAL_A0,
    ASR_OFFICIAL_M0,
    ASR_OFFICIAL_P0,
    ASR_OFFICIAL_R0,
    ASR_OFFICIAL_R1,
    BASE_METHODS,
    DLRS_METHODS,
    DLRS_REGION_SCALE_MAX,
    DLRS_REGION_SCALE_MIN,
    DLRS_RELIABILITY_ALPHA,
    DUAL_RELIABILITY_METHODS,
    EATA_D_MARGIN,
    EATA_E_MARGIN_FACTOR,
    EATA_FISHER_ALPHA,
    FEDDR_DEFINITION,
    FEDDR_NAME,
    FEDDR_PRIMARY_DISABLE_FSLR_RELIABILITY_GATE,
    FEDDR_PRIMARY_NORMALIZE_BY_WEIGHT,
    FEDDR_PRIMARY_WEIGHT_MODE,
    FEDDR_PRIMARY_HYBRID_THRESHOLD,
    FEDDR_PRIMARY_ANCHOR_RECTIFY_BETA,
    FEDDR_PRIMARY_LR,
    FEDDR_PRIMARY_BN_SCOPE,
    FEDDR_PRIMARY_FSLR_CONTINUOUS_RISK,
    FEDDR_PRIMARY_DLRS_VIEW_CONSISTENCY,
    FEDDR_PRIMARY_DLRS_VIEW_ALPHA,
    FEDDR_PRIMARY_DLRS_VIEW_JS_SCALE,
    FEDDR_PRIMARY_FSLR_HIERARCHICAL_BUDGET,
    FEDDR_PRIMARY_FSLR_LAYER_QUOTA_CAP,
    FEDDR_TTA_METHOD,
    FSLR_GRAD_CLIP_NORM,
    L0_FACTOR,
    MARGIN_QUALITY_THRESHOLD,
    OFFICIAL_SCALE,
    RECAP_SOFT_WEIGHT_LOGIT_LIMIT,
    RECAP_SOFT_WEIGHT_TEMPERATURE_FACTOR,
    REGION_SCALE,
    REWEIGHT_CAP,
    TAU_RE_FACTOR,
    TTA_METHOD_LABELS,
    TTA_METHODS,
    ASRAdapter,
    DLRSAdapter,
    DisabledSparseRestorer,
    DualReliabilityAdapter,
    RegionConfidenceProxy,
    dlrs_joint_reliability_weight,
    dlrs_sample_reliability,
    dual_perturbation_consistency,
    get_linear_classifier,
    prediction_js_divergence,
    recap_region_soft_weight,
    resolve_tta_method,
    top1_top2_probability_margin,
)

from evaluator import (
    CIFARCDataset,
    CONTINUAL_CYCLES,
    CORRUPTIONS,
    DATASET_STATS,
    REVISIT_CORRUPTIONS,
    SEVERITY,
    build_main_adapter,
    build_transform,
    configure_bn_adaptation,
    configure_feddr_bn_adaptation,
    configure_official_bn_adaptation,
    evaluate_corruption,
    evaluate_frozen,
    evaluate_initial_clean,
    evaluate_post_stream,
    load_deployment_model,
    load_feature_statistics,
    load_petta_statistics,
    make_corruption_loader,
    run_continual,
    run_domain_wise,
    select_anchor_state,
    select_lookahead_state,
    validate_statistics,
)

from metrics import (
    _counter_delta,
    _mapping_delta,
    _synchronize,
    resource_summary,
    restoration_stats_for_output,
    summarize_corruptions,
)

from test.tta_baselines import (
    CoTTAAdapter,
    DeYOAdapter,
    PeTTAAdapter,
    RoTTAAdapter,
    SARAdapter,
    configure_cotta_model,
    configure_deyo_model,
    configure_rotta_model,
    configure_sar_model,
)

LOGGER = logging.getLogger("main_test")
PROJECT_ROOT = Path(__file__).resolve().parent

# Frozen main-method protocol constants
FEDDR_PROTOCOL_BATCH_SIZE = 128
FEDDR_PROTOCOL_LR = 1.0e-3
FEDDR_PROTOCOL_MOMENTUM = 0.9
FEDDR_PROTOCOL_WEIGHT_DECAY = 0.0
FEDDR_BN_ADAPT_LR = FEDDR_PROTOCOL_LR

RESTORE_DRIFT_THRESHOLDS = (0.10, 0.20, 0.35)
RESTORE_DRIFT_RATIOS = (0.0, 0.02, 0.05, 0.10)
RESTORE_RELIABILITY_THRESHOLD = 0.55
RESTORE_RELIABILITY_EMA_DECAY = 0.90
SPARSE_RATIO = 0.01

PRIMARY_COMPARISON_METHODS = (
    "Source",
    "Tent",
    "EATA",
    "ASR",
    "SAR",
    "CoTTA",
    "RoTTA",
    "PeTTA",
    "ReCAP",
    FEDDR_NAME,
)
FEDERATED_SUPPLEMENT_METHODS = (
    "fedavg",
    "fedprox",
    "scaffold",
    "feddyn",
    "fedadam",
)


def resolve_device(name: str) -> torch.device:
    if name.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(name)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Matched federated TTA adaptation with sparse restoration"
    )
    parser.add_argument(
        "--dataset",
        choices=["cifar10", "cifar100", "tiny-imagenet"],
        default="cifar100",
    )
    parser.add_argument("--model_path", required=True)
    parser.add_argument(
        "--federated_method",
        choices=[
            "fedacg",
            "fedavg",
            "fedprox",
            "scaffold",
            "feddyn",
            "fedadam",
        ],
        default="fedacg",
    )
    parser.add_argument(
        "--anchor_key",
        choices=["lookahead_anchor", "adaptation_anchor"],
        default="lookahead_anchor",
    )
    parser.add_argument(
        "--tta_method",
        choices=TTA_METHODS,
        default=None,
        help="Matched TTA comparison method. Default entry is feddr.",
    )
    parser.add_argument(
        "--strict_mechanism_ablation",
        action="store_true",
        help="Use common all-BN-affine parameter scope for strict Base/DLRS/FSLR ablation.",
    )
    parser.add_argument(
        "--statistics_path",
        required=True,
        help="Output from prepare_main_statistics.py for the same checkpoint.",
    )
    parser.add_argument("--cifar_c_path", default=None)
    parser.add_argument("--clean_data_root", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Evaluation/adaptation batch size; defaults to the FedDR protocol value.",
    )
    parser.add_argument(
        "--adapt_lr",
        type=float,
        default=None,
        help="TTA adaptation learning rate; defaults to the FedDR protocol value.",
    )
    parser.add_argument(
        "--adapt_bn_scope",
        choices=["method_default", "all", "excluding_layer4", "last_two", "last_stage"],
        default="method_default",
        help="Override the BN affine scope for ASR/Tent/ReCAP stability checks.",
    )
    parser.add_argument(
        "--feddr_bn_scope",
        choices=["method_default", "all", "excluding_layer4", "last_two", "last_stage"],
        default="method_default",
        help="BN affine scope for the FedDR DLRS+FSLR entrypoint.",
    )
    parser.add_argument(
        "--asr_anchor_fisher_alpha",
        type=float,
        default=0.0,
        help="Normalized Fisher penalty toward the immutable ASR deployment anchor.",
    )
    parser.add_argument(
        "--asr_anchor_l2_alpha",
        type=float,
        default=0.0,
        help="Normalized L2 penalty toward the immutable ASR deployment anchor.",
    )
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument(
        "--margin_threshold",
        type=float,
        default=MARGIN_QUALITY_THRESHOLD,
        help="Absolute Top-1 minus Top-2 probability margin for non-EATA ReCAP.",
    )
    parser.add_argument(
        "--recap_confidence_gate",
        choices=["none", "source_agreement", "source_calibrated"],
        default="none",
        help="Optional anchor-aware ReCAP confidence gate for continual streams.",
    )
    parser.add_argument(
        "--recap_gate_margin_headroom",
        type=float,
        default=0.10,
        help="Allowed current-vs-source margin increase for source_calibrated ReCAP.",
    )
    parser.add_argument(
        "--recap_soft_weight_temperature_factor",
        type=float,
        default=RECAP_SOFT_WEIGHT_TEMPERATURE_FACTOR,
    )
    parser.add_argument(
        "--fslr_grad_clip_norm",
        type=float,
        default=FSLR_GRAD_CLIP_NORM,
    )
    parser.add_argument("--eata_e_margin_factor", type=float, default=EATA_E_MARGIN_FACTOR)
    parser.add_argument("--eata_d_margin", type=float, default=EATA_D_MARGIN)
    parser.add_argument("--eata_fisher_alpha", type=float, default=EATA_FISHER_ALPHA)
    parser.add_argument("--cotta_mt_alpha", type=float, default=0.999)
    parser.add_argument("--cotta_restore_probability", type=float, default=0.01)
    parser.add_argument("--cotta_anchor_confidence", type=float, default=0.72)
    parser.add_argument("--cotta_augmentation_count", type=int, default=32)
    parser.add_argument("--deyo_entropy_margin_factor", type=float, default=0.5)
    parser.add_argument("--deyo_reweight_margin_factor", type=float, default=0.4)
    parser.add_argument("--deyo_plpd_threshold", type=float, default=0.2)
    parser.add_argument("--deyo_patch_grid", type=int, default=4)
    # CoLA paper settings.  The generic batch/seed/order/cycle protocol still
    # comes from the main FEDEATA evaluation; these are the method-specific
    # principal-agent parameters from the ETA+CoLA implementation.
    parser.add_argument("--cola_weight_margin", type=float, default=0.01)
    parser.add_argument(
        "--cola_shift_detection_threshold", type=float, default=0.1
    )
    parser.add_argument("--cola_domain_ema", type=float, default=0.2)
    parser.add_argument("--cola_alpha_lr", type=float, default=0.1)
    parser.add_argument("--cola_alpha_weight_decay", type=float, default=0.1)
    parser.add_argument("--cola_delta_weight_decay", type=float, default=0.4)
    parser.add_argument(
        "--cola_entropy_margin_factor", type=float, default=EATA_E_MARGIN_FACTOR
    )
    parser.add_argument("--cola_diversity_margin", type=float, default=EATA_D_MARGIN)
    parser.add_argument(
        "--cola_max_num_vectors",
        type=int,
        default=32,
        help="Maximum CoLA domain vectors, matching the official lifelong setting.",
    )
    parser.add_argument("--sar_margin_factor", type=float, default=0.4)
    parser.add_argument("--sar_reset_threshold", type=float, default=0.2)
    parser.add_argument("--sar_sam_rho", type=float, default=0.05)
    parser.add_argument("--rotta_memory_size", type=int, default=64)
    parser.add_argument("--rotta_update_frequency", type=int, default=64)
    parser.add_argument("--rotta_nu", type=float, default=0.001)
    parser.add_argument("--rotta_bn_alpha", type=float, default=0.05)
    parser.add_argument("--petta_lambda", type=float, default=10.0)
    parser.add_argument("--petta_anchor_weight", type=float, default=1.0)
    parser.add_argument("--petta_alpha", type=float, default=0.001)
    parser.add_argument(
        "--petta_regularizer",
        choices=["l2", "cosine", "none"],
        default="cosine",
    )
    parser.add_argument(
        "--petta_loss_func",
        choices=["sce", "ce"],
        default="sce",
    )
    parser.add_argument(
        "--dlrs_region_scale_min",
        type=float,
        default=DLRS_REGION_SCALE_MIN,
    )
    parser.add_argument(
        "--dlrs_region_scale_max",
        type=float,
        default=DLRS_REGION_SCALE_MAX,
    )
    parser.add_argument(
        "--dlrs_reliability_alpha",
        type=float,
        default=DLRS_RELIABILITY_ALPHA,
    )
    parser.add_argument(
        "--dlrs_sample_reliability_mode",
        choices=["full", "entropy"],
        default="full",
        help="Sample-side RGA reliability: full composite score or entropy-only.",
    )
    parser.add_argument(
        "--dlrs_ablation_variant",
        choices=[
            "full",
            "uniform_recap",
            "regional_only",
            "anchor_only",
            "regional_anchor",
        ],
        default="full",
        help="Mechanism-screen variant; full preserves the official RGA path.",
    )
    parser.add_argument(
        "--dlrs_cue_ablation",
        choices=[
            "full",
            "minus_entropy",
            "minus_patch_shuffle",
            "minus_margin",
            "minus_anchor",
            "minus_weak_view",
            "reduced",
        ],
        default="full",
        help=(
            "RGA cue ablation; reduced removes entropy, margin and weak-view "
            "cues and also sends the reduced reliability signal to DSR/FSLR."
        ),
    )
    parser.add_argument(
        "--dlrs_normalize_by_weight",
        action="store_true",
    )
    parser.add_argument(
        "--dlrs_weight_mode",
        choices=["product", "sample_only", "region_only", "uniform"],
        default="product",
    )
    parser.add_argument(
        "--dlrs_dual_perturbation",
        action="store_true",
    )
    parser.add_argument(
        "--dlrs_consistency_weight",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--anchor_disagreement_diagnostic",
        action="store_true",
    )
    parser.add_argument(
        "--disable_fslr_reliability_gate",
        action="store_true",
    )
    parser.add_argument(
        "--recovery_interval",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--restore_strength",
        type=float,
        default=0.7,
    )
    parser.add_argument(
        "--restore_budget_mode",
        choices=["fixed", "drift_tiered"],
        default="drift_tiered",
    )
    parser.add_argument(
        "--sparse_ratio",
        type=float,
        default=0.01,
        help="Fixed-budget restoration ratio used when --restore_budget_mode=fixed.",
    )
    parser.add_argument(
        "--restore_drift_thresholds",
        type=float,
        nargs=3,
        default=RESTORE_DRIFT_THRESHOLDS,
    )
    parser.add_argument(
        "--restore_drift_ratios",
        type=float,
        nargs=4,
        default=RESTORE_DRIFT_RATIOS,
    )
    parser.add_argument(
        "--restore_reliability_threshold",
        type=float,
        default=RESTORE_RELIABILITY_THRESHOLD,
    )
    parser.add_argument(
        "--restore_reliability_ema_decay",
        type=float,
        default=RESTORE_RELIABILITY_EMA_DECAY,
    )
    parser.add_argument(
        "--dlrs_c_max",
        type=float,
        default=2.0,
        help="Maximum scale factor for Bounded Scale Normalization.",
    )
    parser.add_argument(
        "--dlrs_hybrid_threshold",
        type=float,
        default=0.2,
        help="Hard-threshold below which sample reliability is zeroed out (Hybrid Soft-Hard DLRS).",
    )
    parser.add_argument(
        "--fslr_event_trigger",
        action="store_true",
        default=True,
        help="Enable sustained triple-condition event trigger for emergency FSLR.",
    )
    parser.add_argument(
        "--disable_fslr_event_trigger",
        dest="fslr_event_trigger",
        action="store_false",
        help="Disable sustained triple-condition event trigger for emergency FSLR.",
    )
    parser.add_argument(
        "--fslr_event_k",
        type=int,
        default=3,
        help="Number of consecutive batches required to confirm an emergency event.",
    )
    parser.add_argument(
        "--fslr_event_tau_r",
        type=float,
        default=0.40,
        help="Reliability threshold for emergency event condition.",
    )
    parser.add_argument(
        "--fslr_event_tau_h",
        type=float,
        default=1.50,
        help="Entropy z-score drop threshold for emergency event condition.",
    )
    parser.add_argument(
        "--fslr_event_tau_d",
        type=float,
        default=0.20,
        help="BN drift threshold for emergency event condition.",
    )
    parser.add_argument(
        "--fslr_event_lambda_0",
        type=float,
        default=0.50,
        help="Base emergency restore strength.",
    )
    parser.add_argument(
        "--fslr_event_lambda_max",
        type=float,
        default=0.90,
        help="Maximum emergency restore strength.",
    )
    parser.add_argument(
        "--dlrs_skip_update_tau_s",
        type=float,
        default=0.05,
        help="Skip update threshold for sum of weights over batch size S_t / B.",
    )
    parser.add_argument(
        "--dlrs_skip_update_tau_ess",
        type=float,
        default=0.05,
        help="Skip update threshold for effective sample size ratio ESS_t / B.",
    )
    parser.add_argument(
        "--fslr_event_tau_a",
        type=float,
        default=0.50,
        help="Lookahead anchor disagreement rate threshold for emergency event condition.",
    )
    parser.add_argument(
        "--fslr_event_k_rho",
        type=float,
        default=2.0,
        help="Multiplier for sparse restore budget during emergency event.",
    )
    parser.add_argument(
        "--fslr_event_rho_max",
        type=float,
        default=0.20,
        help="Maximum restore budget ratio during emergency event.",
    )
    parser.add_argument(
        "--fslr_event_cooldown",
        type=int,
        default=20,
        help="Cooldown period in batches after an emergency event is triggered.",
    )
    parser.add_argument(
        "--fslr_online_fisher_beta",
        type=float,
        default=0.0,
        help="Weight of online test-time gradient second-moment in hybrid Fisher.",
    )
    parser.add_argument(
        "--dlrs_early_exit",
        action="store_true",
        help="Enable Fast-DLRS early-exit for high-confidence samples.",
    )
    parser.add_argument(
        "--dlrs_fast_perturbation",
        action="store_true",
        help="Enable Fast-DLRS lightweight perturbation.",
    )
    parser.add_argument(
        "--dlrs_anchor_rectify_beta",
        type=float,
        default=0.0,
        help="Soft penalty factor for Lookahead Anchor disagreement in DLRS sample reliability (P3).",
    )
    parser.add_argument(
        "--dlrs_view_consistency",
        action="store_true",
        default=None,
        help="Enable Composite Multi-view Weak Perturbation Consistency (Scheme 1).",
    )
    parser.add_argument(
        "--disable_dlrs_view_consistency",
        dest="dlrs_view_consistency",
        action="store_false",
        help="Disable Composite Multi-view Weak Perturbation Consistency.",
    )
    parser.add_argument(
        "--dlrs_view_alpha",
        type=float,
        default=None,
        help="Base confidence retention factor in weak view soft fusion.",
    )
    parser.add_argument(
        "--dlrs_view_js_scale",
        type=float,
        default=None,
        help="Scale factor for Jensen-Shannon divergence exponent in view reliability.",
    )
    parser.add_argument(
        "--fslr_continuous_risk",
        action="store_true",
        default=None,
        help="Enable Continuous Effective Risk budget control for FSLR (P1).",
    )
    parser.add_argument(
        "--disable_fslr_continuous_risk",
        dest="fslr_continuous_risk",
        action="store_false",
        help="Disable Continuous Effective Risk budget control for FSLR.",
    )
    parser.add_argument(
        "--fslr_risk_rho_min",
        type=float,
        default=0.0,
        help="Minimum restore budget ratio for continuous risk FSLR.",
    )
    parser.add_argument(
        "--fslr_risk_rho_max",
        type=float,
        default=0.10,
        help="Maximum restore budget ratio for continuous risk FSLR.",
    )
    parser.add_argument(
        "--fslr_risk_gamma_ess",
        type=float,
        default=0.50,
        help="Weight of (1 - ESS/B) in continuous effective risk calculation.",
    )
    parser.add_argument(
        "--fslr_risk_gamma_anchor",
        type=float,
        default=0.50,
        help="Weight of anchor disagreement in continuous effective risk calculation.",
    )
    parser.add_argument(
        "--fslr_risk_gamma_drift",
        type=float,
        default=0.0,
        help="Weight of parameter drift in continuous effective risk calculation.",
    )
    parser.add_argument(
        "--fslr_risk_tau_drift",
        type=float,
        default=0.35,
        help="Drift normalization threshold for continuous effective risk calculation.",
    )
    parser.add_argument(
        "--fslr_hierarchical_budget",
        action="store_true",
        default=None,
        help="Enable Two-Stage Hierarchical Layer Risk Budget Allocation (Scheme 2).",
    )
    parser.add_argument(
        "--disable_fslr_hierarchical_budget",
        dest="fslr_hierarchical_budget",
        action="store_false",
        help="Disable Two-Stage Hierarchical Layer Risk Budget Allocation.",
    )
    parser.add_argument(
        "--fslr_layer_quota_cap",
        type=float,
        default=None,
        help="Maximum parameter restoration fraction allowed for any individual layer.",
    )
    parser.add_argument(
        "--fslr_momentum_discounting",
        action="store_true",
        help="Enable Direction-Aware and Risk-Adaptive Momentum Discounting (Scheme 3).",
    )
    parser.add_argument(
        "--fslr_selection_criterion",
        choices=["fisher_drift", "drift_only", "fisher_only", "random"],
        default="fisher_drift",
        help="Coordinate selection criterion for FSLR ablation (Block C).",
    )
    parser.add_argument(
        "--fslr_selection_seed",
        type=int,
        default=None,
        help="Random seed for random coordinate selection in FSLR ablation.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--max_batches",
        type=int,
        default=-1,
    )
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--review_diagnostics_path", default=None)
    parser.add_argument("--review_run_id", default=None)
    parser.add_argument("--review_beta_a", type=float, default=0.3)
    parser.add_argument("--review_continual_only", action="store_true")
    args = parser.parse_args()
    # Preserve explicit CLI overrides when the public FedDR entrypoint fills
    # in primary defaults. This is needed for controlled ablations that set
    # a primary-config field to a deliberate non-default value.
    args._explicit_cli_options = {
        token.split("=", 1)[0]
        for token in sys.argv[1:]
        if token.startswith("--")
    }
    return args


def apply_official_baseline_config(args):
    """Apply the single FedDR protocol to every comparison method."""
    if getattr(args, "batch_size", None) is None:
        args.batch_size = FEDDR_PROTOCOL_BATCH_SIZE
    args.optimizer = "sgd"
    if getattr(args, "adapt_lr", None) is None:
        args.adapt_lr = FEDDR_PROTOCOL_LR
    args.momentum = FEDDR_PROTOCOL_MOMENTUM
    args.weight_decay = FEDDR_PROTOCOL_WEIGHT_DECAY
    return args


def apply_feddr_primary_config(args, public_tta_method):
    """Fix the public FedDR entrypoint to the retained primary configuration."""
    if public_tta_method != "feddr":
        if getattr(args, "dlrs_view_consistency", None) is None:
            args.dlrs_view_consistency = False
        if getattr(args, "dlrs_view_alpha", None) is None:
            args.dlrs_view_alpha = 0.70
        if getattr(args, "dlrs_view_js_scale", None) is None:
            args.dlrs_view_js_scale = 1.0
        if getattr(args, "fslr_continuous_risk", None) is None:
            args.fslr_continuous_risk = False
        if getattr(args, "fslr_hierarchical_budget", None) is None:
            args.fslr_hierarchical_budget = False
        if getattr(args, "fslr_layer_quota_cap", None) is None:
            args.fslr_layer_quota_cap = 0.35
        return args

    if not getattr(args, "_adapt_lr_explicit", False):
        args.adapt_lr = FEDDR_PRIMARY_LR
    if getattr(args, "feddr_bn_scope", "method_default") == "method_default":
        args.feddr_bn_scope = FEDDR_PRIMARY_BN_SCOPE

    args.dlrs_normalize_by_weight = FEDDR_PRIMARY_NORMALIZE_BY_WEIGHT
    explicit_options = getattr(args, "_explicit_cli_options", set())
    if "--dlrs_weight_mode" not in explicit_options:
        args.dlrs_weight_mode = FEDDR_PRIMARY_WEIGHT_MODE
    args.disable_fslr_reliability_gate = (
        FEDDR_PRIMARY_DISABLE_FSLR_RELIABILITY_GATE
    )
    if "--dlrs_hybrid_threshold" not in explicit_options:
        args.dlrs_hybrid_threshold = FEDDR_PRIMARY_HYBRID_THRESHOLD
    if getattr(args, "fslr_grad_clip_norm", None) is None:
        args.fslr_grad_clip_norm = 15.0
    args.dlrs_anchor_rectify_beta = FEDDR_PRIMARY_ANCHOR_RECTIFY_BETA
    if getattr(args, "fslr_continuous_risk", None) is None:
        args.fslr_continuous_risk = FEDDR_PRIMARY_FSLR_CONTINUOUS_RISK
    if getattr(args, "dlrs_view_consistency", None) is None:
        args.dlrs_view_consistency = FEDDR_PRIMARY_DLRS_VIEW_CONSISTENCY
    if getattr(args, "dlrs_view_alpha", None) is None:
        args.dlrs_view_alpha = FEDDR_PRIMARY_DLRS_VIEW_ALPHA
    if getattr(args, "dlrs_view_js_scale", None) is None:
        args.dlrs_view_js_scale = FEDDR_PRIMARY_DLRS_VIEW_JS_SCALE
    if getattr(args, "fslr_hierarchical_budget", None) is None:
        args.fslr_hierarchical_budget = FEDDR_PRIMARY_FSLR_HIERARCHICAL_BUDGET
    if getattr(args, "fslr_layer_quota_cap", None) is None:
        args.fslr_layer_quota_cap = FEDDR_PRIMARY_FSLR_LAYER_QUOTA_CAP
    return args


def apply_cola_primary_config(args):
    """Apply ETA+CoLA's paper-specific settings on the shared protocol."""
    if getattr(args, "public_tta_method", None) != "cola":
        return args
    if not getattr(args, "_adapt_lr_explicit", False):
        args.adapt_lr = FEDDR_PROTOCOL_LR
    if getattr(args, "batch_size", None) is None:
        args.batch_size = FEDDR_PROTOCOL_BATCH_SIZE
    args.optimizer = "sgd"
    args.momentum = FEDDR_PROTOCOL_MOMENTUM
    # CoLA applies its paper-specified weight decay to the current domain delta
    # in a separate optimizer; the shared protocol-level value remains zero.
    args.weight_decay = FEDDR_PROTOCOL_WEIGHT_DECAY
    return args


def run_review_evaluations(args, device, feature_variance, fishers, statistics_meta):
    """Run the review-requested stages and optionally omit domain-wise work."""
    if getattr(args, "review_continual_only", False):
        continual, metadata = run_continual(
            args, device, feature_variance, fishers, statistics_meta
        )
        return None, continual, metadata
    domain_wise, domain_metadata = run_domain_wise(
        args, device, feature_variance, fishers, statistics_meta
    )
    continual, continual_metadata = run_continual(
        args, device, feature_variance, fishers, statistics_meta
    )
    if domain_metadata != continual_metadata:
        raise RuntimeError("Domain-wise and Continual method builds are inconsistent")
    return domain_wise, continual, domain_metadata

def run():
    args = parse_args()
    if args.review_diagnostics_path:
        if not args.review_run_id:
            raise ValueError("--review_run_id is required with review diagnostics")
        if not math.isfinite(args.review_beta_a) or not 0.0 <= args.review_beta_a < 1.0:
            raise ValueError("--review_beta_a must be finite and in [0, 1)")
        review_path = Path(args.review_diagnostics_path)
        if review_path.exists():
            raise FileExistsError(f"refusing to append to existing diagnostics: {review_path}")
        review_path.parent.mkdir(parents=True, exist_ok=True)
    args._adapt_lr_explicit = args.adapt_lr is not None
    args.public_tta_method = args.tta_method
    args.tta_method = resolve_tta_method(args.public_tta_method)
    args = apply_official_baseline_config(args)
    args = apply_feddr_primary_config(args, args.public_tta_method)
    args = apply_cola_primary_config(args)
    if args.tta_method.startswith("eata"):
        args.sample_filter = "eata"
    elif args.tta_method == "tent" or args.tta_method in BASE_METHODS:
        args.sample_filter = "all_samples"
    elif args.tta_method in DLRS_METHODS:
        if getattr(args, "dlrs_hybrid_threshold", None) is not None and float(args.dlrs_hybrid_threshold) > 0.0:
            args.sample_filter = f"hybrid_soft_hard_tau{float(args.dlrs_hybrid_threshold):g}"
        else:
            args.sample_filter = "soft_reliability_all_samples"
    elif args.tta_method == "asr":
        args.sample_filter = "asr_entropy"
    elif args.tta_method == "cola":
        args.sample_filter = "eta"
    else:
        args.sample_filter = "margin"
    expected_anchor = {
        "fedacg": "lookahead_anchor",
        "fedavg": "adaptation_anchor",
        "fedprox": "adaptation_anchor",
        "scaffold": "adaptation_anchor",
        "feddyn": "adaptation_anchor",
        "fedadam": "adaptation_anchor",
    }[args.federated_method]
    if args.anchor_key != expected_anchor:
        raise ValueError(
            f"{args.federated_method} requires --anchor_key {expected_anchor}, "
            f"got {args.anchor_key}"
        )
    federated_label = {
        "fedacg": "FedACG",
        "fedavg": "FedAvg",
        "fedprox": "FedProx",
        "scaffold": "SCAFFOLD",
        "feddyn": "FedDyn",
        "fedadam": "FedAdam",
    }[args.federated_method]

    if args.federated_method == "fedacg":
        method_name = (
            FEDDR_NAME
            if args.tta_method == FEDDR_TTA_METHOD
            else TTA_METHOD_LABELS[args.tta_method]
        )
    else:
        method_name = f"{federated_label}+{TTA_METHOD_LABELS[args.tta_method]}"

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    device = resolve_device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    feature_variance, statistics_meta = load_feature_statistics(
        args.statistics_path
    )
    if args.tta_method == "petta":
        petta_mean, petta_variance = load_petta_statistics(args.statistics_path)
        if petta_mean.size(0) != DATASET_STATS[args.dataset]["num_classes"]:
            raise ValueError("PeTTA class feature statistics do not match dataset")
    fishers = load_fishers(args.statistics_path)
    initial_clean = evaluate_initial_clean(args, device)

    LOGGER.info("method: %s", method_name)
    LOGGER.info("federated initialization: %s", args.federated_method)
    LOGGER.info("adaptation anchor: %s", args.anchor_key)
    LOGGER.info("public TTA entrypoint: %s", args.public_tta_method)
    LOGGER.info("TTA method: %s", args.tta_method)
    LOGGER.info("sample filter: %s", args.sample_filter)
    LOGGER.info("prediction protocol: pre_update")

    domain_wise, continual, metadata = run_review_evaluations(
        args, device, feature_variance, fishers, statistics_meta
    )
    final_clean = continual["post_stream"]["clean"]
    clean_forgetting = initial_clean["accuracy"] - final_clean["accuracy"]
    is_region_method = (
        "recap" in args.tta_method
        or args.tta_method in BASE_METHODS
        or args.tta_method in DLRS_METHODS
    )
    is_base_method = args.tta_method in BASE_METHODS

    import hashlib
    config_dict = {
        k: (list(v) if isinstance(v, tuple) else v)
        for k, v in sorted(vars(args).items())
        if not k.startswith("_")
    }
    config_hash = hashlib.sha256(
        json.dumps(config_dict, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:16]

    payload = {
        "schema_version": 9,
        "method": method_name,
        "experiment_name": method_name,
        "public_tta_entrypoint": args.public_tta_method,
        "feddr_definition": FEDDR_DEFINITION if method_name == FEDDR_NAME else None,
        "primary_comparison_roster": list(PRIMARY_COMPARISON_METHODS),
        "federated_supplement_roster": list(FEDERATED_SUPPLEMENT_METHODS),
        "federated_method": args.federated_method,
        "anchor_key": args.anchor_key,
        "dataset": args.dataset,
        "backbone": metadata["backbone"],
        "model_path": args.model_path,
        "statistics_path": args.statistics_path,
        "protocol": {
            "severity": SEVERITY,
            "corruptions": CORRUPTIONS,
            "domain_wise_and_continual_reported_separately": not args.review_continual_only,
            "domain_wise_executed": not args.review_continual_only,
            "continual_cycles": CONTINUAL_CYCLES,
            "prediction": "pre_update",
            "seed": args.seed,
            "max_batches": args.max_batches,
            "batch_size": args.batch_size,
        },
        "adaptation": {
            "tta_method": args.tta_method,
            "parameter_scope": metadata.get("parameter_scope"),
            "strict_mechanism_ablation": bool(
                getattr(args, "strict_mechanism_ablation", False)
            ),
            "optimizer": metadata.get("optimizer", args.optimizer),
            "sample_filter": args.sample_filter,
            "learning_rate": args.adapt_lr,
            "momentum": args.momentum,
            "weight_decay": args.weight_decay,
            "bn_scope": getattr(args, "adapt_bn_scope", "method_default"),
            "feddr_bn_scope": getattr(args, "feddr_bn_scope", "method_default"),
            "asr_anchor_fisher_alpha": getattr(
                args, "asr_anchor_fisher_alpha", 0.0
            ),
            "asr_anchor_l2_alpha": getattr(args, "asr_anchor_l2_alpha", 0.0),
            "recap_confidence_gate": getattr(
                args, "recap_confidence_gate", "none"
            ),
            "recap_gate_margin_headroom": getattr(
                args, "recap_gate_margin_headroom", 0.10
            ),
            "recap_margin_threshold": getattr(args, "margin_threshold", None),
            "recap_soft_weight_temperature_factor": getattr(
                args, "recap_soft_weight_temperature_factor", None
            ),
            "dlrs_sample_reliability_mode": getattr(
                args, "dlrs_sample_reliability_mode", "full"
            ),
            "dlrs_cue_ablation": getattr(args, "dlrs_cue_ablation", "full"),
            "cola_weight_margin": getattr(args, "cola_weight_margin", None),
            "cola_shift_detection_threshold": getattr(
                args, "cola_shift_detection_threshold", None
            ),
            "cola_domain_ema": getattr(args, "cola_domain_ema", None),
            "cola_alpha_lr": getattr(args, "cola_alpha_lr", None),
            "cola_alpha_weight_decay": getattr(
                args, "cola_alpha_weight_decay", None
            ),
            "cola_delta_weight_decay": getattr(
                args, "cola_delta_weight_decay", None
            ),
            "cola_entropy_margin_factor": getattr(
                args, "cola_entropy_margin_factor", None
            ),
            "cola_diversity_margin": getattr(
                args, "cola_diversity_margin", None
            ),
            "cola_max_num_vectors": getattr(args, "cola_max_num_vectors", None),
        },
        "restoration": {
            "enabled": bool(getattr(metadata, "fslr_enabled", args.tta_method.endswith("_fslr"))),
            "budget_mode": (
                "continuous_risk"
                if getattr(args, "fslr_continuous_risk", False)
                else args.restore_budget_mode
            ),
            "interval": args.recovery_interval,
            "sparse_ratio": args.sparse_ratio,
            "restore_strength": args.restore_strength,
            "selection_criterion": getattr(args, "fslr_selection_criterion", "fisher_drift"),
            "selection_seed": getattr(args, "fslr_selection_seed", None),
            "continuous_risk": bool(getattr(args, "fslr_continuous_risk", False)),
            "hierarchical_budget": bool(getattr(args, "fslr_hierarchical_budget", False)),
            "layer_quota_cap": getattr(args, "fslr_layer_quota_cap", None),
            "drift_thresholds": list(args.restore_drift_thresholds),
            "drift_ratios": list(args.restore_drift_ratios),
        },
        "config_hash": config_hash,
        "config": config_dict,
        "domain_wise": domain_wise,
        "continual": continual,
        "clean_retention": {
            "initial_clean": initial_clean,
            "final_clean": final_clean,
            "clean_forgetting": float(clean_forgetting),
            "retention_ratio": float(
                final_clean["accuracy"] / max(initial_clean["accuracy"], 1e-12)
            ),
        },
    }

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    LOGGER.info("Saved result schema version 9 to %s", output_path)


if __name__ == "__main__":
    run()
