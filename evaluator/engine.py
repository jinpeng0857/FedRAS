"""Evaluation engine and adapter factory."""

import math
from pathlib import Path
import time
from typing import Any, Dict, Tuple

import torch
from torch import nn

from sparse_restore import PeriodicSparseRestorer
from adapters.base import (
    ASR_OFFICIAL_A0,
    ASR_OFFICIAL_M0,
    ASR_OFFICIAL_P0,
    ASR_OFFICIAL_R0,
    ASR_OFFICIAL_R1,
    BASE_METHODS,
    DLRS_METHODS,
    DisabledSparseRestorer,
    FEDDR_PRIMARY_DLRS_VIEW_ALPHA,
    FEDDR_PRIMARY_DLRS_VIEW_CONSISTENCY,
    FEDDR_PRIMARY_DLRS_VIEW_JS_SCALE,
    FEDDR_PRIMARY_FSLR_HIERARCHICAL_BUDGET,
    FEDDR_PRIMARY_FSLR_LAYER_QUOTA_CAP,
    REWEIGHT_CAP,
    get_linear_classifier,
)
from adapters.dual_reliability import DualReliabilityAdapter
from adapters.dlrs import DLRSAdapter
from adapters.asr import ASRAdapter
from adapters.cola import (
    CoLAAdapter,
    CoLAOptimizer,
    CoLAProxyRestorer,
    configure_cola_model,
)
from evaluator.dataset import (
    CONTINUAL_CYCLES,
    CORRUPTIONS,
    DATASET_STATS,
)
from evaluator.corruption_runner import (
    evaluate_corruption,
    evaluate_post_stream,
)
from metrics.collector import _counter_delta, _mapping_delta, resource_summary
from metrics.schema import restoration_stats_for_output, summarize_corruptions
from tests.sar_official.sam import SAM
from tests.tta_baselines import (
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
from utils.model_utils import build_model, infer_backbone_from_state_dict


def select_anchor_state(checkpoint, anchor_key="lookahead_anchor"):
    if not isinstance(checkpoint, dict) or anchor_key not in checkpoint:
        raise KeyError(
            f"the experiment requires a checkpoint containing {anchor_key}"
        )
    state = checkpoint[anchor_key]
    if not isinstance(state, dict):
        raise TypeError(f"checkpoint['{anchor_key}'] must be a state_dict")
    if any(name.startswith("module.") for name in state):
        state = {name.removeprefix("module."): value for name, value in state.items()}
    return state


def select_lookahead_state(checkpoint):
    """Backward-compatible selector for existing FedACG callers."""
    return select_anchor_state(checkpoint, "lookahead_anchor")


def load_deployment_model(
    model_path: str,
    device: torch.device,
    anchor_key: str = "lookahead_anchor",
):
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    state = select_anchor_state(checkpoint, anchor_key)
    classifier_weight = state.get("fc.weight")
    if classifier_weight is None:
        raise KeyError(f"{anchor_key} is missing fc.weight")

    backbone = infer_backbone_from_state_dict(state)
    model = build_model(
        backbone=backbone,
        num_classes=int(classifier_weight.shape[0]),
        norm_type="bn",
    )
    model.load_state_dict(state, strict=True)
    return model.to(device), backbone


def _skip_top_layer(module_name: str) -> bool:
    return module_name == "layer4" or module_name.startswith("layer4.")


def configure_bn_adaptation(model: nn.Module, scope: str = "excluding_layer4"):
    """Use batch statistics with an explicit BN affine update scope."""
    valid_scopes = {"all", "excluding_layer4", "last_two", "last_stage"}
    if scope not in valid_scopes:
        raise ValueError(f"unsupported BN adaptation scope: {scope}")

    def scope_allows(module_name: str) -> bool:
        if scope == "all":
            return True
        if scope == "excluding_layer4":
            return not _skip_top_layer(module_name)
        if scope == "last_stage":
            return _skip_top_layer(module_name)
        return (
            module_name == "layer3"
            or module_name.startswith("layer3.")
            or _skip_top_layer(module_name)
        )

    model.train()
    model.requires_grad_(False)
    for name, module in model.named_modules():
        if not isinstance(module, nn.BatchNorm2d):
            continue
        module.train()
        update_affine = scope_allows(name)
        if module.weight is not None:
            module.weight.requires_grad_(update_affine)
        if module.bias is not None:
            module.bias.requires_grad_(update_affine)
        module.track_running_stats = False
        module.running_mean = None
        module.running_var = None

    params = [param for param in model.parameters() if param.requires_grad]
    names = [name for name, param in model.named_parameters() if param.requires_grad]
    if not params:
        raise RuntimeError("no trainable BN affine parameters were found")
    return params, names


def configure_official_bn_adaptation(model: nn.Module):
    """Enable every BatchNorm2d affine parameter and batch statistics."""
    model.train()
    model.requires_grad_(False)
    for module in model.modules():
        if not isinstance(module, nn.BatchNorm2d):
            continue
        module.train()
        module.requires_grad_(True)
        module.track_running_stats = False
        module.running_mean = None
        module.running_var = None

    params = [param for param in model.parameters() if param.requires_grad]
    names = [name for name, param in model.named_parameters() if param.requires_grad]
    if not params:
        raise RuntimeError("no trainable BN affine parameters were found")
    return params, names


def configure_feddr_bn_adaptation(
    model: nn.Module, scope: str = "excluding_layer4"
):
    """FedDR plasticity scope: non-layer4 BatchNorm2d affine tensors (freezing Layer 4 to eliminate gap with DeYO/SAR and reduce noise near classifier head)."""
    if scope == "method_default":
        scope = "excluding_layer4"
    return configure_bn_adaptation(model, scope=scope)


def load_feature_statistics(path: str):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError("statistics file must contain a dictionary")
    variance = payload.get("feature_variance")
    if variance is None:
        raise KeyError("statistics file is missing feature_variance")
    variance = torch.as_tensor(variance, dtype=torch.float32).flatten()
    if (
        variance.numel() == 0
        or not torch.isfinite(variance).all()
        or (variance < 0).any()
    ):
        raise ValueError("feature_variance must be finite and non-negative")
    return variance, payload.get("meta", {})


def load_petta_statistics(path: str):
    """Load class-conditional source feature moments for PeTTA."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError("statistics file must contain a dictionary")
    means = payload.get("class_feature_mean")
    variances = payload.get("class_feature_variance")
    if means is None or variances is None:
        raise KeyError(
            "PeTTA requires class_feature_mean and class_feature_variance in statistics"
        )
    means = torch.as_tensor(means, dtype=torch.float32)
    variances = torch.as_tensor(variances, dtype=torch.float32)
    if (
        means.ndim != 2
        or variances.ndim != 2
        or means.shape != variances.shape
        or not torch.isfinite(means).all()
        or not torch.isfinite(variances).all()
        or (variances < 0).any()
    ):
        raise ValueError("invalid class-conditional PeTTA feature moments")
    return means, variances


def validate_statistics(meta, args, num_classes):
    stats_classes = meta.get("num_classes")
    if stats_classes is not None and int(stats_classes) != num_classes:
        raise ValueError(
            f"statistics class mismatch: {stats_classes} != {num_classes}"
        )
    stats_key = meta.get("actual_ckpt_key")
    if stats_key is not None and stats_key != args.anchor_key:
        raise ValueError(
            "statistics anchor mismatch: "
            f"statistics={stats_key}, evaluation={args.anchor_key}"
        )
    stats_dataset = meta.get("dataset")
    if stats_dataset is not None and stats_dataset != args.dataset:
        raise ValueError(
            f"statistics dataset mismatch: {stats_dataset} != {args.dataset}"
        )
    stats_model = meta.get("model_path")
    if stats_model is not None:
        expected = Path(args.model_path).expanduser().resolve()
        recorded = Path(stats_model).expanduser().resolve()
        if recorded != expected:
            raise ValueError(
                "statistics/checkpoint mismatch: "
                f"statistics={recorded}, evaluation={expected}"
            )


def build_main_adapter(
    args,
    device,
    feature_variance,
    fishers,
    statistics_meta,
    continual_adam_exp_avg_decay=False,
    continual_adam_exp_avg_sq_decay=False,
    continual_adam_exp_avg_sq_decay_factor=None,
):
    """Build a completely fresh model, optimizer, adapter, and restorer."""
    model, backbone = load_deployment_model(
        args.model_path, device, anchor_key=args.anchor_key
    )
    num_classes = int(model.fc.out_features)
    expected_classes = DATASET_STATS[args.dataset]["num_classes"]
    if num_classes != expected_classes:
        raise ValueError(
            f"checkpoint has {num_classes} classes, expected {expected_classes}"
        )
    validate_statistics(statistics_meta, args, num_classes)
    strict_mechanism_methods = {"base", "dlrs", "base_fslr", "dlrs_fslr"}
    scope_override = getattr(args, "adapt_bn_scope", "method_default")
    use_strict_mechanism_scope = (
        bool(getattr(args, "strict_mechanism_ablation", False))
        and args.tta_method in strict_mechanism_methods
    )
    if (
        scope_override != "method_default"
        and args.tta_method in {"asr", "tent", "recap"}
    ):
        params, param_names = configure_bn_adaptation(model, scope_override)
        parameter_scope = f"bn_affine_{scope_override}"
    elif use_strict_mechanism_scope:
        params, param_names = configure_official_bn_adaptation(model)
        parameter_scope = "bn_affine_all_strict_mechanism_ablation"
    elif args.tta_method == "cola":
        params, param_names = configure_cola_model(
            model,
            weight_margin=args.cola_weight_margin,
            shift_detection_threshold=args.cola_shift_detection_threshold,
            domain_ema=args.cola_domain_ema,
            max_num_vectors=args.cola_max_num_vectors,
        )
        parameter_scope = "bn_affine_all_cola_domain_vectors"
    elif args.tta_method == "cotta":
        params, param_names = configure_cotta_model(model)
        parameter_scope = "all_weight_and_bias"
    elif args.tta_method == "tent":
        params, param_names = configure_official_bn_adaptation(model)
        parameter_scope = "bn_affine_all"
    elif args.tta_method == "asr":
        params, param_names = configure_official_bn_adaptation(model)
        parameter_scope = "bn_affine_all"
    elif args.tta_method in DLRS_METHODS:
        feddr_scope = getattr(args, "feddr_bn_scope", "excluding_layer4")
        params, param_names = configure_feddr_bn_adaptation(
            model, scope=feddr_scope
        )
        parameter_scope = f"bn_affine_{feddr_scope}"
    elif args.tta_method == "deyo":
        params, param_names = configure_deyo_model(model)
        parameter_scope = "bn_affine_excluding_layer4"
    elif args.tta_method == "sar":
        params, param_names = configure_sar_model(model)
        parameter_scope = "norm_affine_excluding_layer4"
    elif args.tta_method in ("rotta", "petta"):
        params, param_names = configure_rotta_model(
            model, alpha=args.rotta_bn_alpha
        )
        parameter_scope = "robust_bn_affine"
    else:
        params, param_names = configure_bn_adaptation(model)
        parameter_scope = "bn_affine_excluding_layer4"

    optimizer_name = getattr(args, "optimizer", "sgd")
    if args.tta_method == "cola":
        optimizer_name = "cola_eta_sgd_adamw"
        optimizer = CoLAOptimizer(
            model.cola_state,
            [
                parameter
                for name, parameter in model.named_parameters()
                if parameter.requires_grad and not name.startswith("cola_state.")
            ],
            epsilon_lr=args.adapt_lr,
            epsilon_momentum=args.momentum,
            epsilon_weight_decay=args.cola_delta_weight_decay,
            alpha_lr=args.cola_alpha_lr,
            alpha_weight_decay=args.cola_alpha_weight_decay,
        )
    elif optimizer_name == "auto":
        optimizer_name = "sgd"
    elif args.tta_method == "sar":
        optimizer_name = "sam_sgd"
        optimizer = SAM(
            params,
            torch.optim.SGD,
            rho=args.sar_sam_rho,
            adaptive=False,
            lr=args.adapt_lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
    elif optimizer_name == "adam":
        optimizer = torch.optim.Adam(
            params,
            lr=args.adapt_lr,
            weight_decay=args.weight_decay,
        )
    else:
        optimizer = torch.optim.SGD(
            params,
            lr=args.adapt_lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )

    use_fslr = args.tta_method.endswith("_fslr")
    use_fslr_reliability_gate = (
        use_fslr
        and args.tta_method in DLRS_METHODS
        and not getattr(args, "disable_fslr_reliability_gate", False)
    )
    if use_fslr:
        restorer = PeriodicSparseRestorer(
            model=model,
            optimizer=optimizer,
            fishers=fishers,
            interval=args.recovery_interval,
            sparse_ratio=getattr(args, "sparse_ratio", 0.01),
            restore_strength=args.restore_strength,
            budget_mode=("continuous_risk" if getattr(args, "fslr_continuous_risk", False) else args.restore_budget_mode),
            drift_thresholds=args.restore_drift_thresholds,
            drift_ratios=args.restore_drift_ratios,
            reliability_threshold=args.restore_reliability_threshold,
            reliability_ema_decay=args.restore_reliability_ema_decay,
            adam_exp_avg_decay=continual_adam_exp_avg_decay,
            adam_exp_avg_sq_decay=continual_adam_exp_avg_sq_decay,
            adam_exp_avg_sq_decay_factor=continual_adam_exp_avg_sq_decay_factor,
            event_trigger=getattr(args, "fslr_event_trigger", False),
            event_k=getattr(args, "fslr_event_k", 3),
            event_tau_r=getattr(args, "fslr_event_tau_r", 0.40),
            event_tau_a=getattr(args, "fslr_event_tau_a", 0.50),
            event_tau_h=getattr(args, "fslr_event_tau_h", 1.50),
            event_tau_d=getattr(args, "fslr_event_tau_d", 0.20),
            event_lambda_0=getattr(args, "fslr_event_lambda_0", 0.50),
            event_lambda_max=getattr(args, "fslr_event_lambda_max", 0.90),
            event_k_rho=getattr(args, "fslr_event_k_rho", 2.0),
            event_rho_max=getattr(args, "fslr_event_rho_max", 0.20),
            event_cooldown=getattr(args, "fslr_event_cooldown", 20),
            disable_reliability_gate=getattr(args, "disable_fslr_reliability_gate", False),
            online_fisher_beta=getattr(args, "fslr_online_fisher_beta", 0.0),
            risk_rho_min=getattr(args, "fslr_risk_rho_min", 0.0),
            risk_rho_max=getattr(args, "fslr_risk_rho_max", 0.10),
            risk_gamma_anchor=getattr(args, "fslr_risk_gamma_anchor", 0.50),
            risk_gamma_drift=getattr(args, "fslr_risk_gamma_drift", 0.0),
            risk_tau_drift=getattr(args, "fslr_risk_tau_drift", 0.35),
            hierarchical_budget=getattr(
                args, "fslr_hierarchical_budget", FEDDR_PRIMARY_FSLR_HIERARCHICAL_BUDGET
            ),
            layer_quota_cap=getattr(
                args, "fslr_layer_quota_cap", FEDDR_PRIMARY_FSLR_LAYER_QUOTA_CAP
            ),
            momentum_discounting=getattr(args, "fslr_momentum_discounting", False),
            momentum_away_factor=getattr(args, "fslr_momentum_away_factor", 0.20),
            selection_criterion=getattr(args, "fslr_selection_criterion", "fisher_drift"),
            selection_seed=getattr(args, "fslr_selection_seed", None),
        )
    elif args.tta_method == "cola":
        restorer = CoLAProxyRestorer(model)
    else:
        restorer = DisabledSparseRestorer(model)

    if args.tta_method == "cola":
        adapter = CoLAAdapter(
            model=model,
            optimizer=optimizer,
            restorer=restorer,
            num_classes=num_classes,
            entropy_margin_factor=args.cola_entropy_margin_factor,
            diversity_margin=args.cola_diversity_margin,
        )
    elif args.tta_method == "cotta":
        adapter = CoTTAAdapter(
            model=model,
            optimizer=optimizer,
            restorer=restorer,
            mt_alpha=args.cotta_mt_alpha,
            restore_probability=args.cotta_restore_probability,
            anchor_confidence=args.cotta_anchor_confidence,
            augmentation_count=args.cotta_augmentation_count,
            input_mean=DATASET_STATS[args.dataset]["mean"],
            input_std=DATASET_STATS[args.dataset]["std"],
        )
    elif args.tta_method == "deyo":
        adapter = DeYOAdapter(
            model=model,
            optimizer=optimizer,
            restorer=restorer,
            num_classes=num_classes,
            entropy_margin_factor=args.deyo_entropy_margin_factor,
            reweight_margin_factor=args.deyo_reweight_margin_factor,
            plpd_threshold=args.deyo_plpd_threshold,
            patch_grid=args.deyo_patch_grid,
        )
    elif args.tta_method == "sar":
        sar_margin = args.sar_margin_factor * math.log(float(num_classes))
        adapter = SARAdapter(
            model=model,
            optimizer=optimizer,
            restorer=restorer,
            margin=sar_margin,
            reset_threshold=args.sar_reset_threshold,
        )
    elif args.tta_method == "rotta":
        adapter = RoTTAAdapter(
            model=model,
            optimizer=optimizer,
            restorer=restorer,
            num_classes=num_classes,
            memory_size=args.rotta_memory_size,
            update_frequency=args.rotta_update_frequency,
            nu=args.rotta_nu,
            input_mean=DATASET_STATS[args.dataset]["mean"],
            input_std=DATASET_STATS[args.dataset]["std"],
        )
    elif args.tta_method == "petta":
        petta_mean, petta_variance = load_petta_statistics(args.statistics_path)
        adapter = PeTTAAdapter(
            model=model,
            optimizer=optimizer,
            restorer=restorer,
            num_classes=num_classes,
            class_feature_mean=petta_mean.to(device),
            class_feature_variance=petta_variance.to(device),
            memory_size=args.rotta_memory_size,
            update_frequency=args.rotta_update_frequency,
            nu=args.rotta_nu,
            petta_lambda=args.petta_lambda,
            anchor_weight=args.petta_anchor_weight,
            petta_alpha=args.petta_alpha,
            input_mean=DATASET_STATS[args.dataset]["mean"],
            input_std=DATASET_STATS[args.dataset]["std"],
        )
    elif args.tta_method == "asr":
        adapter = ASRAdapter(
            model=model,
            optimizer=optimizer,
            feature_variance=feature_variance.to(device),
            restorer=restorer,
            fishers=fishers,
            param_names=param_names,
            m0=ASR_OFFICIAL_M0,
            p0=ASR_OFFICIAL_P0,
            a0=ASR_OFFICIAL_A0,
            r0=ASR_OFFICIAL_R0,
            r1=ASR_OFFICIAL_R1,
            anchor_fisher_alpha=getattr(args, "asr_anchor_fisher_alpha", 0.0),
            anchor_l2_alpha=getattr(args, "asr_anchor_l2_alpha", 0.0),
        )
    elif args.tta_method in DLRS_METHODS:
        adapter = DLRSAdapter(
            model=model,
            optimizer=optimizer,
            feature_variance=feature_variance.to(device),
            restorer=restorer,
            fishers=fishers,
            tta_method=args.tta_method,
            deyo_entropy_margin_factor=args.deyo_entropy_margin_factor,
            deyo_plpd_threshold=args.deyo_plpd_threshold,
            deyo_patch_grid=args.deyo_patch_grid,
            region_scale_min=args.dlrs_region_scale_min,
            region_scale_max=args.dlrs_region_scale_max,
            reliability_alpha=args.dlrs_reliability_alpha,
            sample_reliability_mode=getattr(
                args, "dlrs_sample_reliability_mode", "full"
            ),
            normalize_by_weight=args.dlrs_normalize_by_weight,
            c_max=getattr(args, "dlrs_c_max", 2.0),
            skip_update_tau_s=getattr(args, "dlrs_skip_update_tau_s", 0.05),
            skip_update_tau_ess=getattr(args, "dlrs_skip_update_tau_ess", 0.05),
            weight_mode=args.dlrs_weight_mode,
            dual_perturbation=getattr(args, "dlrs_dual_perturbation", False),
            consistency_weight=getattr(args, "dlrs_consistency_weight", 0.5),
            anchor_disagreement_diagnostic=getattr(
                args, "anchor_disagreement_diagnostic", False
            ),
            disable_fslr_reliability_gate=getattr(
                args, "disable_fslr_reliability_gate", False
            ),
            hybrid_threshold=getattr(args, "dlrs_hybrid_threshold", None),
            anchor_rectify_beta=getattr(args, "dlrs_anchor_rectify_beta", 0.0),
            view_consistency=getattr(
                args, "dlrs_view_consistency", FEDDR_PRIMARY_DLRS_VIEW_CONSISTENCY
            ),
            view_alpha=getattr(
                args, "dlrs_view_alpha", FEDDR_PRIMARY_DLRS_VIEW_ALPHA
            ),
            view_js_scale=getattr(
                args, "dlrs_view_js_scale", FEDDR_PRIMARY_DLRS_VIEW_JS_SCALE
            ),
            rga_ablation_variant=getattr(
                args, "dlrs_ablation_variant", "full"
            ),
            rga_cue_ablation=getattr(
                args, "dlrs_cue_ablation", "full"
            ),
        )
    else:
        adapter = DualReliabilityAdapter(
            model=model,
            optimizer=optimizer,
            feature_variance=feature_variance.to(device),
            restorer=restorer,
            fishers=fishers,
            tta_method=args.tta_method,
            eata_e_margin_factor=args.eata_e_margin_factor,
            eata_d_margin=args.eata_d_margin,
            eata_fisher_alpha=args.eata_fisher_alpha,
            margin_threshold=args.margin_threshold,
            recap_soft_weight_temperature_factor=(
                args.recap_soft_weight_temperature_factor
            ),
            fslr_grad_clip_norm=args.fslr_grad_clip_norm,
            confidence_gate=getattr(args, "recap_confidence_gate", "none"),
            gate_margin_headroom=getattr(
                args, "recap_gate_margin_headroom", 0.10
            ),
        )
    use_recap = getattr(adapter, "use_recap", False)
    is_base = getattr(adapter, "use_base", False)
    use_eata = getattr(adapter, "use_eata", False)
    trainable_bn_params = []
    for module_name, module in model.named_modules():
        if not isinstance(module, nn.BatchNorm2d):
            continue
        for parameter_name, parameter in module.named_parameters(recurse=False):
            if parameter.requires_grad:
                trainable_bn_params.append(
                    (
                        f"{module_name}.{parameter_name}",
                        parameter,
                    )
                )
    metadata = {
        "backbone": backbone,
        "trainable_bn_tensors": len(trainable_bn_params),
        "trainable_bn_scalars": sum(
            param.numel() for _name, param in trainable_bn_params
        ),
        "trainable_parameter_tensors": len(param_names),
        "trainable_parameter_scalars": sum(param.numel() for param in params),
        "parameter_scope": parameter_scope,
        "strict_mechanism_ablation": bool(
            getattr(args, "strict_mechanism_ablation", False)
        ),
        "restoration_initial_scalars_per_event": restorer.selected_count,
        "restoration_scalars_per_event": restorer.selected_count,
        "restoration_max_scalars_per_event": getattr(
            restorer, "maximum_selected_count", 0
        ),
        "restoration_budget_mode": (
            args.restore_budget_mode if use_fslr else "disabled"
        ),
        "restoration_reliability_threshold": (
            args.restore_reliability_threshold
            if use_fslr_reliability_gate
            else None
        ),
        "restoration_reliability_ema_decay": (
            args.restore_reliability_ema_decay
            if use_fslr_reliability_gate
            else None
        ),
        "federated_method": args.federated_method,
        "anchor_key": args.anchor_key,
        "tta_method": args.tta_method,
        "optimizer": optimizer_name,
        "sample_filter": getattr(adapter, "sample_filter", None),
        "dlrs_sample_reliability_mode": getattr(
            adapter, "sample_reliability_mode", None
        ),
        "dlrs_cue_ablation": getattr(adapter, "rga_cue_ablation", "full"),
        "dlrs_view_consistency_effective": getattr(
            adapter, "view_consistency", None
        ),
        "dlrs_effective_rga_cues": (
            ["patch_shuffle_plpd", "federated_anchor", "regional_confidence"]
            if getattr(adapter, "rga_cue_ablation", "full") == "reduced"
            else None
        ),
        "dlrs_recovery_reliability_source": (
            "reduced_rga_signal"
            if getattr(adapter, "rga_cue_ablation", "full") == "reduced"
            else "official_composite_signal"
        ),
        "margin_threshold": getattr(adapter, "margin_threshold", None),
    }
    return adapter, metadata


def run_domain_wise(args, device, feature_variance, fishers, statistics_meta):
    """Reset the complete method before every corruption."""
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    results = []
    metadata = None
    for corruption in CORRUPTIONS:
        adapter, current_metadata = build_main_adapter(
            args, device, feature_variance, fishers, statistics_meta
        )
        if metadata is None:
            metadata = current_metadata
        elif current_metadata != metadata:
            raise RuntimeError("fresh Domain-wise builds produced inconsistent metadata")
        result = evaluate_corruption(
            args, adapter, corruption, device, flush_adapter=True
        )
        if hasattr(adapter, "diagnostics"):
            result["cola_diagnostics"] = adapter.diagnostics()
        result["final_bn_affine_drift"] = adapter.restorer.drift_stats()
        result["restoration"] = restoration_stats_for_output(
            args, adapter.restorer
        )
        results.append(result)
        del adapter

    wall_seconds = time.perf_counter() - start
    return {
        "protocol": "reset_model_optimizer_and_restorer_for_each_corruption",
        "state_continuity": False,
        "summary": summarize_corruptions(results),
        "resource": resource_summary(results, wall_seconds, device),
        "per_corruption": results,
    }, metadata


def run_continual(args, device, feature_variance, fishers, statistics_meta):
    """Keep all state across 15 corruptions and repeat the order three times."""
    adapter, metadata = build_main_adapter(
        args,
        device,
        feature_variance,
        fishers,
        statistics_meta,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    results = []
    cycle_summaries = []
    for cycle in range(1, CONTINUAL_CYCLES + 1):
        restore_before = dict(adapter.restorer.stats)
        cycle_results = [
            evaluate_corruption(args, adapter, corruption, device, cycle=cycle)
            for corruption in CORRUPTIONS
        ]
        restore_after = dict(adapter.restorer.stats)
        cycle_summary = summarize_corruptions(cycle_results)
        cycle_summary.update(
            {
                "cycle": cycle,
                "restored_assignments_total": int(
                    restore_after["restored_assignments_total"]
                    - restore_before["restored_assignments_total"]
                ),
                "newly_restored_unique_parameters": int(
                    restore_after["restored_unique_parameters"]
                    - restore_before["restored_unique_parameters"]
                ),
                "restore_events": int(
                    restore_after["restore_events"]
                    - restore_before["restore_events"]
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
                "final_bn_affine_drift": adapter.restorer.drift_stats(),
            }
        )
        cycle_summaries.append(cycle_summary)
        results.extend(cycle_results)

    adapter.finalize()
    cola_diagnostics = (
        adapter.diagnostics() if hasattr(adapter, "diagnostics") else None
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    wall_seconds = time.perf_counter() - start
    resource = resource_summary(results, wall_seconds, device)
    post_stream = evaluate_post_stream(args, adapter, results, device)
    overall_summary = summarize_corruptions(results)
    overall_summary.pop("second_half_definition", None)
    overall_summary.pop("second_half_mean_accuracy", None)
    overall_summary.update(
        {
            "cycle_summaries": cycle_summaries,
            "late_delta_cycle3_minus_cycle1": float(
                cycle_summaries[2]["mean_accuracy"]
                - cycle_summaries[0]["mean_accuracy"]
            ),
            "final_bn_affine_drift": adapter.restorer.drift_stats(),
        }
    )
    return {
        "protocol": "continuous_state_over_15_corruptions_repeated_3_cycles",
        "state_continuity": True,
        "cycles": CONTINUAL_CYCLES,
        "summary": overall_summary,
        "resource": resource,
        "restoration": restoration_stats_for_output(args, adapter.restorer),
        "cola": cola_diagnostics,
        "per_corruption": results,
        "post_stream": post_stream,
    }, metadata
