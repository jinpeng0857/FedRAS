#!/usr/bin/env bash
set -Eeuo pipefail

# Primary TTA comparison roster under the matched FedDR protocol: Source /
# Tent / EATA / ASR / SAR / CoTTA / RoTTA / PeTTA / ReCAP / FedDR.
# Method-specific scopes,
# losses, and controllers remain unchanged; batch size, SGD optimizer, learning
# rate, and FSLR settings are shared across all runnable comparisons.
cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES=0
FORCE_RERUN="${FORCE_RERUN:-0}"

skip_completed() {
  local output_path="$1"
  if [[ "$FORCE_RERUN" != "1" && -s "$output_path" ]]; then
    echo "Skipping completed result: $output_path"
    return 0
  fi
  return 1
}

statistics_match() {
  local statistics_path="$1"
  local checkpoint="$2"
  local fisher_mode="$3"
  python - "$statistics_path" "$checkpoint" "$fisher_mode" <<'PY'
import sys
from pathlib import Path

import torch

statistics_path, checkpoint, fisher_mode = sys.argv[1:]
try:
    payload = torch.load(statistics_path, map_location="cpu", weights_only=False)
    meta = payload.get("meta", {})
    recorded_model = Path(meta.get("model_path", "")).expanduser().resolve()
    expected_model = Path(checkpoint).expanduser().resolve()
    if recorded_model != expected_model:
        raise SystemExit(1)
    if meta.get("fisher_mode") != fisher_mode:
        raise SystemExit(1)
    # PeTTA consumes class-conditional moments added by the unified
    # statistics preparation path.  Older files must be regenerated.
    if "class_feature_mean" not in payload or "class_feature_variance" not in payload:
        raise SystemExit(1)
except Exception:
    raise SystemExit(1)
PY
}

prepare_statistics() {
  if [[ "$FORCE_RERUN" != "1" && -s "$3" ]] \
    && statistics_match "$3" "$1" main; then
    echo "Skipping completed matching statistics: $3"
    return
  fi
  python prepare_main_statistics.py \
    --dataset cifar100 \
    --model_path "$1" \
    --anchor_key "$2" \
    --data_root data \
    --feature_samples 500 \
    --fisher_max_batches -1 \
    --fisher_mode main \
    --fisher_samples -1 \
    --batch_size 128 \
    --num_workers 2 \
    --seed 0 \
    --device cuda \
    --output_path "$3"
}

prepare_eata_statistics() {
  if [[ "$FORCE_RERUN" != "1" && -s "$3" ]] \
    && statistics_match "$3" "$1" eata_official; then
    echo "Skipping completed matching EATA statistics: $3"
    return
  fi
  python prepare_main_statistics.py \
    --dataset cifar100 \
    --model_path "$1" \
    --anchor_key "$2" \
    --data_root data \
    --feature_samples 500 \
    --fisher_max_batches -1 \
    --fisher_mode eata_official \
    --fisher_samples 2000 \
    --batch_size 128 \
    --num_workers 2 \
    --seed 0 \
    --device cuda \
    --output_path "$3"
}

run_source_only() {
  if skip_completed "$3"; then
    return
  fi
  python -m test.source_only_test \
    --dataset cifar100 \
    --model_path "$1" \
    --anchor_key "$2" \
    --cifar_c_path data/CIFAR-100-C \
    --clean_data_root data \
    --device cuda \
    --batch_size 128 \
    --num_workers 2 \
    --seed 0 \
    --max_batches -1 \
    --output_path "$3"
}

run_tta() {
  if skip_completed "$6"; then
    return
  fi
  python main_test.py \
    --dataset cifar100 \
    --federated_method "$1" \
    --model_path "$2" \
    --anchor_key "$3" \
    --tta_method "$4" \
    --statistics_path "$5" \
    --cifar_c_path data/CIFAR-100-C \
    --clean_data_root data \
    --device cuda \
    --num_workers 2 \
    --seed 0 \
    --max_batches -1 \
    --output_path "$6" \
    "${@:7}"
}

run_fedacg_comparisons() {
  local federated_method="fedacg"
  local checkpoint="$1"
  local anchor_key="$2"
  local statistics_path="$3"
  local eata_statistics_path="$4"
  local output_dir="$5"

  prepare_statistics "$checkpoint" "$anchor_key" "$statistics_path"
  prepare_eata_statistics "$checkpoint" "$anchor_key" "$eata_statistics_path"

  run_source_only "$checkpoint" "$anchor_key" \
    "$output_dir/cifar100_source.json"

  # Matched FedDR optimization protocol; Tent keeps its entropy objective.
  run_tta "$federated_method" "$checkpoint" "$anchor_key" tent \
    "$statistics_path" "$output_dir/cifar100_${federated_method}_tent.json"

  # EATA keeps its selection thresholds and 2,000-sample pseudo-label Fisher.
  run_tta "$federated_method" "$checkpoint" "$anchor_key" eata \
    "$eata_statistics_path" "$output_dir/cifar100_${federated_method}_eata.json" \
    --eata_e_margin_factor 0.4 \
    --eata_d_margin 0.05 \
    --eata_fisher_alpha 2000.0

  # ASR keeps its ETA entropy objective and entropy-triggered reset controller.
  run_tta "$federated_method" "$checkpoint" "$anchor_key" asr \
    "$statistics_path" "$output_dir/cifar100_${federated_method}_asr.json"

  # SAR keeps its norm-affine scope and SAM wrapper with an SGD base optimizer.
  run_tta "$federated_method" "$checkpoint" "$anchor_key" sar \
    "$statistics_path" "$output_dir/cifar100_${federated_method}_sar.json" \
    --sar_margin_factor 0.4 \
    --sar_reset_threshold 0.2 \
    --sar_sam_rho 0.05

  # Unified RoTTA: robust BN, class-balanced memory and EMA teacher.
  run_tta "$federated_method" "$checkpoint" "$anchor_key" rotta \
    "$statistics_path" "$output_dir/cifar100_${federated_method}_rotta.json" \
    --rotta_memory_size 64 \
    --rotta_update_frequency 64 \
    --rotta_nu 0.001 \
    --rotta_bn_alpha 0.05

  # Unified PeTTA: class-conditional moments are read from the prepared
  # statistics file; no source images are accessed during this evaluation.
  run_tta "$federated_method" "$checkpoint" "$anchor_key" petta \
    "$statistics_path" "$output_dir/cifar100_${federated_method}_petta.json" \
    --rotta_memory_size 64 \
    --rotta_update_frequency 64 \
    --rotta_nu 0.001 \
    --rotta_bn_alpha 0.05 \
    --petta_lambda 10.0 \
    --petta_anchor_weight 1.0 \
    --petta_alpha 0.001

  # REM is intentionally not invoked here: its official implementation
  # requires a 384x384 custom ViT with attention/token-pruning support, while
  # this comparison script is locked to the matched CIFAR-100-C ResNet18
  # protocol.  It must be evaluated in a separate ViT experiment.

  # FedDR = DLRS + FSLR with normalized product weighting and FSLR no-gate.
  run_tta "$federated_method" "$checkpoint" "$anchor_key" feddr \
    "$statistics_path" "$output_dir/cifar100_feddr.json" \
    --dlrs_normalize_by_weight \
    --dlrs_weight_mode product \
    --disable_fslr_reliability_gate \
    --fslr_grad_clip_norm 15.0 \
    --recovery_interval 10 \
    --restore_strength 0.7 \
    --restore_budget_mode drift_tiered \
    --restore_drift_thresholds 0.10 0.20 0.35 \
    --restore_drift_ratios 0.0 0.02 0.05 0.10 \
    --restore_reliability_threshold 0.55 \
    --restore_reliability_ema_decay 0.90

  # DeYO has no official CIFAR-100-C recipe; use the official portable core.
  run_tta "$federated_method" "$checkpoint" "$anchor_key" deyo \
    "$statistics_path" "$output_dir/cifar100_${federated_method}_deyo.json" \
    --deyo_entropy_margin_factor 0.5 \
    --deyo_reweight_margin_factor 0.4 \
    --deyo_plpd_threshold 0.2 \
    --deyo_patch_grid 4

  run_tta "$federated_method" "$checkpoint" "$anchor_key" recap \
    "$statistics_path" "$output_dir/cifar100_${federated_method}_recap.json"
}

# FedACG run1/run2/run3: matched checkpoints for the current comparison subset
# and FedDR main experiment. All listed methods use the same matched protocol.
run_fedacg_comparisons \
  train/results/cifar100_100clients_alpha0.3_equal/fedacg_resnet18/cifar100_resnet18_run1/model_round499.pt \
  lookahead_anchor \
  test/outputs/fedacg_run1_statistics.pt \
  test/outputs/fedacg_run1_eata_statistics.pt \
  test/outputs/fedacg_run1

run_fedacg_comparisons \
  train/results/cifar100_100clients_alpha0.3_equal/fedacg_resnet18/run2/model_round499.pt \
  lookahead_anchor \
  test/outputs/fedacg_run2_statistics.pt \
  test/outputs/fedacg_run2_eata_statistics.pt \
  test/outputs/fedacg_run2

run_fedacg_comparisons \
  train/results/cifar100_100clients_alpha0.3_equal/fedacg_resnet18/run3/model_round499.pt \
  lookahead_anchor \
  test/outputs/fedacg_run3_statistics.pt \
  test/outputs/fedacg_run3_eata_statistics.pt \
  test/outputs/fedacg_run3
