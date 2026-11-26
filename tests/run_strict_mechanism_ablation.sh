#!/usr/bin/env bash
set -Eeuo pipefail

# Strict 2x2 mechanism ablation of the FedDR main experiment on one fixed
# FedACG lookahead checkpoint.
# Base=unweighted regional adaptation, DLRS=Base+DLRS,
# FSLR=Base+FSLR, DLRS+FSLR=FedDR.
# The Python entrypoint fixes SGD/lr/batch protocol, the strict flag fixes the
# all-BN-affine adaptation scope, and clip=0 isolates restoration from clipping.
cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1

CHECKPOINT="train/results/cifar100_100clients_alpha0.3_equal/fedacg_resnet18/cifar100_resnet18_run1/model_round499.pt"
STATISTICS_PATH="test/outputs/fedacg_run1_statistics.pt"
OUTPUT_DIR="test/outputs/strict_feddr_ablation/norm_product_nogate_s07_i10_r010_lri0_run1"
LOG_DIR="$OUTPUT_DIR/logs"

if [[ ! -s "$CHECKPOINT" ]]; then
  echo "Missing checkpoint: $CHECKPOINT" >&2
  exit 1
fi
if [[ ! -s "$STATISTICS_PATH" ]]; then
  echo "Missing statistics: $STATISTICS_PATH" >&2
  exit 1
fi

mkdir -p "$LOG_DIR"

run_one() {
  local label="$1"
  local tta_method="$2"
  local output_path="$OUTPUT_DIR/${label}.json"
  local log_path="$LOG_DIR/${label}.log"

  if [[ -s "$output_path" ]]; then
    echo "Skipping completed $label: $output_path"
    return
  fi

  echo "[strict-ablation] starting $label ($tta_method)"
  python main_test.py \
    --dataset cifar100 \
    --federated_method fedacg \
    --model_path "$CHECKPOINT" \
    --anchor_key lookahead_anchor \
    --tta_method "$tta_method" \
    --strict_mechanism_ablation \
    --statistics_path "$STATISTICS_PATH" \
    --cifar_c_path data/CIFAR-100-C \
    --clean_data_root data \
    --device cuda \
    --num_workers 2 \
    --fslr_grad_clip_norm 0.0 \
    --recovery_interval 10 \
    --restore_strength 0.7 \
    --restore_budget_mode drift_tiered \
    --restore_drift_thresholds 0.10 0.20 0.35 \
    --restore_drift_ratios 0.0 0.02 0.05 0.10 \
    --restore_reliability_threshold 0.55 \
    --restore_reliability_ema_decay 0.90 \
    --seed 0 \
    --max_batches -1 \
    --output_path "$output_path" \
    >"$log_path" 2>&1

  if [[ ! -s "$output_path" ]]; then
    echo "Run finished without output: $output_path" >&2
    exit 1
  fi
  echo "[strict-ablation] completed $label: $output_path"
}

run_one base base
run_one dlrs dlrs
run_one fslr base_fslr
run_one feddr feddr
