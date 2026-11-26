#!/usr/bin/env bash
set -Eeuo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-/home/ubuntu/anaconda3/envs/vllm/bin/python}"
OUT="${OUT:-test/outputs/cifar100_resnet34_unified_20260910}"
MODEL="${MODEL:-train/results/cifar100_100clients_alpha0.3_equal/fedacg_resnet34/run1_match/model_round499.pt}"
SEED="${SEED:-0}"
STATS="$OUT/statistics.pt"
EATA_STATS="$OUT/eata_statistics.pt"
COMMON=(
  --dataset cifar100
  --model_path "$MODEL"
  --anchor_key lookahead_anchor
  --cifar_c_path data/CIFAR-100-C
  --clean_data_root data
  --device cuda
  --batch_size 128
  --num_workers 2
  --seed "$SEED"
  --max_batches -1
)
TTA_COMMON=(--federated_method fedacg "${COMMON[@]}")

mkdir -p "$OUT"

"$PYTHON" prepare_main_statistics.py \
  --dataset cifar100 \
  --model_path "$MODEL" \
  --anchor_key lookahead_anchor \
  --data_root data \
  --feature_samples 500 \
  --fisher_max_batches -1 \
  --fisher_mode main \
  --fisher_samples -1 \
  --batch_size 128 \
  --num_workers 2 \
  --seed "$SEED" \
  --device cuda \
  --output_path "$STATS"

"$PYTHON" prepare_main_statistics.py \
  --dataset cifar100 \
  --model_path "$MODEL" \
  --anchor_key lookahead_anchor \
  --data_root data \
  --feature_samples 500 \
  --fisher_max_batches -1 \
  --fisher_mode eata_official \
  --fisher_samples 2000 \
  --batch_size 128 \
  --num_workers 2 \
  --seed "$SEED" \
  --device cuda \
  --output_path "$EATA_STATS"

"$PYTHON" -m test.source_only_test \
  "${COMMON[@]}" \
  --output_path "$OUT/source.json"

"$PYTHON" main_test.py \
  "${TTA_COMMON[@]}" \
  --tta_method tent \
  --statistics_path "$STATS" \
  --output_path "$OUT/tent.json"

"$PYTHON" main_test.py \
  "${TTA_COMMON[@]}" \
  --tta_method eata \
  --statistics_path "$EATA_STATS" \
  --eata_e_margin_factor 0.4 \
  --eata_d_margin 0.05 \
  --eata_fisher_alpha 2000.0 \
  --output_path "$OUT/eata.json"

"$PYTHON" main_test.py \
  "${TTA_COMMON[@]}" \
  --tta_method sar \
  --statistics_path "$STATS" \
  --sar_margin_factor 0.4 \
  --sar_reset_threshold 0.2 \
  --sar_sam_rho 0.05 \
  --output_path "$OUT/sar.json"

"$PYTHON" main_test.py \
  "${TTA_COMMON[@]}" \
  --tta_method asr \
  --statistics_path "$STATS" \
  --output_path "$OUT/asr.json"

"$PYTHON" main_test.py \
  "${TTA_COMMON[@]}" \
  --tta_method recap \
  --statistics_path "$STATS" \
  --output_path "$OUT/recap.json"

"$PYTHON" main_test.py \
  "${TTA_COMMON[@]}" \
  --tta_method feddr \
  --statistics_path "$STATS" \
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
  --restore_reliability_ema_decay 0.90 \
  --output_path "$OUT/feddr.json"

echo "CIFAR100_RESNET34_COMPLETE $OUT"
