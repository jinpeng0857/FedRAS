#!/usr/bin/env bash
set -euo pipefail

ROOT=/mnt/sda/PythonProject/CJP_project/FEDEATA_revision_20261018
DATA=/mnt/sda/PythonProject/Jiang/dataset
TRAIN=/mnt/sda/PythonProject/CJP_project/FEDEATA/train/results/tiny-imagenet_100clients_alpha0.3_equal
ASSET_ROOT=/mnt/sda/PythonProject/CJP_project/FEDEATA
OUT="$ROOT/test/outputs/tiny_cache_replay_orderfix_retry1"
PY=/home/ubuntu/anaconda3/envs/vllm/bin/python
RUN_CASE="$ROOT/scripts/revision_20261018/run_case.py"
mkdir -p "$OUT"

export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

run_a() {
  local backbone="$1" model="$2" stats="$3"
  "$PY" -u "$ROOT/scripts/unified_tta_eval_remote.py" feddr "$ROOT" "$OUT" \
    --dataset tiny-imagenet \
    --cifar_c_path "$DATA/Tiny-ImageNet-C" \
    --clean_data_root "$DATA/tiny-imagenet-200" \
    --tta_method feddr --federated_method fedacg --anchor_key lookahead_anchor \
    --model_path "$model" --statistics_path "$stats" \
    --batch_size 128 --adapt_lr 0.001 --num_workers 2 --seed 0 --max_batches -1 \
    --device cuda:0 --output_path "$OUT/scheme_a_${backbone}.json"
}

run_b() {
  local backbone="$1" model="$2" stats="$3"
  "$PY" -u "$ROOT/scripts/run_tiny_ordered_revision_case.py" "$RUN_CASE" "$OUT" \
    --revision-controller disagreement --revision-ranking drift_only \
    --dataset tiny-imagenet \
    --cifar_c_path "$DATA/Tiny-ImageNet-C" \
    --clean_data_root "$DATA/tiny-imagenet-200" \
    --tta_method feddr --federated_method fedacg --anchor_key lookahead_anchor \
    --model_path "$model" --statistics_path "$stats" \
    --batch_size 128 --adapt_lr 0.001 --num_workers 2 --seed 0 --max_batches -1 \
    --fslr_selection_criterion drift_only --dlrs_ablation_variant anchor_only \
    --dlrs_hybrid_threshold 0.0 --dlrs_weight_mode sample_only \
    --disable_dlrs_view_consistency --disable_fslr_event_trigger \
    --device cuda:0 --output_path "$OUT/scheme_b_${backbone}.json"
}

run_a rn34 \
  "$TRAIN/fedacg_resnet34/tiny_imagenet_resnet34_cifar100_protocol/model_round499.pt" \
  "$ASSET_ROOT/test/outputs/tiny_resnet34_unified_20260907/statistics.pt" \
  > "$OUT/scheme_a_rn34.log" 2>&1
echo "Completed scheme A RN34"

run_a rn18 \
  "$TRAIN/fedacg_resnet18/tiny_imagenet_resnet18_fedacg_20260907_seed42/model_round499.pt" \
  "$ASSET_ROOT/test/outputs/tiny_resnet18_unified_20260907/statistics.pt" \
  > "$OUT/scheme_a_rn18.log" 2>&1
echo "Completed scheme A RN18"

run_b rn34 \
  "$TRAIN/fedacg_resnet34/tiny_imagenet_resnet34_cifar100_protocol/model_round499.pt" \
  "$ASSET_ROOT/test/outputs/tiny_resnet34_unified_20260907/statistics.pt" \
  > "$OUT/scheme_b_rn34.log" 2>&1
echo "Completed scheme B RN34"

run_b rn18 \
  "$TRAIN/fedacg_resnet18/tiny_imagenet_resnet18_fedacg_20260907_seed42/model_round499.pt" \
  "$ASSET_ROOT/test/outputs/tiny_resnet18_unified_20260907/statistics.pt" \
  > "$OUT/scheme_b_rn18.log" 2>&1
echo "Completed scheme B RN18"
