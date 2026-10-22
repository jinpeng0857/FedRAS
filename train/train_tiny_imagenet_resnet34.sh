#!/usr/bin/env bash
set -Eeuo pipefail

# Usage:
#   DATASET_PATH=/path/to/tiny-imagenet-200 DEVICE=cuda:0 \
#     bash train/train_tiny_imagenet_resnet34.sh
#
# The dataset path must contain train/<wnid>/... and val/.  The training code
# reads the official flat validation layout directly; it never reorganizes it.
cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-/home/ubuntu/anaconda3/envs/vllm/bin/python}"
DATASET_PATH="${DATASET_PATH:?Set DATASET_PATH to the tiny-imagenet-200 directory}"
DEVICE="${DEVICE:-cuda:0}"

exec "$PYTHON_BIN" -m train.train_fedacg \
  --method fedacg \
  --fedacg_mode fedacg \
  --dataset tiny-imagenet \
  --dataset_path "$DATASET_PATH" \
  --backbone resnet34 \
  --num_client "${NUM_CLIENTS:-100}" \
  --join_ratio "${JOIN_RATIO:-0.15}" \
  --alpha "${ALPHA:-0.3}" \
  --equal_size \
  --global_rounds "${ROUNDS:-500}" \
  --local_epochs "${LOCAL_EPOCHS:-3}" \
  --batch_size "${BATCH_SIZE:-64}" \
  --num_workers "${NUM_WORKERS:-2}" \
  --learning_rate "${LR:-0.05}" \
  --momentum 0.9 \
  --weight_decay 0.0005 \
  --local_lr_decay 0.998 \
  --lr_scheduler cosine \
  --label_smoothing 0.1 \
  --use_randaugment \
  --randaugment_n 2 \
  --randaugment_m 9 \
  --server_momentum 0.9 \
  --ema_decay 0.999 \
  --eval_clean_every "${EVAL_CLEAN_EVERY:-50}" \
  --save_best_clean \
  --save_freq "${SAVE_FREQ:-50}" \
  --seed "${SEED:-42}" \
  --dataset_seed "${DATASET_SEED:-21}" \
  --new_dataset_seed 30 \
  --task_name "${TASK_NAME:-tiny_imagenet_resnet34}" \
  --device "$DEVICE" \
  --use_bn_layer \
  --amp
