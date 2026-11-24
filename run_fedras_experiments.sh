#!/usr/bin/env bash
set -euo pipefail

# Tables refer to FEDRAS (2).pdf. Use --dry-run to print the resolved commands.
usage() {
  cat <<'USAGE'
Usage: bash run_fedras_experiments.sh {components|appendix-f|random|fixed|cost|sensitivity|all} [--dry-run]
  components    Tables 6, 7, 12 and 13, three checkpoints
  appendix-f    Table 13 only, run1
  random        Table 8, displacement-reference recording and matched random replay
  fixed         Table 8, fixed rho 0.02/0.05/0.10, three checkpoints
  cost          Table 9, run1, three measured runs with in-process warm-up
  sensitivity   Table 11, six non-default points on run1
  all           All formal groups, sharing the component trajectories for Table 13
Overrides: REVISION_ROOT, ASSET_ROOT, OUTPUT_ROOT, PYTHON_BIN, DEVICE, DATA_ROOT,
           CLEAN_ROOT, CKPT1..3, COV_PATH1..3, FIXED_STRENGTH, SELECTION_SEED
USAGE
}

stage=${1:-}
case "$stage" in
  -h|--help) usage; exit 0 ;;
  components|appendix-f|random|fixed|cost|sensitivity|all) ;;
  *) usage >&2; exit 2 ;;
esac
dry_run=false
if [[ ${2:-} == --dry-run ]]; then dry_run=true; fi
if (( $# > 2 )) || { (( $# == 2 )) && ! "$dry_run"; }; then
  usage >&2
  exit 2
fi

root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd -- "$root"
revision_root=${REVISION_ROOT:-"$root"}
asset_root=${ASSET_ROOT:-"$root"}
output_root=${OUTPUT_ROOT:-"$root/output/fedras_pdf_supplement"}
python_bin=${PYTHON_BIN:-python}
device=${DEVICE:-cuda:0}
data_root=${DATA_ROOT:-"$asset_root/data/CIFAR-100-C"}
clean_root=${CLEAN_ROOT:-"$asset_root/data/cifar100"}
fixed_strength=${FIXED_STRENGTH:-0.7}
selection_seed=${SELECTION_SEED:-2026}
runner="$revision_root/scripts/revision_20261018/run_case.py"
model_root="$asset_root/train/results/cifar100_100clients_alpha0.3_equal/fedacg_resnet18"
checkpoints=("" "${CKPT1:-"$model_root/cifar100_resnet18_run1/model_round499.pt"}"
  "${CKPT2:-"$model_root/run2/model_round499.pt"}" "${CKPT3:-"$model_root/run3/model_round499.pt"}")
statistics=("" "${COV_PATH1:-"$asset_root/test/outputs/fedacg_run1_statistics.pt"}"
  "${COV_PATH2:-"$asset_root/test/outputs/fedacg_run2_statistics.pt"}"
  "${COV_PATH3:-"$asset_root/test/outputs/fedacg_run3_statistics.pt"}")
indices=(1)
case "$stage" in components|random|fixed|all) indices=(1 2 3) ;; esac

if [[ "$stage" == cost || "$stage" == all ]]; then
  if [[ "$device" != cuda && ! "$device" =~ ^cuda:[0-9]+$ ]]; then
    printf 'Table 9 requires a CUDA device. DEVICE=%s\n' "$device" >&2
    exit 2
  fi
fi

if ! "$dry_run"; then
  inputs=("$runner" "$data_root/labels.npy")
  for split in train test meta; do inputs+=("$clean_root/cifar-100-python/$split"); done
  corruptions=(gaussian_noise shot_noise impulse_noise defocus_blur glass_blur motion_blur
    zoom_blur snow frost fog brightness contrast elastic_transform pixelate jpeg_compression)
  for corruption in "${corruptions[@]}"; do inputs+=("$data_root/$corruption.npy"); done
  for index in "${indices[@]}"; do inputs+=("${checkpoints[index]}" "${statistics[index]}"); done
  missing=()
  for input in "${inputs[@]}"; do
    [[ -f "$input" && -r "$input" ]] || missing+=("$input")
  done
  if (( ${#missing[@]} )); then
    printf 'Missing or unreadable input: %s\n' "${missing[@]}" >&2
    exit 2
  fi
  if [[ "$stage" == cost || "$stage" == all ]]; then
    "$python_bin" -c '
import sys
import torch

device = torch.device(sys.argv[1])
index = device.index if device.index is not None else 0
if not torch.cuda.is_available() or index >= torch.cuda.device_count():
    sys.exit(f"Table 9 requires an available CUDA device. DEVICE={device}")
' "$device"
  fi
fi

common=(
  --dataset cifar100 --federated_method fedacg --anchor_key lookahead_anchor
  --tta_method dlrs_fslr --revision-ranking drift_only --review_continual_only --device "$device"
  --cifar_c_path "$data_root" --clean_data_root "$clean_root"
  --batch_size 128 --seed 0 --num_workers 2 --max_batches -1 --adapt_lr 0.001
  --dlrs_sample_reliability_mode entropy --dlrs_weight_mode sample_only
  --dlrs_hybrid_threshold 0 --dlrs_anchor_rectify_beta 0.3 --disable_dlrs_view_consistency
  --fslr_grad_clip_norm 15 --disable_fslr_reliability_gate --fslr_hierarchical_budget
  --fslr_layer_quota_cap 0.35 --fslr_selection_criterion drift_only
)
# Zero ESS coefficient is equivalent to setting controller_ess=1 in the
# original disagreement wrapper. This also avoids wrapper overhead in Table 9.
adaptive=(
  --fslr_continuous_risk --fslr_risk_gamma_ess 0 --fslr_risk_gamma_anchor 0.5
  --fslr_risk_rho_min 0 --fslr_risk_rho_max 0.10 --fslr_risk_tau_drift 0.35
  --recovery_interval 10 --restore_strength 0.7 --fslr_event_trigger
)

run_trial() (
  local index=$1 name=$2
  shift 2
  local trial_dir="$output_root/$name"
  local command=("$python_bin" -u "$runner" "${common[@]}"
    --model_path "${checkpoints[index]}" --statistics_path "${statistics[index]}"
    --output_path "$trial_dir/result.json" "$@")
  if "$dry_run"; then
    printf '%q ' "${command[@]}"
    printf '\n'
    return
  fi
  local lock="$trial_dir.lock" archive invocation
  printf -v invocation '%q ' "${command[@]}"
  mkdir -p -- "$(dirname -- "$trial_dir")"
  if ! mkdir -- "$lock"; then
    printf 'Cannot acquire trial lock: %s\n' "$lock" >&2
    return 1
  fi
  trap 'rmdir -- "$lock"' EXIT
  if [[ -d "$trial_dir" ]]; then
    if [[ -f "$trial_dir/complete" && -s "$trial_dir/result.json" ]]; then
      if [[ "$(cat -- "$trial_dir/complete")" != "$invocation" ]]; then
        printf 'Trial already completed with different arguments: %s. Use another OUTPUT_ROOT.\n' "$trial_dir" >&2
        return 1
      fi
      printf 'Skipping completed trial: %s\n' "$name"
      return
    fi
    archive=$(mktemp -d "$trial_dir.previous.XXXXXX")
    mv -- "$trial_dir" "$archive/trial"
    printf 'Preserved previous trial: %s/trial\n' "$archive"
  fi
  mkdir -- "$trial_dir"
  "${command[@]}" 2>&1 | tee "$trial_dir/run.log"
  printf '%s\n' "$invocation" > "$trial_dir/complete"
)

run_components() {
  local index=$1 group=$2
  local dsr="$group/run$index/dsr_only" full="$group/run$index/fedras"
  local dsr_diagnostics=(--review_diagnostics_path "$output_root/$dsr/diagnostics.jsonl" --review_run_id "$dsr" --review_beta_a 0)
  local full_diagnostics=(--review_diagnostics_path "$output_root/$full/diagnostics.jsonl" --review_run_id "$full" --review_beta_a 0.3)
  run_trial "$index" "$dsr" "${adaptive[@]}" --dlrs_ablation_variant uniform_recap \
    --anchor_disagreement_diagnostic "${dsr_diagnostics[@]}"
  run_trial "$index" "$full" "${adaptive[@]}" --dlrs_ablation_variant anchor_only "${full_diagnostics[@]}"
}

if [[ "$stage" == components || "$stage" == all ]]; then
  for index in "${indices[@]}"; do run_components "$index" components; done
fi
if [[ "$stage" == appendix-f ]]; then run_components 1 appendix-f; fi

if [[ "$stage" == random || "$stage" == all ]]; then
  for index in "${indices[@]}"; do
    schedule="$output_root/random/run$index/reference/schedule.pt"
    run_trial "$index" "random/run$index/reference" "${adaptive[@]}" \
      --dlrs_ablation_variant anchor_only --revision-schedule-record "$schedule"
    run_trial "$index" "random/run$index/matched" "${adaptive[@]}" \
      --dlrs_ablation_variant anchor_only --revision-ranking random \
      --revision-selection-seed "$selection_seed" --revision-schedule-replay "$schedule"
  done
fi

if [[ "$stage" == fixed || "$stage" == all ]]; then
  for index in "${indices[@]}"; do
    for rho in 0.02 0.05 0.10; do
      # Keep a constant pullback coefficient and use periodic restoration only.
      run_trial "$index" "fixed/run$index/rho_$rho" --dlrs_ablation_variant anchor_only \
        --disable_fslr_event_trigger --disable_fslr_continuous_risk --recovery_interval 10 \
        --restore_budget_mode fixed --sparse_ratio "$rho" --restore_strength "$fixed_strength"
    done
  done
fi

if [[ "$stage" == cost || "$stage" == all ]]; then
  for repeat in repeat1 repeat2 repeat3; do
    run_trial 1 "cost/$repeat" "${adaptive[@]}" --dlrs_ablation_variant anchor_only --revision-warmup
  done
fi

if [[ "$stage" == sensitivity || "$stage" == all ]]; then
  for interval in 5 20; do
    run_trial 1 "sensitivity/R_$interval" "${adaptive[@]}" --dlrs_ablation_variant anchor_only \
      --recovery_interval "$interval"
  done
  for tau in 0.25 0.50; do
    run_trial 1 "sensitivity/tauD_$tau" "${adaptive[@]}" --dlrs_ablation_variant anchor_only \
      --fslr_risk_tau_drift "$tau"
  done
  for rho in 0.05 0.20; do
    run_trial 1 "sensitivity/rhoMax_$rho" "${adaptive[@]}" --dlrs_ablation_variant anchor_only \
      --fslr_risk_rho_max "$rho"
  done
fi
