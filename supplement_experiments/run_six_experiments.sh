#!/usr/bin/env bash
set -Eeuo pipefail

# Dispatch one of the six independent launchers.  Arguments after the number
# are passed unchanged, which keeps paths and GPU options shell-safe.
cd "$(dirname "$0")/.."
case "${1:-}" in
  1) script="supplement_experiments/exp1_tiny_imagenet_c.py" ;;
  2) script="supplement_experiments/exp2_resnet34_cifar100c.py" ;;
  3) script="supplement_experiments/exp3_fl_initialization.py" ;;
  4) script="supplement_experiments/exp4_component_ablation.py" ;;
  5) script="supplement_experiments/exp5_coordinate_ablation.py" ;;
  6) script="supplement_experiments/exp6_continual_curve.py" ;;
  *)
    echo "Usage: bash supplement_experiments/run_six_experiments.sh 1|2|3|4|5|6 [options]" >&2
    exit 2
    ;;
esac
shift
exec "${PYTHON:-python}" "$script" "$@"
