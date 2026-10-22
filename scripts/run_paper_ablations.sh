#!/usr/bin/env bash
set -Eeuo pipefail

# ==============================================================================
# FedDR Paper Ablation Suite (Blocks A, B, and C)
# 
# Usage:
#   bash scripts/run_paper_ablations.sh [A|B|C|all] [device]
# 
# Example:
#   bash scripts/run_paper_ablations.sh all cuda:0
#   bash scripts/run_paper_ablations.sh A cuda:0
# ==============================================================================

cd "$(dirname "$0")/.."

BLOCKS="${1:-all}"
DEVICE="${2:-cuda}"
OUTPUT_DIR="test/outputs/paper_ablations"

echo "=================================================================="
echo " Starting FedDR Paper Ablation Suite"
echo " Blocks:     $BLOCKS"
echo " Device:     $DEVICE"
echo " Output Dir: $OUTPUT_DIR"
echo "=================================================================="

python scripts/run_paper_ablations.py \
    --blocks "$BLOCKS" \
    --device "$DEVICE" \
    --output_dir "$OUTPUT_DIR" \
    --cifar_c_path data/CIFAR-100-C \
    --clean_data_root data \
    --num_workers 2

echo "=================================================================="
echo " Generating Summary Table and LaTeX snippet..."
echo "=================================================================="

python scripts/summarize_paper_ablations.py \
    --output_dir "$OUTPUT_DIR" \
    --latex_out "$OUTPUT_DIR/table3_ablation.tex" \
    --strict

echo "Done! Summary available at: $OUTPUT_DIR/table3_ablation.tex"
