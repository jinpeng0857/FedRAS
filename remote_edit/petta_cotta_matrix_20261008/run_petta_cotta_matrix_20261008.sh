#!/usr/bin/env bash
set -uo pipefail

PROJECT_ROOT=/mnt/sda/PythonProject/CJP_project/FEDEATA
cd "$PROJECT_ROOT"
source /home/ubuntu/anaconda3/etc/profile.d/conda.sh
conda activate vllm
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

RUN_ROOT="$PROJECT_ROOT/artifacts/petta_cotta_matrix_20261008"
RESULT_ROOT="$RUN_ROOT/results"
LOG_ROOT="$RUN_ROOT/logs"
STATS_ROOT="$RUN_ROOT/statistics"
mkdir -p "$RESULT_ROOT" "$LOG_ROOT" "$STATS_ROOT"
STATUS_FILE="$RUN_ROOT/status.tsv"
if [[ ! -e "$STATUS_FILE" ]]; then
  printf 'time\tdataset\tcheckpoint\tmethod\tstatus\tdetail\n' > "$STATUS_FILE"
fi

CIFAR_C="$PROJECT_ROOT/data/CIFAR-100-C"
CLEAN_CIFAR="$PROJECT_ROOT/data"
TINY_C=/mnt/sda/PythonProject/Jiang/dataset/Tiny-ImageNet-C
CLEAN_TINY=/mnt/sda/PythonProject/Jiang/dataset/tiny-imagenet-200

C100_CKPTS=(
  "$PROJECT_ROOT/train/results/cifar100_100clients_alpha0.3_equal/fedacg_resnet18/cifar100_resnet18_run1/model_round499.pt"
  "$PROJECT_ROOT/train/results/cifar100_100clients_alpha0.3_equal/fedacg_resnet18/run2/model_round499.pt"
  "$PROJECT_ROOT/train/results/cifar100_100clients_alpha0.3_equal/fedacg_resnet18/run3/model_round499.pt"
)
C100_STATS=(
  "$PROJECT_ROOT/test/outputs/fedacg_run1_statistics.pt"
  "$PROJECT_ROOT/test/outputs/fedacg_run2_statistics.pt"
  "$PROJECT_ROOT/test/outputs/fedacg_run3_statistics.pt"
)
C100_NAMES=(fedacg_run1 fedacg_run2 fedacg_run3)

TINY_CKPTS=(
  "$PROJECT_ROOT/train/results/tiny-imagenet_100clients_alpha0.3_equal/fedacg_resnet18/tiny_imagenet_resnet18_fedacg_20260907_seed42/model_round499.pt"
  "$PROJECT_ROOT/train/results/tiny-imagenet_100clients_alpha0.3_equal/fedacg_resnet34/tiny_imagenet_resnet34_cifar100_protocol/model_round499.pt"
  "$PROJECT_ROOT/train/results/tiny-imagenet_20clients_alpha0.3_equal/fedacg_resnet34/tiny_imagenet_resnet34_run1/best_clean.pt"
)
TINY_NAMES=(fedacg_100c_resnet18_seed42 fedacg_100c_resnet34_cifar100protocol fedacg_20c_resnet34_run1_best_clean)

record() {
  local dataset="$1" checkpoint="$2" method="$3" status="$4" detail="$5"
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$(date -Is)" "$dataset" "$checkpoint" "$method" "$status" "$detail" >> "$STATUS_FILE"
}

validate_result() {
  local output="$1" method="$2" dataset="$3" checkpoint="$4" stats="$5"
  python - "$output" "$method" "$dataset" "$checkpoint" "$stats" <<'PY'
import json, sys
p, method, dataset, checkpoint, stats = sys.argv[1:]
r = json.load(open(p, encoding='utf-8'))
assert r['method'].lower() == method, (r.get('method'), method)
assert r['dataset'] == dataset, (r.get('dataset'), dataset)
assert r['model_path'] == checkpoint, (r.get('model_path'), checkpoint)
assert r['statistics_path'] == stats, (r.get('statistics_path'), stats)
domain = r['domain_wise']['per_corruption']
stream = r['continual']['per_corruption']
assert len(domain) == 15, len(domain)
assert len(stream) == 45, len(stream)
expected = 10000
for block in domain + stream:
    assert block['severity'] == 5
    assert block['total'] == expected, (block.get('corruption'), block.get('cycle'), block['total'], expected)
    assert block['batches'] > 0
    assert block['nonfinite_batches'] == 0
    assert block['nonfinite_gradient_batches'] == 0
assert {x['cycle'] for x in stream} == {1, 2, 3}
assert r['config']['max_batches'] == -1
print(f"VALID method={method} dataset={dataset} domain={len(domain)}/15 stream={len(stream)}/45 samples={sum(x['total'] for x in domain + stream)}")
PY
}

ensure_tiny_statistics() {
  local checkpoint="$1"
  local name="$2"
  local output="$STATS_ROOT/${name}_statistics.pt"
  if [[ -s "$output" ]]; then
    if python - "$output" "$checkpoint" <<'PY'
import sys, torch
p = torch.load(sys.argv[1], map_location='cpu', weights_only=False)
m = p.get('meta', {})
assert m.get('model_path') == sys.argv[2]
assert m.get('dataset') == 'tiny-imagenet' and m.get('num_classes') == 200
assert tuple(p['class_feature_mean'].shape) == (200, 512)
PY
    then echo "$output"; return 0; else echo "Existing statistics do not match checkpoint: $output" >&2; return 1; fi
  fi
  python prepare_main_statistics.py \
    --dataset tiny-imagenet \
    --model_path "$checkpoint" \
    --anchor_key lookahead_anchor \
    --data_root "$CLEAN_TINY" \
    --feature_samples 500 \
    --fisher_samples 500 \
    --fisher_max_batches -1 \
    --fisher_mode main \
    --batch_size 64 \
    --num_workers 2 \
    --seed 0 \
    --device cpu \
    --output_path "$output" \
    > "$LOG_ROOT/${name}_statistics.log" 2>&1 || { tail -50 "$LOG_ROOT/${name}_statistics.log" >&2; return 1; }
  python - "$output" "$checkpoint" <<'PY'
import sys, torch
p=torch.load(sys.argv[1],map_location='cpu',weights_only=False)
m=p.get('meta',{})
assert m.get('model_path') == sys.argv[2]
assert m.get('dataset') == 'tiny-imagenet' and m.get('num_classes') == 200
assert tuple(p['class_feature_mean'].shape) == (200,512)
PY
  echo "$output"
}

run_one() {
  local dataset="$1" checkpoint="$2" label="$3" stats="$4" method="$5" data_path="$6" clean_path="$7"
  local output="$RESULT_ROOT/$dataset/$label/$method.json"
  local logfile="$LOG_ROOT/${dataset}_${label}_${method}.log"
  mkdir -p "$(dirname "$output")"
  if [[ -e "$output" ]]; then
    if [[ -s "$output" ]] && validate_result "$output" "$method" "$dataset" "$checkpoint" "$stats"; then
      echo "SKIP_VALID_EXISTING $dataset $label $method"
      record "$dataset" "$label" "$method" "SKIP_VALID_EXISTING" "$output"
      return 0
    fi
    echo "CONFLICT: existing nonmatching result left untouched: $output" | tee -a "$logfile"
    record "$dataset" "$label" "$method" "CONFLICT" "$output"
    return 1
  fi
  echo "START $(date -Is) dataset=$dataset checkpoint=$label method=$method gpu=${CUDA_VISIBLE_DEVICES}"
  if python main_test.py \
      --dataset "$dataset" \
      --federated_method fedacg \
      --model_path "$checkpoint" \
      --anchor_key lookahead_anchor \
      --tta_method "$method" \
      --statistics_path "$stats" \
      --cifar_c_path "$data_path" \
      --clean_data_root "$clean_path" \
      --device cuda \
      --batch_size 64 \
      --num_workers 4 \
      --seed 0 \
      --max_batches -1 \
      --output_path "$output" \
      > "$logfile" 2>&1; then
    if validate_result "$output" "$method" "$dataset" "$checkpoint" "$stats"; then
      echo "PASS $(date -Is) dataset=$dataset checkpoint=$label method=$method output=$output"
      record "$dataset" "$label" "$method" "PASS" "$output"
      return 0
    fi
    echo "INVALID_RESULT $(date -Is) dataset=$dataset checkpoint=$label method=$method; see $logfile"
    record "$dataset" "$label" "$method" "INVALID_RESULT" "$logfile"
    return 1
  else
    local rc=$?
    echo "FAIL $(date -Is) dataset=$dataset checkpoint=$label method=$method exit=$rc; see $logfile"
    tail -30 "$logfile"
    record "$dataset" "$label" "$method" "FAIL_${rc}" "$logfile"
    return 1
  fi
}

preflight() {
  [[ -d "$CIFAR_C" && -f "$CIFAR_C/labels.npy" ]] || { echo "CIFAR-100-C is unavailable: $CIFAR_C" >&2; return 1; }
  [[ -d "$CLEAN_TINY/train" && -d "$TINY_C" ]] || { echo "Tiny-ImageNet data is unavailable" >&2; return 1; }
  python - "$CLEAN_TINY" "$TINY_C" "${C100_CKPTS[@]}" "${TINY_CKPTS[@]}" <<'PY'
from pathlib import Path
import sys, torch
clean, tiny, *checkpoints = sys.argv[1:]
classes={p.name for p in (Path(clean)/'train').iterdir() if p.is_dir()}
assert len(classes)==200, len(classes)
root=Path(tiny)
corruptions=['gaussian_noise','shot_noise','impulse_noise','defocus_blur','glass_blur','motion_blur','zoom_blur','snow','frost','fog','brightness','contrast','elastic_transform','pixelate','jpeg_compression']
for corruption in corruptions:
    severity=root/corruption/'5'
    assert severity.is_dir(), str(severity)
    found={p.name for p in severity.iterdir() if p.is_dir()}
    assert found==classes, (corruption,len(found),len(found ^ classes))
    total=sum(sum(1 for f in c.iterdir() if f.is_file() and f.suffix.lower() in {'.jpeg','.jpg','.png'}) for c in severity.iterdir() if c.is_dir())
    assert total==10000, (corruption,total)
for i, path in enumerate(checkpoints):
    ckpt=torch.load(path,map_location='cpu',weights_only=False)
    state=ckpt['lookahead_anchor']
    num_classes=int(state['fc.weight'].shape[0])
    expected=100 if i<3 else 200
    assert num_classes==expected, (path,num_classes,expected)
    print(f"CHECKPOINT_OK classes={num_classes} path={path}")
print(f"TINY_C_OK corruptions={len(corruptions)} classes=200 images_per_block=10000 severity=5")
PY
  python - "${C100_STATS[@]}" <<'PY'
import sys,torch
for path in sys.argv[1:]:
 p=torch.load(path,map_location='cpu',weights_only=False)
 assert p['class_feature_mean'].shape == (100,512)
 print('STATISTICS_OK',path)
PY
}

printf 'RUN_ROOT=%s\nGPU_VISIBLE=%s\n' "$RUN_ROOT" "$CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv || true
if ! preflight; then
  record '-' '-' '-' 'PREFLIGHT_FAILED' 'see manager output'
  exit 2
fi

for i in 0 1 2; do
  for method in petta cotta; do
    run_one cifar100 "${C100_CKPTS[$i]}" "${C100_NAMES[$i]}" "${C100_STATS[$i]}" "$method" "$CIFAR_C" "$CLEAN_CIFAR" || true
  done
done

for i in 0 1 2; do
  checkpoint="${TINY_CKPTS[$i]}"
  label="${TINY_NAMES[$i]}"
  stats="$STATS_ROOT/${label}_statistics.pt"
  if [[ ! -s "$stats" ]]; then
    if ! stats="$(ensure_tiny_statistics "$checkpoint" "$label")"; then
      record tiny-imagenet "$label" '-' 'STATISTICS_FAILED' 'see statistics log'
      continue
    fi
  fi
  for method in petta cotta; do
    run_one tiny-imagenet "$checkpoint" "$label" "$stats" "$method" "$TINY_C" "$CLEAN_TINY" || true
  done
done

python - "$STATUS_FILE" <<'PY'
import sys
from pathlib import Path
rows=[x.split('\t') for x in Path(sys.argv[1]).read_text().splitlines()[1:] if x.strip()]
print('FINAL_STATUS_COUNTS', {s:sum(r[4]==s for r in rows) for s in sorted({r[4] for r in rows})})
print('STATUS_FILE',sys.argv[1])
PY
