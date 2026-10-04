#!/usr/bin/env bash
# Runs the full synthetic comparison: uni-modal baselines + none / opm / ogm / both, then prints a table.
# Usage:  bash scripts/run_synthetic_compare.sh [EPOCHS] [TRAIN_N] [IMAGE_ENCODER] [PARALLEL]
#   e.g.  bash scripts/run_synthetic_compare.sh 20 3000 smallcnn 2       (CPU, ~15-20 min)
#         bash scripts/run_synthetic_compare.sh 30 6000 resnet18 1       (GPU)
set -e
cd "$(dirname "$0")/.."
EPOCHS=${1:-20}; N=${2:-3000}; ENC=${3:-smallcnn}; PAR=${4:-2}
COMMON="--data synthetic --image_encoder $ENC --img_size 64 --epochs $EPOCHS --batch_size 32 --syn_train_n $N --syn_val_n 600 --lr 0.01 --num_workers 0 --log_interval 0"
mkdir -p runs
run() { name=$1; shift; echo ">> $name"; OMP_NUM_THREADS=1 python train.py $COMMON "$@" --out_dir runs/$name > runs/$name.out 2>&1; }

jobs_list=(
  "syn_only_image --only image --modulation none"
  "syn_only_text  --only text  --modulation none"
  "syn_none --modulation none"
  "syn_opm  --modulation opm"
  "syn_ogm  --modulation ogm"
  "syn_both --modulation both"
)
i=0
for j in "${jobs_list[@]}"; do
  set -- $j; name=$1; shift
  run $name "$@" &
  i=$((i+1)); if (( i % PAR == 0 )); then wait; fi
done
wait
echo; python compare.py runs/syn_only_image runs/syn_only_text runs/syn_none runs/syn_opm runs/syn_ogm runs/syn_both
echo; echo "(average of last 3 epochs)"; python compare.py runs/syn_none runs/syn_opm runs/syn_ogm runs/syn_both --last 3
