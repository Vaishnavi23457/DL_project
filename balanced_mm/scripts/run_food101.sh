#!/usr/bin/env bash
# Food-101 subset (image + prompt-text): uni-modal baselines + none/opm/ogm/both, seeds 0-2.
# Prereq: Food-101 extracted under data/food101/food-101/ (see scripts/make_food101_csv.py).
# Usage:  bash scripts/run_food101.sh [PAR]        PAR = parallel runs (default 2)
set -e
cd "$(dirname "$0")/.."
[ -f data/food101/train.csv ] || python scripts/make_food101_csv.py --root data/food101

COMMON="--data csv --train_csv data/food101/train.csv --val_csv data/food101/val.csv --img_root data/food101 \
  --dataset_name food101 --image_encoder smallcnn --img_size 64 --epochs 12 --batch_size 32 --lr 0.01 \
  --num_workers 0 --log_interval 0"
PAR=${1:-2}
mkdir -p runs

run() { name=$1; shift; echo ">> $name"; OMP_NUM_THREADS=1 python train.py $COMMON "$@" --out_dir "runs/$name" > "runs/$name.out" 2>&1; }

jobs_list=(
  "food_only_image  --only image --modulation none --seed 0"
  "food_only_text   --only text  --modulation none --seed 0"
)
for s in 0 1 2; do
  for m in none opm ogm both; do
    jobs_list+=("food_s${s}_${m} --modulation $m --seed $s")
  done
done

i=0
for j in "${jobs_list[@]}"; do
  set -- $j; name=$1; shift
  run "$name" "$@" &
  i=$((i+1)); if (( i % PAR == 0 )); then wait; fi
done
wait

fail=0
for j in "${jobs_list[@]}"; do
  set -- $j; name=$1
  [[ -f "runs/$name/summary.json" ]] || { echo "FAILED: $name (see runs/$name.out)"; fail=1; }
done
(( fail == 0 )) && echo "all food101 runs finished OK"
exit $fail
