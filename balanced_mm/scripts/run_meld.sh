#!/usr/bin/env bash
# MELD (text + audio embeddings): uni-modal baselines + none/opm/ogm/both, seeds 0-2.
# Prereq: python scripts/make_meld_csv.py --features_dir <...>   (writes data/meld/*.csv + audio.npz)
# Usage:  bash scripts/run_meld.sh [PAR]        PAR = parallel runs (default 2)
set -e
cd "$(dirname "$0")/.."
if [ ! -f data/meld/train.csv ]; then
  echo "data/meld/train.csv missing - run scripts/make_meld_csv.py first (see its docstring)"
  exit 1
fi

COMMON="--data csv --train_csv data/meld/train.csv --val_csv data/meld/val.csv --img_root data/meld \
  --dataset_name meld --max_len 48 --epochs 12 --batch_size 32 --lr 0.01 \
  --num_workers 0 --log_interval 0"
PAR=${1:-2}
mkdir -p runs

run() { name=$1; shift; echo ">> $name"; OMP_NUM_THREADS=1 python train.py $COMMON "$@" --out_dir "runs/$name" > "runs/$name.out" 2>&1; }

jobs_list=(
  "meld_only_text   --only text  --modulation none --seed 0"
  "meld_only_audio  --only audio --modulation none --seed 0"
)
for s in 0 1 2; do
  for m in none opm ogm both; do
    jobs_list+=("meld_s${s}_${m} --modulation $m --seed $s")
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
(( fail == 0 )) && echo "all meld runs finished OK"
exit $fail
