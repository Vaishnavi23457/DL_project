#!/usr/bin/env bash
# 3-seed reproducibility sweep for the fusion comparison.
#   seed 0 -> already stored in runs/syn_{none,opm,ogm,both}
#   seeds 1,2 -> runs/sweep_s{1,2}_{none,opm,ogm,both}   (created by this script)
# Then regenerate results/ (tables + plots) via scripts/make_report.py.
# Usage:  bash scripts/run_seed_sweep.sh [PAR]        PAR = parallel runs (default 2)
set -e
cd "$(dirname "$0")/.."
PAR=${1:-2}
SEEDS=(1 2)
MODS=(none opm ogm both)
COMMON="--data synthetic --image_encoder smallcnn --img_size 64 --epochs 15 --batch_size 32 --syn_train_n 2400 --syn_val_n 600 --lr 0.01 --num_workers 0 --log_interval 0"
mkdir -p runs

run() { name=$1; shift; echo ">> $name"; OMP_NUM_THREADS=1 python train.py $COMMON "$@" --out_dir "runs/$name" > "runs/$name.out" 2>&1; }

jobs_list=()
for s in "${SEEDS[@]}"; do
  for m in "${MODS[@]}"; do
    jobs_list+=("sweep_s${s}_${m} --modulation $m --seed $s")
  done
done

i=0
for j in "${jobs_list[@]}"; do
  set -- $j; name=$1; shift
  run "$name" "$@" &
  i=$((i+1)); if (( i % PAR == 0 )); then wait; fi
done
wait

# sanity: every run must have produced a summary
fail=0
for j in "${jobs_list[@]}"; do
  set -- $j; name=$1
  if [[ ! -f "runs/$name/summary.json" ]]; then echo "FAILED: $name (see runs/$name.out)"; fail=1; fi
done
(( fail == 0 )) && echo "all sweep runs finished OK"
exit $fail
