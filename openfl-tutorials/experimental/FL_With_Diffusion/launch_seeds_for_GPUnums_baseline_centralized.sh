#!/bin/bash

SEEDS=("NA" "10" "11" "12" "13" "14")

args="$@"

for i in {1..5}; do
  GPU=${!i}
  SEED=${SEEDS[$i]}
  echo "Launching on GPU $GPU with seed $SEED"

  CUDA_VISIBLE_DEVICES=$GPU  python cifar10_with_diffusion_v2.py --model_seed $SEED --comm_round 50 --num_cols 1 --learning_rate 0.0003 > stdout_textfiles/baseline_centralized_seed_${SEED}_lr_0.0003_stdout.txt &
done

wait

echo "All tasks are done."
