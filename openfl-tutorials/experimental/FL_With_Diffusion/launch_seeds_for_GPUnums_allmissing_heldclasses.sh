#!/bin/bash

SEEDS=("NA" "10" "11" "12" "13" "14")

args="$@"

for i in {1..5}; do
  GPU=${!i}
  SEED=${SEEDS[$i]}
  echo "Launching on GPU $GPU with seed $SEED"

  CUDA_VISIBLE_DEVICES=$GPU  python cifar10_with_diffusion_v2.py --model_seed $SEED --learning_rate 0.0003 --comm_round 50 --held_classes 1,6 --hold_from_cols 0,1,2,3,4 > stdout_textfiles/allmissing_1_6_seed_${SEED}_lr_0.0003_stdout.txt &
done

wait

echo "All tasks are done."
