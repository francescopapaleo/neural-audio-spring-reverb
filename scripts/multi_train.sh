#!/bin/bash

configs=("TCN-BL" "TCN-4k" "TCN-44k" "TCN-10" "WN-150" "WN-1k" "WN-1500")

# Change the working directory to the parent directory
cd "$(dirname "${BASH_SOURCE[0]}")/.."

n_epochs=(50)
batch_size=(8)

for epoch in "${n_epochs[@]}"; do
    for batch in "${batch_size[@]}"; do
        for config in "${configs[@]}"; do
            python3 train.py --n_epochs "$epoch" --batch_size "$batch" --config "$config" --logdir "results/runs"
        done
    done
done
