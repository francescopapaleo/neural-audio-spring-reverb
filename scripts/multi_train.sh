#!/bin/bash

# configs=("TCN-1" "TCN-2" "TCN-3" TCN-4 "WN-1" "WN-2" "WN-3" "WN-4")

configs=("TCN-5" "WN-5")

# Change the working directory to the parent directory
cd "$(dirname "${BASH_SOURCE[0]}")/.."

n_epochs=(25)
batch_size=(32)

for epoch in "${n_epochs[@]}"; do
    for batch in "${batch_size[@]}"; do
        for config in "${configs[@]}"; do
            python3 train.py --n_epochs "$epoch" --batch_size "$batch" --config "$config"
        done
    done
done
