#!/bin/bash

# list the configurations to be trained:
configs=("tcn-baseline-v28" "wavenet-1k5-v28" "gcn-250-v28")

# Change the working directory to the parent directory
cd "$(dirname "${BASH_SOURCE[0]}")/.."

n_epochs=(100)
batch_size=(16)

for epoch in "${n_epochs[@]}"; do
    for batch in "${batch_size[@]}"; do
        for config in "${configs[@]}"; do
            python3 train.py --n_epochs "$epoch" --batch_size "$batch" --config "$config"
        done
    done
done