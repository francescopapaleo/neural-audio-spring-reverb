#!/bin/bash

configs=("LSTM-96-smax" "WN-1500" "TCN-4k")

# Change the working directory to the parent directory
cd "$(dirname "${BASH_SOURCE[0]}")/.."

n_epochs=(150)
batch_size=(8)

for epoch in "${n_epochs[@]}"; do
    for batch in "${batch_size[@]}"; do
        for config in "${configs[@]}"; do
            python3 train.py --n_epochs "$epoch" --batch_size "$batch" --config "$config"
        done
    done
done
