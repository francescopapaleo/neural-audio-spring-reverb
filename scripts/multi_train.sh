#!/bin/bash

configs=("WN-8-4-5" "LSTM-64")

# Change the working directory to the parent directory
cd "$(dirname "${BASH_SOURCE[0]}")/.."

n_epochs=(250)
batch_size=(16)

for epoch in "${n_epochs[@]}"; do
    for batch in "${batch_size[@]}"; do
        for config in "${configs[@]}"; do
            python3 train.py --n_epochs "$epoch" --batch_size "$batch" --config "$config"
        done
    done
done
