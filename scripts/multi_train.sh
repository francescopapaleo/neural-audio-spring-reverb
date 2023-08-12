#!/bin/bash

configs=("TCN-BL" "TCN-64-10-5-7" "WN-16-10-3" "LSTM-32-4")

# Change the working directory to the parent directory
cd "$(dirname "${BASH_SOURCE[0]}")/.."

n_epochs=(1000)
batch_size=(8)

for epoch in "${n_epochs[@]}"; do
    for batch in "${batch_size[@]}"; do
        for config in "${configs[@]}"; do
            python3 train.py --n_epochs "$epoch" --batch_size "$batch" --config "$config"
        done
    done
done
