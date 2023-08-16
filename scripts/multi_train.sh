#!/bin/bash

configs=("TCN-BL" "TCN-5k4" "WN-1k5" "WN-150" "LSTM-96" "LSTM-96-2")

# Change the working directory to the parent directory
cd "$(dirname "${BASH_SOURCE[0]}")/.."

n_epochs=(500)
batch_size=(16)

for epoch in "${n_epochs[@]}"; do
    for batch in "${batch_size[@]}"; do
        for config in "${configs[@]}"; do
            python3 train.py --n_epochs "$epoch" --batch_size "$batch" --config "$config"
        done
    done
done