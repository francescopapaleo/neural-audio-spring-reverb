#!/bin/bash

configs=("LSTM-BL" "LSTM-32-4" "BiLSTM" "BiLSTM-8" "BiLSTM-96" "LSTM-96" "LSTM-96-gated")

# Change the working directory to the parent directory
cd "$(dirname "${BASH_SOURCE[0]}")/.."

n_epochs=(100)
batch_size=(8)

for epoch in "${n_epochs[@]}"; do
    for batch in "${batch_size[@]}"; do
        for config in "${configs[@]}"; do
            python3 train.py --n_epochs "$epoch" --batch_size "$batch" --config "$config"
        done
    done
done
