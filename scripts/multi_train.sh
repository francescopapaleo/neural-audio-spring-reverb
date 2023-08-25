#!/bin/bash

# configs=("tcn-baseline-mse" "wavenet-ff-mse" "lstm-mse" "lstm-conv-skip-mse" "gcn-mse")

# Change the working directory to the parent directory
cd "$(dirname "${BASH_SOURCE[0]}")/.."

n_epochs=(50)
batch_size=(4)

for epoch in "${n_epochs[@]}"; do
    for batch in "${batch_size[@]}"; do
        for config in "${configs[@]}"; do
            python3 train.py --n_epochs "$epoch" --batch_size "$batch" --config "$config"
        done
    done
done