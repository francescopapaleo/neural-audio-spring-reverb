#!/bin/bash

# list the configurations to be trained:
# configs=("tcn-baseline" "wavenet-10" "lstm-cs-96" "gcn-250")

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