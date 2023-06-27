#!/bin/bash

# Change the working directory to the parent directory
cd "$(dirname "${BASH_SOURCE[0]}")/.."

n_epochs=(25 50)
batch_size=(8 32)

for epoch in "${n_epochs[@]}"; do
    for batch in "${batch_size[@]}"; do
        python3 train.py --n_epochs "$epoch" --batch_size "$batch" --config TCN_1
    done
done
