#!/bin/bash

# Change the working directory to the parent directory
cd "$(dirname "${BASH_SOURCE[0]}")/.."

n_epochs=(100 200)
batch_size=(16 32)

for epoch in "${n_epochs[@]}"; do
    for batch in "${batch_size[@]}"; do
        python3 train.py --n_epochs "$epoch" --batch_size "$batch"
    done
done
