#!/bin/bash
# Change the working directory to the parent directory
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# Define the directory with the checkpoints
modelsdir="results/checkpoints/"

n_epochs="2"
batch_size="4"

for checkpoint in "$modelsdir"/*.pt; do
    printf "Resuming training for %s\n" "$checkpoint"
    python train.py --n_epochs "$n_epochs" --batch_size "$batch_size" --checkpoint "$checkpoint"
done