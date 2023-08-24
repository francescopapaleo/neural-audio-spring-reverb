#!/bin/bash

modelsdir="results/checkpoints/"
n_epochs="2"
batch_size="4"

for checkpoint in "$modelsdir"/*.pt; do
    python train.py --n_epochs "$n_epochs" --batch_size "$batch_size" --checkpoint "$checkpoint"
done
