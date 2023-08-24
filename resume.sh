#!/bin/bash

modelsdir="results/checkpoints/tmp/"
n_epochs="1"
batch_size="4"

for checkpoint in "$modelsdir"/*.pt; do
    python train.py --n_epochs "$n_epochs" --batch_size "$batch_size" --checkpoint "$checkpoint"
done


python train.py --n_epochs 100 --batch_size "$batch_size" --config tcn-baseline

python train.py --n_epochs 100 --batch_size "$batch_size" --config wavenet-18