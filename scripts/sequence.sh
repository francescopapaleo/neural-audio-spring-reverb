#!/bin/bash
# Change the working directory to the parent directory
cd "$(dirname "${BASH_SOURCE[0]}")/.."

python train.py --n_epochs 1 --batch_size 4 --checkpoint results/checkpoints/gcn-2500.pt

python train.py --n_epochs 10 --batch_size 4 --checkpoint results/checkpoints/wavenet-10.pt

python train.py --n_epochs 20 --batch_size 4 --checkpoint results/checkpoints/lstm-cs-96.pt
