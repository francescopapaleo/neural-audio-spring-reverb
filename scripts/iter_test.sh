#!/bin/bash

# Change the working directory to the parent directory
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# Define the directory with the checkpoints
checkpoints_dir="results/48k/models"

# Iterate over all .pt (PyTorch checkpoint) files in the directory
for checkpoint in "$checkpoints_dir"/*.pt; do
    # Call your test script with the checkpoint file as an argument
    printf "Testing checkpoint: %s\n" "$checkpoint"
    
    # for impulse response measurement uncomment the line below
    # python -m src.ir_model --checkpoint "$checkpoint"  --device cpu --sample_rate 48000 --bit_rate 24
    
    # for testing on egfxset uncomment the line below
    python3 test.py --checkpoint "$checkpoint" --logdir "results/48k" --device cuda:0 --dataset egfxset  --sample_rate 48000 --bit_rate 24 --batch_size 8
    
    # python test.py --checkpoint "$checkpoint" --logdir "results/16k" --device cuda:0 --dataset springset --sample_rate 16000 --bit_rate 16
done