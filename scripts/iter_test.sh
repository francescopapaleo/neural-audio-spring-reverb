#!/bin/bash

# Change the working directory to the parent directory
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# Define the directory with the checkpoints
checkpoints_dir="results/16k/models/"

# Iterate over all .pt (PyTorch checkpoint) files in the directory
for checkpoint in "$checkpoints_dir"/*.pt; do
    # Call your test script with the checkpoint file as an argument
    printf "Testing checkpoint: %s\n" "$checkpoint"
    
    # for impulse response measurement uncomment the line below
    # python -m src.impulse_response --checkpoint "$checkpoint"  --device cpu --sample_rate 48000 
    
    # for testing on egfxset uncomment the line below
    # python test.py --checkpoint "$checkpoint" --logdir "results/48k" --device cpu --dataset egfxset --sample_rate 48000
    python test.py --checkpoint "$checkpoint" --logdir "results/16k" --device cpu --dataset springset --sample_rate 16000 
    done