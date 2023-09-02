#!/bin/bash

# Change the working directory to the parent directory
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# Define the directory with the checkpoints
checkpoints_dir="models"

# Iterate over all .pt (PyTorch checkpoint) files in the directory
for checkpoint in "$checkpoints_dir"/*.pt; do
    # Call your test script with the checkpoint file as an argument
    printf "Testing checkpoint: %s\n" "$checkpoint"
    
    # for testing on egfxset uncomment the line below
    python test.py --checkpoint "$checkpoint" --dataset egfxset --logdir "logs" --sample_rate 48000 --bit_rate 24
    
    # for testing on springset uncomment the line below
    # python test.py --checkpoint "$checkpoint" --dataset springset --logdir "logs" --sample_rate 16000 --bit_rate 16
    printf "Done.\n\n"

done
