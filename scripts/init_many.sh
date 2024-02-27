#!/bin/bash

# Change the working directory to the parent directory
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# Define the directory with the checkpoints
configs_dir="configs/"

# Iterate over all .pt (PyTorch checkpoint) files in the directory
for config in "$configs_dir"/*.yaml; do
    # Call your test script with the checkpoint file as an argument
    
    nafxsr train --init "$config" --dataset springset --sample_rate 16000 --bit_depth 16 --batch_size 96 --num_workers 8
    nafxsr train --init "$config" --dataset egfxset --sample_rate 48000 --bit_depth 24 --batch_size 32 --num_workers 8
    printf "Done.\n\n"

done