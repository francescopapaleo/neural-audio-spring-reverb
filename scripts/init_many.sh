#!/bin/bash

# Change the working directory to the parent directory
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# Define the directory with the checkpoints
configs_dir="configs/48k-egfxset"

# Iterate over all .pt (PyTorch checkpoint) files in the directory
for config in "$configs_dir"/*.yaml; do
    # Call your test script with the checkpoint file as an argument
    
    python main.py train --init "$config"
    printf "Done.\n\n"

done