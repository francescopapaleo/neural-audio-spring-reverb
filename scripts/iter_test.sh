#!/bin/bash
# Change the working directory to the parent directory
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# Define the directory with the checkpoints
checkpoints_dir="results/checkpoints/28"
log_dir="results/runs/28_test"

# Iterate over all .pt (PyTorch checkpoint) files in the directory
for checkpoint in "$checkpoints_dir"/*.pt; do
    # Call your test script with the checkpoint file as an argument
    python3 test.py --checkpoint_path "$checkpoint" --logdir "$log_dir"
    done