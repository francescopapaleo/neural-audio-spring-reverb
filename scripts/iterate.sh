#!/bin/bash

# Define the directory with the checkpoints
checkpoints_dir="checkpoints"

# Iterate over all .pt (PyTorch checkpoint) files in the directory
for checkpoint in "$checkpoints_dir"/*.pt; do
    # Call your test script with the checkpoint file as an argument
    python3 test.py --load "$checkpoint" --sub_dir "test01"
    done