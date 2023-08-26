#!/bin/bash

# Change the working directory to the parent directory
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# Define the directory with the checkpoints
IRs_dir="results/measured_IR"

# Iterate over all .pt (PyTorch checkpoint) files in the directory
for ir_audio in "$IRs_dir"/*.wav; do
    # Call your test script with the checkpoint file as an argument
    printf "Measuring RT60: %s\n" "$ir_audio"
    
    # for impulse response measurement uncomment the line below
    python -m src.rt60 --device cpu --sample_rate 48000 --input "$ir_audio"
    
    # for testing on egfxset uncomment the line below
    # python test.py --checkpoint "$checkpoint" --logdir "results/48k" --device cpu --dataset egfxset --sample_rate 48000
    done