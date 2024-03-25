#!/bin/bash

# Change the working directory to the parent directory
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# Define the directory with the checkpoints
IRs_dir="audio/IR_models"

# Iterate over all .pt (PyTorch checkpoint) files in the directory
for ir_audio in "$IRs_dir"/*.wav; do
    # Call your test script with the checkpoint file as an argument
    printf "Measuring RT60: %s\n" "$ir_audio"
    
    # for impulse response measurement uncomment the line below
    nafx-springrev rt60 --input "$ir_audio"
    
    done

printf "Done!\n"