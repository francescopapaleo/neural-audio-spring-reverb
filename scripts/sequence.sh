#!/bin/bash
# Change the working directory to the parent directory
cd "$(dirname "${BASH_SOURCE[0]}")/.."


python3 train.py --n_epochs 5 --batch_size 16 --checkpoint neural-audio-spring-reverb/results/16k/models/tcn-baseline-16.0k.pt  --sample_rate 16000 --bit_rate 16 --dataset springset


python3 train.py --n_epochs 5 --batch_size 16 --checkpoint neural-audio-spring-reverb/results/48k/models/tcn-baseline-v28-48.0k.pt  --sample_rate 48000 --bit_rate 24 --dataset egfxset


python3 train.py --n_epochs 50 --batch_size 8 --conf LSTM-96 --sample_rate 48000 --bit_rate 24 --dataset egfxset
