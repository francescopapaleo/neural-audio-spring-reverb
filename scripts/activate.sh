#!/bin/bash

srun --nodes=1 --partition=high --gres=gpu:1 --cpus-per-task=4 --mem=16g --pty bash -i
