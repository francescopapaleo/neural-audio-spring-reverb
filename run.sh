#!/bin/bash
#SBATCH -J TCN
#SBATCH -p short
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --mem=16g
#SBATCH --time=8:00
#SBATCH -o %N.%J.OUTPUT.out
#SBATCH -e %N.%J.ERROR.err

module load CUDA/11.4.3

source /etc/profile.d/lmod.sh
source /etc/profile.d/zz_hpcnow-arch.sh

# ACTIVATE ANACONDA
eval "$(conda shell.bash hook)"
conda activate jupyter


python training.py
