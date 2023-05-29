#!/bin/bash
#SBATCH -J TCN
#SBATCH -p short
#SBATCH -N 1
#SBATCH --gres=gpu:tesla:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16g
#SBATCH --time=2:00
#SBATCH -o %N.%J.OUT.out
#SBATCH -e %N.%J.ERR.err

source /etc/profile.d/lmod.sh
source /etc/profile.d/zz_hpcnow-arch.sh

# ACTIVATE ANACONDA
eval "$(conda shell.bash hook)"
conda activate envtorch

python train.py  --batch_size 16 --epochs 50 --device cuda:0 --crop 3200
