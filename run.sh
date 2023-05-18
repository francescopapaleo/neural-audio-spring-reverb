#!/bin/bash
#SBATCH -J sweep_small
#SBATCH -p medium
#SBATCH -N 1
#SBATCH --gres=gpu:tesla:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16g
#SBATCH --time=2:00:00
#SBATCH -o %N.%J.OUTPUT.out
#SBATCH -e %N.%J.ERROR_LOGS.err

source /etc/profile.d/lmod.sh
source /etc/profile.d/zz_hpcnow-arch.sh

module load Anaconda3/2020.02

source activate jupyter
python train_TCN.py
