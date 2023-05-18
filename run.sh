#!/bin/bash
#SBATCH -J TrainTCN1805
#SBATCH -p high
#SBATCH -N 1
#SBATCH -n 2
#SBATCH --chdir=/homedtic/fpapaleo/smc-spring-reverb
#SBATCH --cpus-per-task=4
#SBATCH --mem=16g
#SBATCH --time=:00:00
#SBATCH -o %N.%J.OUTPUT.out
#SBATCH -e %N.%J.ERROR_LOGS.err

source /etc/profile.d/lmod.sh
source /etc/profile.d/zz_hpcnow-arch.sh

source conda activate jupyter

python train.py
