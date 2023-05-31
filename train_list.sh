#!/bin/bash
#SBATCH -J TCN
#SBATCH -p medium
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16g
#SBATCH --time=7:00:00
#SBATCH -o %N.%J.OUT.out
#SBATCH -e %N.%J.ERR.err

source /etc/profile.d/lmod.sh

source /etc/profile.d/zz_hpcnow-arch.sh

module load Anaconda3/2020.02

eval "$(conda shell.bash hook)"

source activate envtorch

python train_list.py  --device cuda:0
