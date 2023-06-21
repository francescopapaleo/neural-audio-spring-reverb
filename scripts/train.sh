#!/bin/bash
#SBATCH -J TCN
#SBATCH -p medium
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16g
#SBATCH --time=8:00:00
#SBATCH -o ./runs/sbatch/%N.%J.OUT.out
#SBATCH -e ./runs/sbatch/%N.%J.ERR.err

source /etc/profile.d/lmod.sh

source /etc/profile.d/zz_hpcnow-arch.sh

module load Anaconda3/2020.02

eval "$(conda shell.bash hook)"

source activate envtorch

python train.py  --device cuda:0 --n_epochs 100 --batch_size 4 --sub_dir train01
