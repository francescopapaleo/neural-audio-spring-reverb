#!/bin/bash
#SBATCH -J TCN
#SBATCH -p short
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16g
#SBATCH --time=2:00:00
#SBATCH -o /homedtic/fpapaleo/smc-spring-reverb/logs/%N.%J.OUT.out
#SBATCH -e /homedtic/fpapaleo/smc-spring-rever/logs/%N.%J.ERR.err

source /etc/profile.d/lmod.sh

source /etc/profile.d/zz_hpcnow-arch.sh

module load Anaconda3/2020.02

conda activate envtorch

python train.py  --batch_size 64 --epochs 50 --device cuda:0 --crop 3200
