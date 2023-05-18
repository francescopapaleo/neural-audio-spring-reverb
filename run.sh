#!/bin/bash
#SBATCH -J TrainTCN
#SBATCH -p high
#SBATCH -N 1
#SBATCH -n 2
#SBATCH --chdir=/homedtic/fpapaleo/smc-spring-reverb/logs
#SBATCH --cpus-per-task=4
#SBATCH --mem=16g
#SBATCH --time=1:00:00
#SBATCH -o %N.%J.OUTPUT.out
#SBATCH -e %N.%J.ERROR_LOGS.err
hostname
date
 
module load CUDA/11.0.2
cudaMemTest=/soft/slurm_templates/bin/cuda_memtest-1.2.3/cuda_memtest
cudaDevs=$(echo $CUDA_VISIBLE_DEVICES | sed -e 's/,/ /g')
echo CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES

for cudaDev in $cudaDevs
do
  echo cudaDev = $cudaDev
  $cudaMemTest --num_passes 1 --device $cudaDev > gpuMemTest.out.$cudaDev 2>&1 &
done
wait
 
date

source /etc/profile.d/lmod.sh
source /etc/profile.d/zz_hpcnow-arch.sh

module load Anaconda3/2020.02

conda activate jupyter

python train.py
