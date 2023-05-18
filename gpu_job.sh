#!/bin/bash
#SBATCH --job-name=TCN_training_02
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=1
#SBATCH --distribution=cyclic:cyclic
#SBATCH --time=1:00:00
#SBATCH --mem-per-cpu=2000
#SBATCH --partition=medium
#SBATCH --gres=gpu:1
#SBATCH -p medium                     # Partition to submit to
#SBATCH -o %x-%j.out # File to which STDOUT will be written
#SBATCH -e %x-%j.err # File to which STDERR will be written
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

module load Anaconda3/2020.02

conda activate jupyter

python train.py