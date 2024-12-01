#!/bin/bash

#SBATCH -p gpu 
#SBATCH -c 10
#SBATCH --gres=gpu:1
#SBATCH -C gmem80
#SBATCH --gres-flags=enforce-binding

echo CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
module load anaconda3 
source activate gdino 
python train.py



