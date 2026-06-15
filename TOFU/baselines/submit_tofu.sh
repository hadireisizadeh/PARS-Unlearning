#!/bin/bash
#SBATCH --gpus-per-node=3
#SBATCH --nodes=1
#SBATCH --partition=gpuA40x4
#SBATCH --job-name=tofu_minmax
#SBATCH --time=15:00:00
#SBATCH -e log/slurm-%j.err
#SBATCH -o log/slurm-%j.out
#SBATCH --mem=60G
#SBATCH --mail-user=jruan@umn.edu
#SBATCH --mail-type="BEGIN,END"
#SBATCH --account=bhdx-delta-gpu

module load cuda/12.8

python unlearn.py