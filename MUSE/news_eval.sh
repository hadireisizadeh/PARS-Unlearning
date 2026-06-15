#!/bin/bash
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --partition=gpuA40x4
#SBATCH --job-name=eval
#SBATCH --time=15:00:00
#SBATCH -e slurm-%j.err
#SBATCH -o slurm-%j.out
#SBATCH --mem=40G
#SBATCH --mail-user=jruan@umn.edu
#SBATCH --mail-type="BEGIN,END"
#SBATCH --account=bhdx-delta-gpu

python eval.py \
  --model_dirs "/work/nvme/bhdx/jiajunr2/PARS/MUSE/lr_1e-5_epochs_10_probe_layers_6_20_25_28_beta_0.5_r_f_both/checkpoint-250" "/work/nvme/bhdx/jiajunr2/PARS/MUSE/lr_1e-5_epochs_10_probe_layers_6_20_25_28_beta_0.5_r_f_both" \
  --names "ckeckpoint" "final" \
  --corpus "news" \
  --out_file "./PARS.csv"



