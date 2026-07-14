#!/bin/bash
#SBATCH --gpus-per-node=8
#SBATCH --nodes=1
#SBATCH --partition=gpuA100x8
#SBATCH --job-name=muse_PARS_r_f_both
#SBATCH --time=7:00:00
#SBATCH -e slurm-%j.err
#SBATCH -o slurm-%j.out
#SBATCH --mem=100G
#SBATCH --mail-user=jiajunr2@umn.edu
#SBATCH --mail-type="BEGIN,END"
#SBATCH --account=bhdx-delta-gpu


ALGO="probe_npo_gdr"
CORPUS="news"
FORGET="../data/$CORPUS/raw/forget.txt"
RETAIN="../data/$CORPUS/raw/retain1.txt"
TARGET_DIR="muse-bench/MUSE-News_target"
TOKENIZER_DIR="meta-llama/Llama-2-7b-hf"
MAX_LEN=2048
EPOCHS=10
LR='1e-5'
OUT_BASE="/projects/standard/mhong/shared/hadir/out_dir/$CORPUS/pars"
PER_DEVICE_BATCH_SIZE=2
RETAIN_ALPHA='2.0'
PROBE_LAYERS="24 26 28 30"
PROBE_LR='5e-5'
PROBE_INNER_STEPS=4
PROBE_BETA='0.5'

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

python unlearn.py \
    --algo $ALGO \
    --model_dir $TARGET_DIR \
    --tokenizer_dir $TOKENIZER_DIR \
    --data_file $FORGET \
    --retain_data_file $RETAIN \
    --out_dir "$OUT_BASE" \
    --max_len $MAX_LEN \
    --epochs $EPOCHS \
    --lr $LR \
    --per_device_batch_size $PER_DEVICE_BATCH_SIZE \
    --retain_alpha $RETAIN_ALPHA \
    --probe_layers $PROBE_LAYERS \
    --probe_lr $PROBE_LR \
    --probe_inner_steps $PROBE_INNER_STEPS \
    --probe_beta $PROBE_BETA 
