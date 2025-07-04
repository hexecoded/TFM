#!/bin/bash

#SBATCH --job-name PE_Stats_Lags_FixedPE                # Nombre del proceso

#SBATCH --partition dgx2   # Cola para ejecutar

#SBATCH --gres=gpu:1                           # Numero de gpus a usar

export PATH="/opt/anaconda/anaconda3/bin:$PATH"

export PATH="/opt/anaconda/bin:$PATH"

eval "$(conda shell.bash hook)"

conda activate /mnt/homeGPU/hexecode/pt23_env

export TFHUB_CACHE_DIR=.

python -u experimentacion.py \ 
    --model informer \
    --data ETTh1 \
    --ex_name Stats_Lags_FixedPE \
    --data_path household_power_consumption.txt \
    --freq t \
    --folder InformerVanilla \
    --root_path ./Datasets/HPC \
    --batch_size 32 \
    --dropout 0.3 \
    --itr 3 \
    --attn full \
    --window 60 \
    --seq_len 180 \
    --label_len 60 \
    --pred_len 60 \
    --embed stats_lags
