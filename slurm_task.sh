#!/bin/bash

#SBATCH --job-name PE_all_weights  # Nombre del proceso

#SBATCH --partition dgx            # Cola para ejecutar

#SBATCH --gres=gpu:1               # Numero de gpus a usar

export PATH="/opt/anaconda/anaconda3/bin:$PATH"

export PATH="/opt/anaconda/bin:$PATH"

eval "$(conda shell.bash hook)"

conda activate /mnt/homeGPU/hexecode/pt23_env

python -u experimentacion.py --model informer --data HPCm --ex_name all_pe_weighted --data_path household_power_consumption.txt --freq t --folder InformerVanilla --root_path ./Datasets/HPC --batch_size 32 --dropout 0.3 --itr 3 --attn full --window 60 --seq_len 180 --label_len 60 --pred_len 60 --time_encoding all_pe_weighted
