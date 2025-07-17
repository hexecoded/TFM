#!/bin/bash

#SBATCH --job-name PE_TINA_Informer                # Nombre del proceso

#SBATCH --partition dgx2   # Cola para ejecutar

#SBATCH --gres=gpu:1                           # Numero de gpus a usar

#SBATCH --mem=48G

export PATH="/opt/anaconda/anaconda3/bin:$PATH"

export PATH="/opt/anaconda/bin:$PATH"

eval "$(conda shell.bash hook)"

conda activate /mnt/homeGPU/hexecode/pt23_env

export TFHUB_CACHE_DIR=.

python -u experimentacion.py --model informer --data TINA --ex_name tina_informer_opt --data_path tina_30s.csv --folder InformerVanilla --root_path ./Datasets/tina --batch_size 32 --dropout 0.2 --itr 3 --attn full --window 60 --seq_len 288 --label_len 96 --pred_len 48 --time_encoding informer --freq 30s --enc_in 103 --dec_in 103 --c_out 103 --features M