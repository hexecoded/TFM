#!/bin/bash

#SBATCH --job-name=ETTm2_WinSize               # Nombre del proceso
#SBATCH --partition=dgx2                        # Cola para ejecutar
#SBATCH --gres=gpu:1                            # Número de GPUs a usar
#SBATCH --ntasks=1                              # Número de tareas (procesos)
#SBATCH --cpus-per-task=32                      # Número de CPUs por tarea

export PATH="/opt/anaconda/anaconda3/bin:$PATH"
export PATH="/opt/anaconda/bin:$PATH"
export TFHUB_CACHE_DIR="/mnt/homeGPU/hexecode/cache"

eval "$(conda shell.bash hook)"
conda activate /mnt/homeGPU/hexecode/pt23_env

# Parámetros configurables
start=3
end=96
dataset="ETTm2"
data_path="${dataset}.csv"
root_path="./Datasets/ETT-small"

# Bucle de barrido de tamaños de ventana
for window in $(seq $start $end); do
    echo ">>> Ejecutando con ventana: $window (all_pe_weighted)"
    python -u run_exp.py \
    --model informer \
    --data $dataset \
    --ex_name ${dataset}_all_pe_weighted_win$window \
    --data_path $data_path \
    --freq t \
    --folder InformerVanilla \
    --root_path $root_path \
    --batch_size 32 \
    --dropout 0.2 \
    --itr 10 \
    --attn full \
    --window $window \
    --seq_len 96 \
    --label_len 48 \
    --pred_len 24 \
    --time_encoding all_pe_weighted
    
    echo ">>> Ejecutando con ventana: $window (tpe)"
    python -u run_exp.py \
    --model informer \
    --data $dataset \
    --ex_name ${dataset}_tpe_win$window \
    --data_path $data_path \
    --freq t \
    --folder InformerVanilla \
    --root_path $root_path \
    --batch_size 32 \
    --dropout 0.2 \
    --itr 10 \
    --attn full \
    --window $window \
    --seq_len 96 \
    --label_len 48 \
    --pred_len 24 \
    --time_encoding tpe
done
