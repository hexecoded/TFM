"""
------------------------------------------------------------------------------
Script de Experimentación para comprobación de modelos con diferente
Positional Encoding

Autor: Cristhian Moya Mota
------------------------------------------------------------------------------

Este script permite realizar un experimento usando un modelo que siga la
estructura de clases de Informer. Permite configurar los parámetros de entrada
y la ubicación de los datos, así como sus propiedades.
"""

import sys
import argparse
import torch
import numpy as np
import csv
import time
import os
import shutil
import glob

METRIC_LABS = ["MAE", "MSE", "RMSE", "MAPE", "MSPE"]
# Establecemos el nombre del modelo, ie, su nombre en carpeta


# Inserción de parámetros. Se usa el formato argparse para obtener realimentación en caso de requerir ayuda para conocer
# las opciones disponibles
parser = argparse.ArgumentParser(
    description='[Experimentación Informer] Long Sequences Forecasting')

parser.add_argument('--model', type=str, required=True, default='informer',
                    help='model of experiment, options: [informer, informerstack, informerlight(TBD)]')
parser.add_argument('--ex_name', type=str, default="Informer_Original",
                    required=True, help='Experiment name')

parser.add_argument('--folder', type=str, required=True,
                    default='InformerVanilla', help='model folder for experiment')
parser.add_argument('--data', type=str, required=True,
                    default='ETTh1', help='data')
parser.add_argument('--root_path', type=str,
                    default='./Datasets/ETT-small', help='root path of the data file')
parser.add_argument('--data_path', type=str,
                    default='ETTh1.csv', help='data file')
parser.add_argument('--features', type=str, default='M',
                    help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default='OT',
                    help='target feature in S or MS task')
parser.add_argument('--freq', type=str, default='h',
                    help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
parser.add_argument('--checkpoints', type=str,
                    default='./checkpoints/', help='location of model checkpoints')

parser.add_argument('--seq_len', type=int, default=96,
                    help='input sequence length of Informer encoder')
parser.add_argument('--label_len', type=int, default=48,
                    help='start token length of Informer decoder')
parser.add_argument('--pred_len', type=int, default=24,
                    help='prediction sequence length')

parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
parser.add_argument('--c_out', type=int, default=7, help='output size')
parser.add_argument('--d_model', type=int, default=512,
                    help='dimension of model')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--e_layers', type=int, default=2,
                    help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1,
                    help='num of decoder layers')
parser.add_argument('--s_layers', type=str, default='3,2,1',
                    help='num of stack encoder layers')
parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
parser.add_argument('--factor', type=int, default=5,
                    help='probsparse attn factor')
parser.add_argument('--padding', type=int, default=0, help='padding type')
parser.add_argument('--distil', action='store_false',
                    help='whether to use distilling in encoder, using this argument means not using distilling',
                    default=True)
parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
parser.add_argument('--attn', type=str, default='prob',
                    help='attention used in encoder, options:[prob, full]')

parser.add_argument('--time_encoding', type=str, default='',
                    help='time features encoding, options:[no_pe, informer, stats, stats_lags, all_pe_weighted, tpe]')

parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, fixed, learned]')


parser.add_argument('--activation', type=str,
                    default='gelu', help='activation')
parser.add_argument('--window', type=int, default=24,
                    help='window size for stats computing')
parser.add_argument('--output_attention', action='store_true',
                    help='whether to output attention in encoder')
parser.add_argument('--do_predict', action='store_true',
                    help='whether to predict unseen future data')
parser.add_argument('--mix', action='store_false',
                    help='use mix attention in generative decoder', default=True)
parser.add_argument('--cols', type=str, nargs='+',
                    help='certain cols from the data files as the input features')
parser.add_argument('--num_workers', type=int, default=0,
                    help='data loader num workers')
parser.add_argument('--itr', type=int, default=1, help='experiments times')
parser.add_argument('--train_epochs', type=int,
                    default=100, help='train epochs')
parser.add_argument('--batch_size', type=int, default=32,
                    help='batch size of train input data')
parser.add_argument('--patience', type=int, default=3,
                    help='early stopping patience')
parser.add_argument('--learning_rate', type=float,
                    default=0.0005, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='test', help='exp description')
parser.add_argument('--loss', type=str, default='mse', help='loss function')
parser.add_argument('--lradj', type=str, default='type1',
                    help='adjust learning rate')
parser.add_argument('--use_amp', action='store_true',
                    help='use automatic mixed precision training', default=False)
parser.add_argument('--inverse', action='store_true',
                    help='inverse output data', default=False)

parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true',
                    help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0,1,2,3',
                    help='device ids of multile gpus')
parser.add_argument('--shuffle_decoder_input', action='store_true',
                    help='Shuffle decoder input during test')

args = parser.parse_args()

# Cargamos el modelo seleccionado
sys.path += [args.folder]
print(sys.path)


# Comprobación de uso de CUDA para el cómputo
args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
print(f"Usando CUDA: {torch.cuda.is_available()}")
print(args)

if args.use_gpu and args.use_multi_gpu:
    args.devices = args.devices.replace(' ', '')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]

# Datasets Disponibles en el paper de Informer
data_parser = {
    'HPC': {'data': 'household_power_consumption.txt', 'T': 'Global_active_power', 'M': [7, 7, 7], 'S': [1, 1, 1],
            'MS': [7, 7, 1]},
    'Taxi': {'data': 'yellow_tripdata_clean_grouped.parquet', 'T': 'total_amount', 'M': [6, 6, 6], 'S': [1, 1, 1],
            'MS': [6, 6, 1]},
    'ETTh1': {'data': 'ETTh1.csv', 'T': 'OT', 'M': [7, 7, 7], 'S': [1, 1, 1], 'MS': [7, 7, 1]},
    'ETTh2': {'data': 'ETTh2.csv', 'T': 'OT', 'M': [7, 7, 7], 'S': [1, 1, 1], 'MS': [7, 7, 1]},
    'ETTm1': {'data': 'ETTm1.csv', 'T': 'OT', 'M': [7, 7, 7], 'S': [1, 1, 1], 'MS': [7, 7, 1]},
    'ETTm2': {'data': 'ETTm2.csv', 'T': 'OT', 'M': [7, 7, 7], 'S': [1, 1, 1], 'MS': [7, 7, 1]}
}

# Insertamos parámetros al modelo
if args.data in data_parser.keys():
    data_info = data_parser[args.data]
    args.data_path = data_info['data']
    args.target = data_info['T']
    args.enc_in, args.dec_in, args.c_out = data_info[args.features]

# Iniciamos el experimento
from exp.exp_informer import Exp_Informer
Exp = Exp_Informer

# === Inicialización de estructuras ===
metrics = np.zeros(len(METRIC_LABS))
all_metrics = []  
train_times = []
test_times = []

# Preparación del directorio de resultados

results_folder = "results"
setting = '{}_{}_ft{}_sl{}_ll{}_pl{}_win{}_dm{}_nh{}_el{}_dl{}_df{}_at{}_fc{}_eb{}_dt{}_mx{}_{}_{}_{}'.format(
    args.model, args.data, args.features, args.seq_len, args.label_len, args.pred_len, args.window,
    args.d_model, args.n_heads, args.e_layers, args.d_layers, args.d_ff, args.attn, args.factor,
    args.embed, args.distil, args.mix, args.des, args.ex_name, 0
)

for file_path in glob.glob(os.path.join(results_folder, f"{setting[:-2]}*")):
    if os.path.isfile(file_path):
        os.remove(file_path)
    elif os.path.isdir(file_path):
        shutil.rmtree(file_path)

# === Ejecuciones ===
for ii in range(args.itr):
    print("-------------------------------------- Ejecución {} --------------------------------------".format(ii))

    iter_setting = setting[:-2] + str(ii)
    exp = Exp(args)

    print("Entrenamiento")
    start_train = time.time()
    exp.train(iter_setting)
    end_train = time.time()
    train_duration = end_train - start_train
    train_times.append(train_duration)

    print("Evaluación")
    start_test = time.time()
    exp.test(iter_setting)
    end_test = time.time()
    test_duration = end_test - start_test
    test_times.append(test_duration)

    print(f"Tiempo de entrenamiento: {train_duration:.2f}s")
    print(f"Tiempo de test: {test_duration:.2f}s")

    # Cargar métricas de test
    metric_path = f"./results/{iter_setting}/metrics.npy"
    if os.path.exists(metric_path):
        run_metrics = np.load(metric_path)
        if run_metrics.shape[0] == len(METRIC_LABS):
            metrics += run_metrics
            all_metrics.append(run_metrics)
        else:
            print("Error: métrica inesperada.")
    else:
        print(f"No se encontró el archivo: {metric_path}")

    torch.cuda.empty_cache()

# === Resultados Finales ===
mean_metrics = metrics / args.itr
std_metrics = np.std(np.array(all_metrics), axis=0)
mean_train_time = np.mean(train_times)
std_train_time = np.std(train_times)
mean_test_time = np.mean(test_times)
std_test_time = np.std(test_times)

metric_dict = {}

# Agregar métricas con media y std
for i, label in enumerate(METRIC_LABS):
    metric_dict[f"{label}_mean"] = mean_metrics[i]
    metric_dict[f"{label}_std"] = std_metrics[i]

# Agregar tiempos con media y std
metric_dict["TrainTime_mean(s)"] = mean_train_time
metric_dict["TrainTime_std(s)"] = std_train_time
metric_dict["TestTime_mean(s)"] = mean_test_time
metric_dict["TestTime_std(s)"] = std_test_time

# Imprimir resultados

print(f"\n\n{'Metric':<25}{'Mean':>12}{'Std':>12}")
print("-" * 50)
for label in METRIC_LABS:
    mean = metric_dict[f"{label}_mean"]
    std = metric_dict[f"{label}_std"]
    print(f"{label:<25}{mean:>12.4f}{std:>12.4f}")

print(
    f"{'TrainTime(s)':<25}{metric_dict['TrainTime_mean(s)']:>12.4f}{metric_dict['TrainTime_std(s)']:>12.4f}")
print(
    f"{'TestTime(s)':<25}{metric_dict['TestTime_mean(s)']:>12.4f}{metric_dict['TestTime_std(s)']:>12.4f}")


# === Guardar métricas ===
os.makedirs("Experimentos", exist_ok=True)
csv_path = f"Experimentos/metricas_{args.folder}_{args.ex_name}_{setting[:-2]}.csv"

with open(csv_path, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Metric", "Mean", "STD"])

    for label in METRIC_LABS:
        writer.writerow(
            [label, metric_dict[f"{label}_mean"], metric_dict[f"{label}_std"]])

    writer.writerow(["TrainTime(s)", metric_dict["TrainTime_mean(s)"],
                    metric_dict["TrainTime_std(s)"]])
    writer.writerow(
        ["TestTime(s)", metric_dict["TestTime_mean(s)"], metric_dict["TestTime_std(s)"]])
