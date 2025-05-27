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

METRIC_LABS = ["MAE", "MSE", "RMSE", "MAPE", "MSPE"]
# Establecemos el nombre del modelo, ie, su nombre en carpeta


# Inserción de parámetros. Se usa el formato argparse para obtener realimentación en caso de requerir ayuda para conocer
# las opciones disponibles
parser = argparse.ArgumentParser(description='[Experimentación Informer] Long Sequences Forecasting')

parser.add_argument('--model', type=str, required=True, default='informer',
                    help='model of experiment, options: [informer, informerstack, informerlight(TBD)]')
parser.add_argument('--folder', type=str, required=True, default='InformerPE', help='model folder for experiment')
parser.add_argument('--data', type=str, required=True, default='ETTh1', help='data')
parser.add_argument('--root_path', type=str, default='./Datasets/ETT-small', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
parser.add_argument('--features', type=str, default='M',
                    help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
parser.add_argument('--freq', type=str, default='h',
                    help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

parser.add_argument('--seq_len', type=int, default=96, help='input sequence length of Informer encoder')
parser.add_argument('--label_len', type=int, default=48, help='start token length of Informer decoder')
parser.add_argument('--pred_len', type=int, default=24, help='prediction sequence length')

parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
parser.add_argument('--c_out', type=int, default=7, help='output size')
parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--s_layers', type=str, default='3,2,1', help='num of stack encoder layers')
parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
parser.add_argument('--factor', type=int, default=5, help='probsparse attn factor')
parser.add_argument('--padding', type=int, default=0, help='padding type')
parser.add_argument('--distil', action='store_false',
                    help='whether to use distilling in encoder, using this argument means not using distilling',
                    default=True)
parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
parser.add_argument('--attn', type=str, default='prob', help='attention used in encoder, options:[prob, full]')
parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in encoder')
parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')
parser.add_argument('--mix', action='store_false', help='use mix attention in generative decoder', default=True)
parser.add_argument('--cols', type=str, nargs='+', help='certain cols from the data files as the input features')
parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
parser.add_argument('--itr', type=int, default=1, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=100, help='train epochs')
parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='test', help='exp description')
parser.add_argument('--loss', type=str, default='mse', help='loss function')
parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)

parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

args = parser.parse_args()

# Cargamos el modelo seleccionado
sys.path += [args.folder]
from exp.exp_informer import Exp_Informer

# Comprobación de uso de CUDA para el cómputo
args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
print(f"Usando CUDA: {torch.cuda.is_available()}")

if args.use_gpu and args.use_multi_gpu:
    args.devices = args.devices.replace(' ', '')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]

# Datasets Disponibles en el paper de Informer
data_parser = {
    'HPC': {'data': 'household_power_consumption.txt', 'T': 'OT', 'M': [7, 7, 7], 'S': [1, 1, 1], 'MS': [7, 7, 1]},
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
Exp = Exp_Informer

for ii in range(args.itr):
    print("========================= Ejecución {} =========================".format(ii))
    # setting record of experiments
    setting = '{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_at{}_fc{}_eb{}_dt{}_mx{}_{}_{}'.format(args.model,
                                                                                                         args.data,
                                                                                                         args.features,
                                                                                                         args.seq_len,
                                                                                                         args.label_len,
                                                                                                         args.pred_len,
                                                                                                         args.d_model,
                                                                                                         args.n_heads,
                                                                                                         args.e_layers,
                                                                                                         args.d_layers,
                                                                                                         args.d_ff,
                                                                                                         args.attn,
                                                                                                         args.factor,
                                                                                                         args.embed,
                                                                                                         args.distil,
                                                                                                         args.mix,
                                                                                                         args.des, ii)

    # Insertamos la información del experimento a realizar
    exp = Exp(args)
    # Entrenamiento
    print("Entrenamiento")
    exp.train(setting)
    # Evaluación
    print("Evaluación")
    exp.test(setting)

    # Limpieza de cachés para nueva ejecición desde 0.
    torch.cuda.empty_cache()

# Recopilamos los experimentos realizados

metrics = np.zeros(5)

for i in range(args.itr):
    datos = np.load("results/" + setting[:-1] + str(i) + "/metrics.npy")
    metrics += np.array(datos)

# Mostramos la media de las métricas evaluadas
print("========================= Resultados del experimento =========================")

# Calculamosla media
mean = metrics / args.itr
metric_dict = {label: mean[i] for i, label in enumerate(METRIC_LABS)}

# Mostramos las métricas
for label, value in metric_dict.items():
    print(f"{label} >> {value}")

# Guardado en disco de las métricas
with open(f"results/metricas_{setting[:-2]}.csv", mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Métrica", "Valor"])
    for label, value in metric_dict.items():
        writer.writerow([label, value])
