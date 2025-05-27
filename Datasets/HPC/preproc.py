"""
Preprocesado del dataset de Household Power Consumption para
cumplir los formatos de fecha estándar.

https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption


Genera un archivo de salida, HPC.csv, ya estandarizado y preparado para pasar
por parámetros a un modelo de aprendizaje de TS.

Autor: Cristhian Moya Mota
"""


import pandas as pd

# Lectura del archivo CSV
df = pd.read_csv("household_power_consumption.txt", sep=';')

# Unimos fecha y hora en una sola columna tipo datetime
df['date'] = pd.to_datetime(
    df['Date'] + ' ' + df['Time'], format='%d/%m/%Y %H:%M:%S')

# Cambio de formato a 'YYYY-MM-DD HH:MM:SS'
df['date'] = df['date'].dt.strftime('%Y-%m-%d %H:%M:%S')
df = df.drop(columns=['Date', 'Time'])

# Guardamos en CSV
df.to_csv("HPC.csv", index=False)
