# Estudio sobre la Efectividad del Positional Encoding en Transformers para Series Temporales y Diseño de Mecanismos Adaptados

![License](https://img.shields.io/badge/license-Apache%202.0-orange.svg?style=for-the-badge)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)

## Propuesta
Los modelos Transformer, exitosos en procesamiento del lenguaje natural y visión por computador, están siendo empleados en el análisis de series temporales (forecasting, detección de anomalías) [1]. Sin embargo, la estructura única de los datos temporales (ciclos, estacionalidad) difiere del lenguaje, cuestionando si los mecanismos de positional encoding son idóneos para este dominio. Investigaciones previas ya sugieren que su efectividad en series temporales podría ser limitada [2]. Este proyecto propone analizar los experimentos publicados para validar (o refutar) esos hallazgos en diferentes dominios de aplicación (finanzas, energía, monitorización industrial) y con múltiples métricas de precisión y detección de anomalías.

Exploraremos una alternativa de positional encoding diseñada expresamente para su aplicación en el dominio de las series temporales. Se estudiarán distintas alternativas, tanto positional encodings al uso como arquitecturas específicas que integren estos principios. Se evaluará su impacto sobre la calidad de las predicciones y la detección temprana de anomalías, comparándola frente a los métodos estándar y analizando ventajas e inconvenientes en cuanto a interpretabilidad, capacidad de generalización y coste computacional.


## Objetivos

Realizar una revisión exhaustiva del estado del arte sobre positional encoding y su aplicación en modelos Transformer para series temporales.

Evaluar sistemáticamente la efectividad de los positional encoding estándar en tareas de forecasting y detección de anomalías sobre benchmarks de series temporales.

Estudio de propuestas originales de positional encoding o arquitecturas completas adaptadas a series temporales, explorando vías alternativas al estado del arte.


## Ejecución de los experimentos

Para realizar una comparativa de los diferentes métodos, se han creado un fichero .py que recibe multitud de parámetros de entrada para configurar el modelo como se considere oportuno, pudiendo especificar los diferentes tipos de PE y sus hiperparámetros asociados.
Se trata del fichero `experimentacion.py`.


### Configuración general
--model: Tipo de modelo a usar (informer, informerstack).

--ex_name: Nombre del experimento.

--folder: Carpeta donde se guarda el modelo.

--data: Nombre del dataset.

--root_path: Ruta raíz donde se encuentra el dataset.

--data_path: Nombre del archivo de datos.

--features: Tipo de predicción: M (multi→multi), S (uni→uni), MS (multi→uni).

--target: Variable objetivo para tareas S o MS.

--freq: Frecuencia temporal para codificación (horas: h; minutos: t; segundos: s).

--checkpoints: Ruta para guardar checkpoints del modelo.

### Longitudes de entrada y salida
--seq_len: Longitud de secuencia de entrada del encoder.

--label_len: Longitud del token de inicio del decoder.

--pred_len: Longitud de la secuencia a predecir.

### Configuración del modelo
--enc_in: Número de variables de entrada al encoder.

--dec_in: Número de variables de entrada al decoder.

--c_out: Número de salidas del modelo.

--d_model: Dimensión del modelo.

--n_heads: Número de cabezas en el multi-head attention.

--e_layers: Número de capas en el encoder.

--d_layers: Número de capas en el decoder.

--s_layers: Capas apiladas en encoder (solo stack mode).

--d_ff: Tamaño del feed-forward interno.

--factor: Factor de reducción en atención probabilística.

--padding: Tipo de padding (0: none, 1: same).

--distil: Desactiva el distilling si se incluye.

--dropout: Tasa de dropout.

--attn: Tipo de atención en encoder (full para evitar pérdida de información).

### Codificación temporal
--time_encoding: Tipo de codificación posicional/temporal (ver abajo).

--embed: Tipo de embedding temporal (**timeF**, fixed, learned).

--activation: Función de activación (e.g., gelu, relu).

--window: Tamaño de ventana para estadísticas.

--output_attention: Muestra la atención generada por el encoder..

--cols: Columnas específicas del dataset a usar.

### Entrenamiento y ejecución
--num_workers: Núm. de workers para DataLoader.

--itr: Número de repeticiones del experimento.

--train_epochs: Número de épocas de entrenamiento.

--batch_size: Tamaño de batch para entrenamiento (32).

--patience: Paciencia para early stopping (3).

--learning_rate: Tasa de aprendizaje.

--des: Descripción del experimento.

--loss: Función de pérdida (mse, mae, etc.).

--lradj: Estrategia de ajuste de learning rate.

--use_amp: Usa entrenamiento con precisión mixta (AMP).

--inverse: Invierte la transformación de salida.

--shuffle_decoder_input: Mezcla entradas del decoder durante test.

### GPU
--use_gpu: Usa GPU si está disponible.

--gpu: Índice de GPU a usar.

--use_multi_gpu: Activa uso de múltiples GPUs.

--devices: IDs de las GPUs a usar.

### Tipos de PE: --time_encoding

| Valor             | Descripción                                                       |
|-------------------|------------------------------------------------------------------|
| `no_pe`           | Sin codificación posicional, se usan solo los datos de entrada.      |
| `informer`        | Codificación temporal original de Informer.|
| `stats`           | Codificación basada en estadísticas por ventana temporal deslizante, que calcula media, std y valores extremos.        |
| `stats_lags`      | Igual que `stats`, pero incluye lags como contexto local.    |
| `all_pe_weighted` | Combinación de lo anterior, junto a PE fijos y PE aprendibles (LPE), ponderados con pesos normalizados mediante Softmax.   |
| `tpe`             | Codificación temporal haciendo uso de temporal PE, t-PE, para aportar mayor información local. Contiene la información de lags, ventana y otros PE fijos, haciendo uso de pesos aprendidos y normalizados mediante Softmax. |


> Puede encontrar un ejemplo de ejecución dentro del fichero slurm_task.sh


## Entorno

Para la ejecución de este proyecto, es necesario disponer de un entorno actualizado con Pytorch (Python 3.12). Para ello, puede usarse el fichero `requirements.txt`.

> conda create --name <env> --file requirements.txt


## Referencias

[1] Wen, Q., Zhou, T., Zhang, C., Chen, W., Ma, Z., Yan, J., & Sun, L. (2022). Transformers in time series: A survey. arXiv preprint arXiv:2202.07125.

[2] Zeng, A., Chen, M., Zhang, L., & Xu, Q. (2023, June). Are transformers effective for time series forecasting?. In Proceedings of the AAAI conference on artificial intelligence (Vol. 37, No. 9, pp. 11121-11128).

[3] Haoyi Zhou, Shanghang Zhang, Jieqi Peng, Shuai Zhang, Jianxin Li, Hui Xiong y Wancai Zhang. "Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting." The Thirty-Fifth {AAAI} Conference on Artificial Intelligence, {AAAI} 2021, Virtual Conference, pp. 11106–11115. AAAI Press, 2021.
