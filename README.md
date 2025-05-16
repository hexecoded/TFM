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

[1] Wen, Q., Zhou, T., Zhang, C., Chen, W., Ma, Z., Yan, J., & Sun, L. (2022). Transformers in time series: A survey. arXiv preprint arXiv:2202.07125.

[2] Zeng, A., Chen, M., Zhang, L., & Xu, Q. (2023, June). Are transformers effective for time series forecasting?. In Proceedings of the AAAI conference on artificial intelligence (Vol. 37, No. 9, pp. 11121-11128).

[3] Haoyi Zhou, Shanghang Zhang, Jieqi Peng, Shuai Zhang, Jianxin Li, Hui Xiong y Wancai Zhang. "Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting." The Thirty-Fifth {AAAI} Conference on Artificial Intelligence, {AAAI} 2021, Virtual Conference, pp. 11106–11115. AAAI Press, 2021.
