import os
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader

from utils.tools import StandardScaler
from utils.timefeatures import time_features
import warnings

warnings.filterwarnings('ignore')


class Dataset_ETT_hour(Dataset):
    """
    Clase que permite el preprocesado del dataset ETT en formato horario.
    Dispone de métodos de lectura, estandarización, y particionado de los datos.
    """

    def __init__(self, root_path, flag='train', size=None, features='S', data_path='ETTh1.csv', target='OT', scale=True,
                 inverse=False, timeenc=0, freq='h', cols=None):

        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 - self.seq_len,
                    12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 +
                    4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        data_stamp = time_features(
            df_stamp, timeenc=self.timeenc, freq=self.freq)

        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):

        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = np.concatenate(
                [self.data_x[r_begin:r_begin + self.label_len], self.data_y[r_begin + self.label_len:r_end]], 0)
        else:
            seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


########################################################################################################################

class Dataset_ETT_minute(Dataset):
    """
    Clase para el preprocesado del dataset de ETT, haciendo uso de intervalos minuto 
    a minuto para mayor granularidad.
    """

    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTm1.csv',
                 target='OT', scale=True, inverse=False, timeenc=0, freq='t', cols=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 *
                    30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
        border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 *
                    30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        data_stamp = time_features(
            df_stamp, timeenc=self.timeenc, freq=self.freq)

        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = np.concatenate(
                [self.data_x[r_begin:r_begin + self.label_len], self.data_y[r_begin + self.label_len:r_end]], 0)
        else:
            seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


########################################################################################################################

class Dataset_Custom(Dataset):
    """
    Plantilla para datasets de tipo personalizado que quieran ser introducidos al modelo de 
    Informer original. Permite establecer la frecuencia de muestreo, el path del fichero, y
    parámetros adicionales como la estandarización o la columna objetivo.
    """

    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, inverse=False, timeenc=0, freq='h', cols=None):
        """

        Constructor básico. Permite la lectura del dataset junto a sus parámetros asociados
        de preprocesado y formateo esenciales. Toma como valores por defecto los de ETT horario.

        Args:
            root_path: Directorio principal de trabajo donde se encuentran todos los ficheros.
            flag: fracción del conjunto de datos que se quiere tomar. Puede ser train, val o test. Defaults to 'train'.
            size: Tamaño del conjunto de datos de entrada, ie, número de tuplas. Defaults to None.
            features: Características a tener en cuenta. Pueden ser un problema univariante (S), multivariante (M),
                        o bien multivariante + target (MS) , . Defaults to 'S'.
            data_path: Nombre del fichero de datos que se encuentra dentro de root_path. Defaults to 'ETTh1.csv'.
            target: Columna que ha de tomarse como atributo objetivo. Defaults to 'OT'.
            scale: Booleano que especifica si se desean o no estandarizar los datos. Defaults to True.
            inverse: Flag que especifica si se desea conservar una copia estandarizada de los valores 
                        para posterior representación. Defaults to False.
            timeenc: Permite especificar si se desea hacer encoding de los timestamps. Defaults to 0.
            freq: Frecuencia de muestreo del dataset. Habitualmente, expresable en horas (h) o minutos (m). Defaults to 'h'.
            cols: Número de columnas del dataset. Defaults to None.
        """
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols = cols
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        """
        Realiza la lectura del fichero que contiene las entradas de datos, teniendo en cuenta la
        frecuencia de muestreo elegida, el tamaño de las particiones, y el número de columnas.
        """
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        # cols = list(df_raw.columns);
        if self.cols:
            cols = self.cols.copy()
            cols.remove(self.target)
        else:
            cols = list(df_raw.columns)
            cols.remove(self.target)
            cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]

        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len,
                    len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        data_stamp = time_features(
            df_stamp, timeenc=self.timeenc, freq=self.freq)

        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        """
        Permite seleccionar una posición concreta del dataset una vez
        leído y preprocesado

        Args:
            index: índice que quiere ser escogido. Este no tiene porqué
            coincider con los índices del dataset original, ya que si este
            fue tomado por minutos,  pero el dataset es procesado por horas, 
            habrá menos tuplas, y no hay correspondencia directa de índice.
            Se trata un valor de índice especifico para los datos ya procesados

        Returns:
            La posición escogida del dataset.
        """
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = np.concatenate(
                [self.data_x[r_begin:r_begin + self.label_len], self.data_y[r_begin + self.label_len:r_end]], 0)
        else:
            seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        """
        Devuelve la longitud del dataset procesado

        Returns:
            Longitud una vez procesado
        """
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        """
        Permite obtener una copia de los datos originales antes de ser
        procesado, para fines de visualización principalmente.

        Args:
            data: Datos de entrada preprocesados

        Returns:
            Copia de los datos originales sin estandarización
        """
        return self.scaler.inverse_transform(data)


########################################################################################################################

class Dataset_Pred(Dataset):
    """
    Permite realizar la predicción para cualquier conjunto de datos de entrada
    Dispone de una estructura bastante similar a la clase de DatasetCustom, pero
    haciendo uso de los valores y patrones de estandarización aplicados en entreamiento
    para evitar DataSnooping. 

    Args:
        Dataset: _description_
    """

    def __init__(self, root_path, flag='pred', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, inverse=False, timeenc=0, freq='15min', cols=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['pred']

        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols = cols
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        """
        Realiza la lectura del fichero que contiene las entradas de datos, teniendo en cuenta la
        frecuencia de muestreo elegida, el tamaño de las particiones, y el número de columnas.
        Dichos parámetros son especificados por la partición de entrenamiento.
        """
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        if self.cols:
            cols = self.cols.copy()
            cols.remove(self.target)
        else:
            cols = list(df_raw.columns)
            cols.remove(self.target)
            cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]

        border1 = len(df_raw) - self.seq_len
        border2 = len(df_raw)

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            self.scaler.fit(df_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        tmp_stamp = df_raw[['date']][border1:border2]
        tmp_stamp['date'] = pd.to_datetime(tmp_stamp.date)
        pred_dates = pd.date_range(
            tmp_stamp.date.values[-1], periods=self.pred_len + 1, freq=self.freq)

        df_stamp = pd.DataFrame(columns=['date'])
        df_stamp.date = list(tmp_stamp.date.values) + list(pred_dates[1:])
        data_stamp = time_features(
            df_stamp, timeenc=self.timeenc, freq=self.freq[-1:])

        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        """
        Permite seleccionar una posición concreta del dataset una vez
        leído y preprocesado

        Args:
            index: índice que quiere ser escogido. Este no tiene porqué
            coincider con los índices del dataset original, ya que si este
            fue tomado por minutos,  pero el dataset es procesado por horas, 
            habrá menos tuplas, y no hay correspondencia directa de índice.
            Se trata un valor de índice especifico para los datos ya procesados

        Returns:
            La posición escogida del dataset.
        """
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = self.data_x[r_begin:r_begin + self.label_len]
        else:
            seq_y = self.data_y[r_begin:r_begin + self.label_len]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        """
        Devuelve la longitud del dataset procesado

        Returns:
            Longitud una vez procesado
        """
        return len(self.data_x) - self.seq_len + 1

    def inverse_transform(self, data):
        """
        Permite obtener una copia de los datos originales antes de ser
        procesado, para fines de visualización principalmente.

        Args:
            data: Datos de entrada preprocesados

        Returns:
            Copia de los datos originales sin estandarización
        """
        return self.scaler.inverse_transform(data)


########################################################################################################################

class Dataset_HPC_hour(Dataset):
    """
    Clase que permite el procesadod el conjunto de datos HPC, proveniente del repositorio UCI. Se encarga de realizar
    el preprocesado adecuado para obtener los timestamps, y realiza el procesado implementado en la clase de referencia
    DatasetCustom.
    """

    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='household_power_consumption.txt',
                 target='Global_active_power', scale=True, inverse=False,
                 timeenc=0, freq='h', cols=None):
        """
        Constructor para el procesado y preparado del conjunto de Household Power Consumption.
        Args:
            root_path: Directorio principal de trabajo donde se encuentran todos los ficheros.
            flag: fracción del conjunto de datos que se quiere tomar. Puede ser train, val o test. Defaults to 'train'.
            size: Tamaño del conjunto de datos de entrada, ie, número de tuplas. Defaults to None.
            features: Características a tener en cuenta. Pueden ser un problema univariante (S), multivariante (M),
                        o bien multivariante + target (MS) , . Defaults to 'S'.
            data_path: Nombre del fichero de datos que se encuentra dentro de root_path. Defaults to 'ETTh1.csv'.
            target: Columna que ha de tomarse como atributo objetivo. Defaults to 'OT'.
            scale: Booleano que especifica si se desean o no estandarizar los datos. Defaults to True.
            inverse: Flag que especifica si se desea conservar una copia estandarizada de los valores
                        para posterior representación. Defaults to False.
            timeenc: Permite especificar si se desea hacer encoding de los timestamps. Defaults to 0.
            freq: Frecuencia de muestreo del dataset. Habitualmente, expresable en horas (h) o minutos (m). Defaults to 'h'.
            cols: Número de columnas del dataset. Defaults to None.
        """
        print("HPC (per hour)")

        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 7
            self.label_len = 24
            self.pred_len = 24
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols = cols

        self.root_path = root_path
        self.data_path = data_path
        print("Preprocesado HPC")
        self.__read_data__()

    def __read_data__(self):
        """
        Realiza la lectura del fichero que contiene las entradas de datos, teniendo en cuenta la
        frecuencia de muestreo elegida, el tamaño de las particiones, y el número de columnas.
        """
        # Preparamos el escalado
        self.scaler = StandardScaler()

        # Leemos los datos y unimos las fechas en una sola columna
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path), sep=';',
                             parse_dates={'date': ['Date', 'Time']}, infer_datetime_format=True, low_memory=False,
                             na_values=['?', ''])
        df_raw.set_index('date', inplace=True)
        df_raw.fillna(method='ffill', inplace=True)

        # Transformamos a valores numéricos
        for col in df_raw.columns:
            df_raw[col] = pd.to_numeric(df_raw[col], errors='coerce')

        # Agrupamos por horas y reestablecemos el índice para que sea creciente desde 0
        df_raw = df_raw.resample('H').mean()
        df_raw = df_raw.reset_index()

        # Filtrado de columnas
        if self.features in ['M', 'MS']:
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        # Dividimos en train/val/test mediante porcentajes
        n = len(df_data)
        n_train = int(n * 0.7)
        n_val = int(n * 0.2)
        borders1 = [0, n_train - self.seq_len, n_train + n_val - self.seq_len]
        borders2 = [n_train, n_train + n_val, n]
        b1 = borders1[self.set_type]
        b2 = borders2[self.set_type]

        # Escalado
        if self.scale:
            train_data = df_data.iloc[0:n_train]
            self.scaler.fit(train_data.values)
            data_values = self.scaler.transform(df_data.values)
        else:
            data_values = df_data.values
        # Time features
        df_stamp = df_data.index[b1:b2].to_frame(index=False, name='date')
        data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq)

        # Asignamos los atributos
        self.data_x = data_values[b1:b2]
        self.data_y = (df_data.values[b1:b2] if self.inverse else data_values[b1:b2])
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        """
        Permite seleccionar una posición concreta del dataset una vez
        leído y preprocesado

        Args:
            index: índice que quiere ser escogido. Este no tiene porqué
            coincider con los índices del dataset original, ya que si este
            fue tomado por minutos, pero el dataset es procesado por horas,
            habrá menos tuplas, y no hay correspondencia directa de índice.
            Se trata un valor de índice especifico para los datos ya procesados

        Returns:
            La posición escogida del dataset.
        """
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = np.concatenate(
                [self.data_x[r_begin:r_begin + self.label_len], self.data_y[r_begin + self.label_len:r_end]], 0)
        else:
            seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        """
        Devuelve la longitud del dataset procesado

        Returns:
            Longitud una vez procesado
        """
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        """
        Permite obtener una copia de los datos originales antes de ser
        procesado, para fines de visualización principalmente.

        Args:
            data: Datos de entrada preprocesados

        Returns:
            Copia de los datos originales sin estandarización
        """
        return self.scaler.inverse_transform(data)


########################################################################################################################
class Dataset_HPC_minute(Dataset):
    """
    Clase que permite el procesadod el conjunto de datos HPC, proveniente del repositorio UCI. Se encarga de realizar
    el preprocesado adecuado para obtener los timestamps, y realiza el procesado implementado en la clase de referencia
    DatasetCustom.
    """

    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='household_power_consumption.txt',
                 target='Global_active_power', scale=True, inverse=False,
                 timeenc=0, freq='t', cols=None):
        """
        Constructor para el procesado y preparado del conjunto de Household Power Consumption.
        Args:
            root_path: Directorio principal de trabajo donde se encuentran todos los ficheros.
            flag: fracción del conjunto de datos que se quiere tomar. Puede ser train, val o test. Defaults to 'train'.
            size: Tamaño del conjunto de datos de entrada, ie, número de tuplas. Defaults to None.
            features: Características a tener en cuenta. Pueden ser un problema univariante (S), multivariante (M),
                        o bien multivariante + target (MS) , . Defaults to 'S'.
            data_path: Nombre del fichero de datos que se encuentra dentro de root_path. Defaults to 'ETTh1.csv'.
            target: Columna que ha de tomarse como atributo objetivo. Defaults to 'OT'.
            scale: Booleano que especifica si se desean o no estandarizar los datos. Defaults to True.
            inverse: Flag que especifica si se desea conservar una copia estandarizada de los valores
                        para posterior representación. Defaults to False.
            timeenc: Permite especificar si se desea hacer encoding de los timestamps. Defaults to 0.
            freq: Frecuencia de muestreo del dataset. Habitualmente, expresable en horas (h) o minutos (t). Defaults to 'h'.
            cols: Número de columnas del dataset. Defaults to None.
        """
        # size [seq_len, label_len, pred_len]
        # info
        print("HPC (per minute)")
        if size == None:
            self.seq_len = 24 * 60
            self.label_len = 60
            self.pred_len = 60
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols = cols

        self.root_path = root_path
        self.data_path = data_path
        print("Preprocesado HPC")
        self.__read_data__()

    def __read_data__(self):
        """
        Realiza la lectura del fichero que contiene las entradas de datos, teniendo en cuenta la
        frecuencia de muestreo elegida, el tamaño de las particiones, y el número de columnas.
        """
        # Preparamos el escalado
        self.scaler = StandardScaler()

        # Leemos los datos y unimos las fechas en una sola columna
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path), sep=';',
                             parse_dates={'date': ['Date', 'Time']}, infer_datetime_format=True, low_memory=False,
                             na_values=['?', ''])
        df_raw.set_index('date', inplace=True)
        df_raw.fillna(method='ffill', inplace=True)

        # Transformamos a valores numéricos
        for col in df_raw.columns:
            df_raw[col] = pd.to_numeric(df_raw[col], errors='coerce')

        # Agrupamos por horas y reestablecemos el índice para que sea creciente desde 0
        df_raw = df_raw.reset_index()

        # Filtrado de columnas
        if self.features in ['M', 'MS']:
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        # Dividimos en train/val/test mediante porcentajes
        n = len(df_data)
        n_train = int(n * 0.7)
        n_val = int(n * 0.2)
        borders1 = [0, n_train - self.seq_len, n_train + n_val - self.seq_len]
        borders2 = [n_train, n_train + n_val, n]
        b1 = borders1[self.set_type]
        b2 = borders2[self.set_type]

        # Escalado
        if self.scale:
            train_data = df_data.iloc[0:n_train]
            self.scaler.fit(train_data.values)
            data_values = self.scaler.transform(df_data.values)
        else:
            data_values = df_data.values
        # Time features
        df_stamp = df_data.index[b1:b2].to_frame(index=False, name='date')
        data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq)

        # Asignamos los atributos
        self.data_x = data_values[b1:b2]
        self.data_y = (df_data.values[b1:b2] if self.inverse else data_values[b1:b2])
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        """
        Permite seleccionar una posición concreta del dataset una vez
        leído y preprocesado

        Args:
            index: índice que quiere ser escogido. Este no tiene porqué
            coincider con los índices del dataset original, ya que si este
            fue tomado por minutos, pero el dataset es procesado por horas,
            habrá menos tuplas, y no hay correspondencia directa de índice.
            Se trata un valor de índice especifico para los datos ya procesados

        Returns:
            La posición escogida del dataset.
        """
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = np.concatenate(
                [self.data_x[r_begin:r_begin + self.label_len], self.data_y[r_begin + self.label_len:r_end]], 0)
        else:
            seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        """
        Devuelve la longitud del dataset procesado

        Returns:
            Longitud una vez procesado
        """
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        """
        Permite obtener una copia de los datos originales antes de ser
        procesado, para fines de visualización principalmente.

        Args:
            data: Datos de entrada preprocesados

        Returns:
            Copia de los datos originales sin estandarización
        """
        return self.scaler.inverse_transform(data)
