import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular')
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        # x: [B, L, D]
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(
            1, 2)  # [B, L, d_model]
        return x


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class FourierEncoding(nn.Module):
    """
    Clase para la codificación adicional de información posicional mediante el
    uso de la transformada de Fourier.
    Se presenta como alternativa al encoding sin/cos habitual en transformers,
    descomponiendo la serie y tomando la frecuencia para extraer nuevas características.
    """

    def __init__(self, d_model):
        super(FourierEncoding, self).__init__()
        self.d_model = d_model

    def forward(self, x):
        """
        Realiza la descomposición de Fourier y la selección  de la 
        frecuencia, que es luego interpolada para ser de la misma longitud que
        la serie original.

        Args:
            x: datos de entrada para ser procesados en cada iteración

        Returns:
            información extraída de la serie interpolada para poder ser
            sumada a la original a modo de PE.
        """
        # x: [B, L, d_model]
        # FFT sobre la dimensión temporal (L)
        x_fft = torch.fft.rfft(x, dim=1, norm='ortho')  # [B, L//2+1, d_model]
        # Tomamos el módulo (magnitude)
        x_freq = x_fft.abs()  # [B, L//2+1, d_model]
        # Reescalamos para que tenga longitud L
        x_freq = F.interpolate(x_freq.permute(0, 2, 1), size=x.size(
            1), mode='linear', align_corners=False)
        return x_freq.permute(0, 2, 1)  # [B, L, d_model]


class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()


class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, freq='h'):
        super(TemporalEmbedding, self).__init__()

        minute_size = 4
        half_minute_size = 2
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13

        Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding
        if freq == 't':
            self.minute_embed = Embed(minute_size, d_model)
        elif freq == '30s':
            self.half_minute_embed = Embed(half_minute_size, d_model)

        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)


def forward(self, x):
    x = x.long()

    minute_x = self.minute_embed(x[:, :, 4]) if hasattr(
        self, 'minute_embed') else 0.
    half_minute_x = self.half_minute_embed(
        x[:, :, 5]) if hasattr(self, 'half_minute_embed') else 0.
    hour_x = self.hour_embed(x[:, :, 3])
    weekday_x = self.weekday_embed(x[:, :, 2])
    day_x = self.day_embed(x[:, :, 1])
    month_x = self.month_embed(x[:, :, 0])

    return hour_x + weekday_x + day_x + month_x + minute_x + half_minute_x


class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super(TimeFeatureEmbedding, self).__init__()

        # Agregamos soporte para '30s'
        freq_map = {
            'h': 4,
            't': 5,
            's': 7,
            '30s': 7,
            'm': 1,
            'a': 1,
            'w': 2,
            'd': 3,
            'b': 3
        }

        if freq not in freq_map:
            raise ValueError(
                f"[TimeFeatureEmbedding] Frecuencia no soportada: {freq}")

        d_inp = freq_map[freq]
        self.freq = freq
        self.embed = nn.Linear(d_inp, d_model)

    def forward(self, x):
        """
        Espera que `x` tenga las características temporales concatenadas como:
        [minute, half_minute, hour, day, weekday, month] para freq='30s'
        """
        return self.embed(x)


class RollingFeatureExtractor(nn.Module):
    """
    Clase para el cálculo de estadísticos básicos que funcionan como encoding
    relativo usando una ventana deslizante alrededor de cada valor.
    """

    def __init__(self, window_size: int, input_features: int):
        super(RollingFeatureExtractor, self).__init__()
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        self.window_size = window_size
        self.input_features = input_features
        # El número de características de salida será el original + (media, max, min, std) * input_features
        self.output_features = input_features * 5  # Original + media, max, min, std_dev

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calcula la media, el máximo, el mínimo y la desviación estándar en una ventana deslizante.

        Args:
            x (torch.Tensor): tensor de entrada (batch_size, sequence_length, num_features).

        Returns:
            torch.Tensor: tensor con características añadidas, con forma (batch_size, sequence_length, num_features * 5).
        """
        batch_size, seq_len, num_features = x.shape
        if num_features != self.input_features:
            raise ValueError(
                f"Input features mismatch. Expected {self.input_features}, got {num_features}.")

        # Para manejar el inicio de la secuencia donde no hay suficientes valores, se aplica padding del mismo
        # valor que dicho extremo, pues los 0 provocarían mayor distorsión de los resultados.
        # Pad en la dimensión de secuencia
        padded_x = F.pad(x, (0, 0, self.window_size - 1, 0), mode='replicate')

        rolling_features = []

        for i in range(seq_len):
            # La ventana se extiende desde (i) hasta (i + window_size - 1) en el tensor padded_x
            # Esto es equivalente a la ventana (i - window_size + 1) hasta (i) en el tensor original
            window = padded_x[:, i: i + self.window_size, :]

            # Calculamos los estadísticos
            window_mean = torch.mean(window, dim=1, keepdim=True)
            window_max = torch.max(window, dim=1, keepdim=True).values
            window_min = torch.min(window, dim=1, keepdim=True).values
            window_std = torch.std(window, dim=1, keepdim=True)

            # Concatenamos las estadísticas para esta posición de tiempo
            # Forma: (batch_size, 1, num_features * 4)
            current_rolling_stats = torch.cat(
                [window_mean, window_max, window_min, window_std], dim=-1)
            rolling_features.append(current_rolling_stats)

        # Concatenamos todas las estadísticas. Forma: (batch_size, seq_len, num_features * 4)
        rolling_stats_tensor = torch.cat(rolling_features, dim=1)
        output_tensor = torch.cat([x, rolling_stats_tensor], dim=-1)

        return output_tensor


class tAPE(nn.Module):
    """
    Clase que implementa la codificación posicional tAPE
    (Time Absolute Position Encoding) que tiene como novedad
    usar la longitud de secuencia para tenerla en cuenta en la
    codificación sinusoidal
    """

    def __init__(self, d_model, max_len=5000):
        super(tAPE, self).__init__()
        self.pos_embedding = nn.Embedding(max_len, d_model)
        self.token_proj = nn.Linear(d_model, d_model)
        self.mixer = nn.Linear(2 * d_model, d_model)

    def forward(self, token_embeds, position_ids):
        # token_embeds: [B, L, d_model]
        # position_ids: [B, L]
        pos_embeds = self.pos_embedding(position_ids)
        token_proj = self.token_proj(token_embeds)
        mixed = torch.cat([token_proj, pos_embeds], dim=-1)
        return self.mixer(mixed)


class LearnablePositionalEncoding(nn.Module):
    """
    Learnable positional encoding para modelos tipo Transformer/Informer.
    En lugar de usar codificación sinusoidal fija, se aprende directamente
    un tensor de posiciones.

    Args:
        d_model (int): Dimensión del embedding de entrada.
        max_len (int): Longitud máxima de secuencia que se puede codificar.
    """

    def __init__(self, d_model, max_len=5000):
        super(LearnablePositionalEncoding, self).__init__()

        # Inicialización normal truncada con std ≈ 1/sqrt(d_model)
        std = 1.0 / math.sqrt(d_model)
        self.pe = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.trunc_normal_(self.pe, mean=0.0, std=std,
                              a=-2 * std, b=2 * std)

    def forward(self, x):
        """
        Agrega el encoding posicional aprendido a la entrada `x`.

        Args:
            x (Tensor): Entrada de forma (batch_size, seq_len, d_model)
        Returns:
            Tensor: Codificación posicional de forma (1, seq_len, d_model)
        """
        seq_len = x.size(1)
        return self.pe[:, :seq_len]


class TemporalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, sigma=1.0):
        super(TemporalPositionalEncoding, self).__init__()
        self.d_model = d_model
        self.sigma = sigma

        # Componente geométrico (sinusoidal)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(
            0, max_len, dtype=torch.float).unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)  # par
        pe[:, 1::2] = torch.cos(position * div_term)  # impar
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x: [B, L, D]
        returns: [B, L, D] with TPE applied
        """
        B, L, D = x.size()
        device = x.device

        # 1. Componente geométrica
        pe = self.pe[:, :L, :]  # [1, L, D]

        # 2. Componente semántica S(i,j) ≈ media por fila
        # Calcular distancia entre cada punto de la secuencia: ||xi - xj||^2
        x_exp1 = x.unsqueeze(2)  # [B, L, 1, D]
        x_exp2 = x.unsqueeze(1)  # [B, 1, L, D]
        dist_sq = ((x_exp1 - x_exp2) ** 2).sum(dim=-1)  # [B, L, L]

        # Similaridad gaussiana
        S = torch.exp(-dist_sq / (2 * self.sigma ** 2))  # [B, L, L]

        # Sumar sobre posiciones j para obtener vector de enriquecimiento semántico por i
        sem = S @ x / S.sum(dim=-1, keepdim=True)  # [B, L, D]

        # Combinar
        return x + pe + sem


# ---------- Implementaciones de Embedings ------------

class DataEmbeddingNoPE(nn.Module):
    """
    Clase que no incluye ningún tipo de PE en los datos más allá de sus
    valores de cada instancia.
    """

    def __init__(self, c_in, d_model,  freq='h', dropout=0.1, window=24, lags=[3, 5, 7],
                 max_len=5000):
        super(DataEmbeddingNoPE, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = self.value_embedding(x)
        return self.dropout(x)


class DataEmbedding_Informer(nn.Module):
    """
        Clase que permite la construcción de embeddings para el modelo de Informer original
    """

    def __init__(self, c_in, d_model,  freq='h', dropout=0.1, window=24, lags=[3, 5, 7],
                 max_len=5000, embed_type="timeF"):
        super(DataEmbedding_Informer, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = self.value_embedding(
            x) + self.position_embedding(x) + self.temporal_embedding(x_mark)

        return self.dropout(x)


class DataEmbedding_Stats(nn.Module):
    """
    Clase que permite la construcción de embeddings para el modelo de Informer 
    usando información del valor de cada instancia, así como diferentes métodos
    de PE ponderados para encontrar aquel que ofrece mejores rsultados
    """

    def __init__(self, c_in, d_model,  freq='h', dropout=0.1, window=24, lags=[3, 5, 7]):
        super(DataEmbedding_Stats, self).__init__()
        print("Window size: ", window)

        # Cálculo de estadísticas
        self.est_features = RollingFeatureExtractor(window, c_in)

        self.value_embedding_combined = TokenEmbedding(
            c_in=c_in * 5,
            d_model=d_model
        )

        self.dropout = nn.Dropout(p=dropout * 0.25)
        self.cont = 0

    def forward(self, x, x_mark):
        """
        Función forward para la construcción del embedding.
        """

        # Extracción de estadísticas
        x_stats = self.est_features(x)  # [B, L, C*5]

        # Proyección a espacio de embedding
        x_emb = self.value_embedding_combined(x_stats)  # [B, L, d_model]

        return self.dropout(x_emb)


class DataEmbedding_StatsLags(nn.Module):
    """
    Clase que permite la construcción de embeddings para el modelo de Informer 
    usando información del valor de cada instancia, así como diferentes métodos
    de PE ponderados para encontrar aquel que ofrece mejores rsultados

    """

    def __init__(self, c_in, d_model,  freq='h', dropout=0.1, window=24, lags=[3, 5, 7]):
        super(DataEmbedding_StatsLags, self).__init__()
        print("Window size: ", window)

        # Estadísticas + lags concatenados en una sola rama
        self.est_features = RollingFeatureExtractor(window, c_in)
        self.lags = lags
        self.value_embedding_combined = TokenEmbedding(
            c_in=c_in * (5 + len(lags)), d_model=d_model
        )

        self.dropout = nn.Dropout(p=dropout * 0.25)
        self.cont = 0

    def compute_lags(self, x):
        """
        Dado un conjunto de instancias de entrada, calcula la diferencia entre lags, dando
        como entrada una lista de elementos que indique en que t-n puntos evaluar la 
        diferencia.

        Args:
            x: conjunto de datos de entrada

        Returns:
            lags especificados concatenados en un único tensor
        """
        B, L, C = x.size()
        max_lag = max(self.lags)
        x_padded = F.pad(x, (0, 0, max_lag, 0), mode='replicate')
        lag_diffs = []
        for lag in self.lags:
            x_lagged = x_padded[:, max_lag - lag: max_lag - lag + L, :]
            lag_diffs.append(x - x_lagged)  # Diferencias
        return torch.cat(lag_diffs, dim=-1)  # [B, L, C * len(lags)]

    def forward(self, x, x_mark):
        """
        Función forward para la construcción del embedding. Evalúa los elementos
        necesarios y ajusta los pesos asociados a cada elemento del embedding 
        aditivo que se construye

        Args:
            x: conjunto de datos de entrada
            x_mark: conjunto de datos con información temporal asociada. No utilizada
                    cuando se elimina la codificación posicional global proporcionada
                    en el modelo Informer original.

        Returns:
            Elementos ya procesados, sumados y con dropout aplicado para evitar sobreaprendizaje.
        """

        # Concatenación de features
        x_stats = self.est_features(x)
        x_lags = self.compute_lags(x)
        x_combined = torch.cat([x_stats, x_lags], dim=-1)
        combined_emb = self.value_embedding_combined(x_combined)

        return self.dropout(combined_emb)


class DataEmbedding_ALLPE_Weighted(nn.Module):
    """
        Clase que permite la construcción de embeddings para el modelo de Informer
        usando información del valor de cada instancia, así como diferentes métodos
        de PE ponderados para encontrar aquel que ofrece mejores rsultados
    """

    def __init__(self, c_in, d_model,  freq='h', dropout=0.1, window=24, lags=[3, 5, 7],
                 max_len=5000, pos_enc="window", embed_type="timeF"):
        super(DataEmbedding_ALLPE_Weighted, self).__init__()
        self.pos_enc = pos_enc
        self.est_features = RollingFeatureExtractor(window, c_in)
        self.lags = lags
        self.value_embedding_combined = TokenEmbedding(
            c_in=c_in * (5 + len(lags)), d_model=d_model
        )

        # Fix here: max_len first, d_model second
        self.fixed_pe = FixedEmbedding(max_len, d_model)
        self.learned_pe = LearnablePositionalEncoding(d_model, max_len)
        self.tape = tAPE(d_model, max_len)

        self.weight_params = nn.Parameter(torch.tensor(
            [0.25, 0.25, 0.25, 0.25], dtype=torch.float32
        ))

        self.norm_combined = nn.LayerNorm(d_model)
        self.norm_fixed = nn.LayerNorm(d_model)
        self.norm_learned = nn.LayerNorm(d_model)
        self.norm_tape = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(p=dropout * 0.25)
        self.cont = 0

    def compute_lags(self, x):
        """
            Dado un conjunto de instancias de entrada, calcula la diferencia entre lags, dando
            como entrada una lista de elementos que indique en que t-n puntos evaluar la
            diferencia.

            Args:
                x: conjunto de datos de entrada

            Returns:
                lags especificados concatenados en un único tensor
        """
        B, L, C = x.size()
        max_lag = max(self.lags)
        x_padded = F.pad(x, (0, 0, max_lag, 0), mode='replicate')
        lag_diffs = [x - x_padded[:, max_lag - lag:max_lag - lag + L, :]
                     for lag in self.lags]
        return torch.cat(lag_diffs, dim=-1)

    def forward(self, x, x_mark=None):
        """
            Función forward para la construcción del embedding. Evalúa los elementos
            necesarios y ajusta los pesos asociados a cada elemento del embedding
            aditivo que se construye

            Args:
                x: conjunto de datos de entrada
                x_mark: conjunto de datos con información temporal asociada. No utilizada
                        cuando se elimina la codificación posicional global proporcionada
                        en el modelo Informer original.

            Returns:
                Elementos ya procesados, sumados y con dropout aplicado para evitar sobreaprendizaje.
        """
        self.cont += 1
        x_stats = self.est_features(x)
        x_lags = self.compute_lags(x)
        x_combined = torch.cat([x_stats, x_lags], dim=-1)
        combined_emb = self.value_embedding_combined(x_combined)

        B, L, _ = x.size()
        device = x.device
        position_ids = torch.arange(
            L, device=device).unsqueeze(0).expand(B, -1)

        # Normalizacion
        combined_emb = self.norm_combined(combined_emb)
        pe_fixed = self.norm_fixed(self.fixed_pe(position_ids))
        pe_learned = self.norm_learned(self.learned_pe(x))
        pe_tape = self.norm_tape(self.tape(combined_emb, position_ids))

        weights = F.softmax(self.weight_params, dim=0)

        out = (
            weights[0] * combined_emb +
            weights[1] * pe_fixed +
            weights[2] * pe_learned +
            weights[3] * pe_tape
        )

        if self.cont % 200 == 0:
            self.print_weights()

        return self.dropout(out)

    def print_weights(self, epoch=None):
        """
            Imprime la evolución de los pesos entrenados para cada componente del embedding:
            - Combinado de estadísticas y lags
            - Posicional fijo (sinusoidal)
            - tAPE (token-aware positional encoding)

            Args:
                epoch: Época actual de entrenamiento, si está disponible. Defaults to None.
        """
        weights = F.softmax(self.weight_params, dim=0)
        msg = f"\t[Epoch {epoch}] " if epoch else ""
        print(
            f"\t{msg}Weights => Stats: {weights[0]:.4f}, PE: {weights[1]:.4f}, LPE: {weights[2]:.4f}, tAPE: {weights[3]:.4f}")


class DataEmbedding_TPE(nn.Module):
    """
    Clase de embeding que implementa TPE (Temporal Positional Encoding) junto al resto de características extraídas
    de estadísticos.
    """

    def __init__(self, c_in, d_model,  freq='h', dropout=0.1, window=24, lags=[3, 5, 7],
                 max_len=5000, embed_type="timeF"):
        super(DataEmbedding_TPE, self).__init__()

        self.est_features = RollingFeatureExtractor(window, c_in)
        self.lags = lags
        self.value_embedding_combined = TokenEmbedding(
            c_in=c_in * (5 + len(lags)), d_model=d_model
        )

        self.fixed_pe = FixedEmbedding(max_len, d_model)
        self.learned_pe = LearnablePositionalEncoding(d_model, max_len)
        self.tpe = TemporalPositionalEncoding(d_model, max_len)

        self.weight_params = nn.Parameter(torch.tensor(
            [0.25, 0.25, 0.25, 0.25], dtype=torch.float32
        ))

        self.norm_combined = nn.LayerNorm(d_model)
        self.norm_fixed = nn.LayerNorm(d_model)
        self.norm_learned = nn.LayerNorm(d_model)
        self.norm_tpe = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(p=dropout * 0.25)
        self.cont = 0

    def compute_lags(self, x):
        B, L, C = x.size()
        max_lag = max(self.lags)
        x_padded = F.pad(x, (0, 0, max_lag, 0), mode='replicate')
        lag_diffs = [x - x_padded[:, max_lag - lag:max_lag - lag + L, :]
                     for lag in self.lags]
        return torch.cat(lag_diffs, dim=-1)

    def forward(self, x, x_mark=None):
        self.cont += 1
        x_stats = self.est_features(x)
        x_lags = self.compute_lags(x)
        x_combined = torch.cat([x_stats, x_lags], dim=-1)
        combined_emb = self.value_embedding_combined(x_combined)

        B, L, _ = x.size()
        device = x.device
        position_ids = torch.arange(
            L, device=device).unsqueeze(0).expand(B, -1)

        combined_emb = self.norm_combined(combined_emb)
        pe_fixed = self.norm_fixed(self.fixed_pe(position_ids))
        pe_learned = self.norm_learned(self.learned_pe(x))
        pe_tpe = self.norm_tpe(self.tpe(combined_emb))

        weights = F.softmax(self.weight_params, dim=0)

        out = (
            weights[0] * combined_emb +
            weights[1] * pe_fixed +
            weights[2] * pe_learned +
            weights[3] * pe_tpe
        )

        if self.cont % 100 == 0:
            self.print_weights()

        return self.dropout(out)

    def print_weights(self, epoch=None):
        """
            Imprime la evolución de los pesos entrenados para cada componente del embedding:
            - Combinado de estadísticas y lags
            - Posicional fijo (sinusoidal)
            - Posicional aprendible
            - t-PE (temporal positional encoding)

            Args:
                epoch: Época actual de entrenamiento, si está disponible. Defaults to None.
        """
        weights = F.softmax(self.weight_params, dim=0)
        msg = f"\t[Epoch {epoch}] " if epoch else ""
        print(
            f"\t{msg}Weights -> Combined: {weights[0]:.3f}, "
            f"Fixed: {weights[1]:.3f}, "
            f"Learned: {weights[2]:.3f}, "
            f"TPE: {weights[3]:.3f}")
