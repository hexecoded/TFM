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
    def __init__(self, d_model):
        super(FourierEncoding, self).__init__()
        self.d_model = d_model

    def forward(self, x):
        # x: [B, L, d_model]
        # FFT sobre la dimensión temporal (L)
        x_fft = torch.fft.rfft(x, dim=1, norm='ortho')  # [B, L//2+1, d_model]
        # Tomamos el módulo (magnitude)
        x_freq = x_fft.abs()  # [B, L//2+1, d_model]
        # Reescalamos para que tenga longitud L
        x_freq = F.interpolate(x_freq.permute(0, 2, 1), size=x.size(
            1), mode='linear', align_corners=False)
        return x_freq.permute(0, 2, 1)  # [B, L, d_model]


class SlidingWindowMean(nn.Module):
    """
    Clase que calcula una media deslizante a través de una
    ventana especificable mediante parámetros de entrada

    Args:
        nn (_type_): valores en formato tensor
    """

    def __init__(self, window_size=5):
        super(SlidingWindowMean, self).__init__()
        self.window_size = window_size
        self.padding = window_size // 2

    def forward(self, x):
        x = F.pad(x, (self.padding, self.padding), mode='replicate')
        return F.avg_pool1d(x, kernel_size=self.window_size, stride=1)


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
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super(TemporalEmbedding, self).__init__()

        minute_size = 4
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13

        Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding
        if freq == 't':
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)


class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super(TimeFeatureEmbedding, self).__init__()

        freq_map = {'h': 4, 't': 5, 's': 6,
                    'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model)

    def forward(self, x):
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
    def __init__(self, d_model, max_len=5000):
        super(LearnablePositionalEncoding, self).__init__()
        self.pe = nn.Parameter(torch.randn(1, max_len, d_model))

    def forward(self, x):
        return self.pe[:, :x.size(1)]


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



class TemporalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, sigma=1.0):
        super(TemporalPositionalEncoding, self).__init__()
        self.d_model = d_model
        self.sigma = sigma

        # Componente geométrico (sinusoidal)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

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
#
#
# class DataEmbedding(nn.Module):
#     def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1, window=24, lags=[3, 5, 7],
#                  max_len=5000):
#         super(DataEmbedding, self).__init__()
#
#         self.est_features = RollingFeatureExtractor(window, c_in)
#         self.lags = lags
#         self.value_embedding_combined = TokenEmbedding(
#             c_in=c_in * (5 + len(lags)), d_model=d_model
#         )
#
#         self.fixed_pe = FixedEmbedding(max_len, d_model)
#         self.learned_pe = LearnablePositionalEncoding(d_model, max_len)
#         self.tape = tAPE(d_model, max_len)
#         self.tpe = TemporalPositionalEncoding(d_model, max_len)
#
#         self.weight_params = nn.Parameter(torch.tensor(
#             [0.2, 0.2, 0.2, 0.2, 0.2], dtype=torch.float32
#         ))
#
#         self.norm_combined = nn.LayerNorm(d_model)
#         self.norm_fixed = nn.LayerNorm(d_model)
#         self.norm_learned = nn.LayerNorm(d_model)
#         self.norm_tape = nn.LayerNorm(d_model)
#         self.norm_tpe = nn.LayerNorm(d_model)
#
#         self.dropout = nn.Dropout(p=dropout * 0.25)
#         self.cont = 0
#
#     def compute_lags(self, x):
#         B, L, C = x.size()
#         max_lag = max(self.lags)
#         x_padded = F.pad(x, (0, 0, max_lag, 0), mode='replicate')
#         lag_diffs = [x - x_padded[:, max_lag - lag:max_lag - lag + L, :] for lag in self.lags]
#         return torch.cat(lag_diffs, dim=-1)
#
#     def forward(self, x, x_mark=None):
#         self.cont += 1
#         x_stats = self.est_features(x)
#         x_lags = self.compute_lags(x)
#         x_combined = torch.cat([x_stats, x_lags], dim=-1)
#         combined_emb = self.value_embedding_combined(x_combined)
#
#         B, L, _ = x.size()
#         device = x.device
#         position_ids = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)
#
#         combined_emb = self.norm_combined(combined_emb)
#         pe_fixed = self.norm_fixed(self.fixed_pe(position_ids))
#         pe_learned = self.norm_learned(self.learned_pe(x))
#         pe_tape = self.norm_tape(self.tape(combined_emb, position_ids))
#         pe_tpe = self.norm_tpe(self.tpe(combined_emb))
#
#         weights = F.softmax(self.weight_params, dim=0)
#
#         out = (
#                 weights[0] * combined_emb +
#                 weights[1] * pe_fixed +
#                 weights[2] * pe_learned +
#                 weights[2] * pe_tape +
#                 weights[3] * pe_tpe
#         )
#
#         if self.cont % 100 == 0:
#             self.print_weights()
#
#         return self.dropout(out)
#
#     def print_weights(self, epoch=None):
#         weights = F.softmax(self.weight_params, dim=0)
#         msg = f"\t[Epoch {epoch}] " if epoch else ""
#         print(
#             f"\t{msg}Weights -> Combined: {weights[0]:.3f}, "
#             f"Fixed: {weights[1]:.3f}, "
#             f"Learned: {weights[2]:.3f}, "
#             f"tAPE: {weights[3]:.3f}, "
#             f"TPE: {weights[4]:.3f}")

class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1, window=24, lags=[3, 5, 7],
                 max_len=5000):
        super(DataEmbedding, self).__init__()

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
        lag_diffs = [x - x_padded[:, max_lag - lag:max_lag - lag + L, :] for lag in self.lags]
        return torch.cat(lag_diffs, dim=-1)

    def forward(self, x, x_mark=None):
        self.cont += 1
        x_stats = self.est_features(x)
        x_lags = self.compute_lags(x)
        x_combined = torch.cat([x_stats, x_lags], dim=-1)
        combined_emb = self.value_embedding_combined(x_combined)

        B, L, _ = x.size()
        device = x.device
        position_ids = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)

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
        weights = F.softmax(self.weight_params, dim=0)
        msg = f"\t[Epoch {epoch}] " if epoch else ""
        print(
            f"\t{msg}Weights -> Combined: {weights[0]:.3f}, "
            f"Fixed: {weights[1]:.3f}, "
            f"Learned: {weights[2]:.3f}, "
            f"TPE: {weights[3]:.3f}")
