import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.masking import TriangularCausalMask, ProbMask
from models.encoder import Encoder, EncoderLayer, EncoderStack
from models.decoder import Decoder, DecoderLayer
from models.attn import FullAttention, ProbAttention, AttentionLayer
from models.embed import DataEmbeddingNoPE, DataEmbedding_Informer, DataEmbedding_Stats, DataEmbedding_StatsLags, DataEmbedding_ALLPE_Weighted, DataEmbedding_TPE


class Informer(nn.Module):
    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, out_len,
                 factor=5, d_model=512, n_heads=8, e_layers=3, d_layers=2, d_ff=512,
                 dropout=0.0, attn='prob', embed='timeF', freq='h', activation='gelu',
                 output_attention=False, distil=True, mix=True,
                 device=torch.device('cuda:0'), window=24, time_encoding=""):
        super(Informer, self).__init__()
        self.pred_len = out_len
        self.attn = attn
        self.output_attention = output_attention
        print(time_encoding)
        self.embed_type = time_encoding

        # Encoding
        # Embedding selector
        self.enc_embedding = self.select_embedding(
            pe_type=self.embed_type,
            c_in=enc_in,
            d_model=d_model,
            freq=freq,
            dropout=dropout,
            window=window
        )
        self.dec_embedding = self.select_embedding(
            pe_type=self.embed_type,
            c_in=dec_in,
            d_model=d_model,
            freq=freq,
            dropout=dropout,
            window=window
        )
        # Attention
        Attn = ProbAttention if attn == 'prob' else FullAttention
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention),
                                   d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(Attn(True, factor, attention_dropout=dropout, output_attention=False),
                                   d_model, n_heads, mix=mix),
                    AttentionLayer(FullAttention(False, factor, attention_dropout=dropout, output_attention=False),
                                   d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        # self.end_conv1 = nn.Conv1d(in_channels=label_len+out_len, out_channels=out_len, kernel_size=1, bias=True)
        # self.end_conv2 = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=1, bias=True)
        self.projection = nn.Linear(d_model, c_out, bias=True)

    def select_embedding(self, pe_type, c_in, d_model, freq, dropout, **kwargs):
        print("Selected: ", pe_type)

        if pe_type == "no_pe":
            return DataEmbeddingNoPE(c_in=c_in, d_model=d_model, freq=freq, dropout=dropout, **kwargs)
        elif pe_type == "informer":
            return DataEmbedding_Informer(c_in=c_in, d_model=d_model, freq=freq, dropout=dropout, **kwargs)
        elif pe_type == "stats":
            return DataEmbedding_Stats(c_in=c_in, d_model=d_model, freq=freq, dropout=dropout, **kwargs)
        elif pe_type == "stats_lags":
            return DataEmbedding_StatsLags(c_in=c_in, d_model=d_model, freq=freq, dropout=dropout, **kwargs)
        elif pe_type == "all_pe_weighted":
            return DataEmbedding_ALLPE_Weighted(c_in=c_in, d_model=d_model, freq=freq, dropout=dropout, **kwargs)
        elif pe_type == "tpe":
            return DataEmbedding_TPE(c_in=c_in, d_model=d_model, freq=freq, dropout=dropout, **kwargs)
        else:
            raise ValueError(f"Tipo de embedding desconocido: {pe_type}")

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(
            dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        dec_out = self.projection(dec_out)

        # dec_out = self.end_conv1(dec_out)
        # dec_out = self.end_conv2(dec_out.transpose(2,1)).transpose(1,2)
        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]


class InformerStack(nn.Module):
    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, out_len,
                 factor=5, d_model=512, n_heads=8, e_layers=[3, 2, 1], d_layers=2, d_ff=512,
                 dropout=0.0, attn='prob', embed='fixed', freq='h', activation='gelu',
                 output_attention=False, distil=True, mix=True,
                 device=torch.device('cuda:0'), window=24):
        super(InformerStack, self).__init__()
        self.pred_len = out_len
        self.attn = attn
        self.output_attention = output_attention
        self.embed_type = embed

        # Encoding
        # Embedding selector
        self.enc_embedding = self.select_embedding(
            pe_type=self.embed_type,
            c_in=enc_in,
            d_model=d_model,
            freq=freq,
            dropout=dropout,
            window=window
        )
        self.dec_embedding = self.select_embedding(
            pe_type=self.embed_type,
            c_in=dec_in,
            d_model=d_model,
            freq=freq,
            dropout=dropout,
            window=window
        )
        # Attention
        Attn = ProbAttention if attn == 'prob' else FullAttention
        # Encoder

        # [0,1,2,...] you can customize here
        inp_lens = list(range(len(e_layers)))
        encoders = [
            Encoder(
                [
                    EncoderLayer(
                        AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention),
                                       d_model, n_heads, mix=False),
                        d_model,
                        d_ff,
                        dropout=dropout,
                        activation=activation
                    ) for l in range(el)
                ],
                [
                    ConvLayer(
                        d_model
                    ) for l in range(el-1)
                ] if distil else None,
                norm_layer=torch.nn.LayerNorm(d_model)
            ) for el in e_layers]
        self.encoder = EncoderStack(encoders, inp_lens)
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(Attn(True, factor, attention_dropout=dropout, output_attention=False),
                                   d_model, n_heads, mix=mix),
                    AttentionLayer(FullAttention(False, factor, attention_dropout=dropout, output_attention=False),
                                   d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        # self.end_conv1 = nn.Conv1d(in_channels=label_len+out_len, out_channels=out_len, kernel_size=1, bias=True)
        # self.end_conv2 = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=1, bias=True)
        self.projection = nn.Linear(d_model, c_out, bias=True)

    def select_embedding(self, pe_type, c_in, d_model, freq, dropout, **kwargs):
        print("Selected: ", pe_type)
        if pe_type == "no_pe":
            return DataEmbeddingNoPE(c_in=c_in, d_model=d_model, freq=freq, dropout=dropout, **kwargs)
        elif pe_type == "informer":
            return DataEmbedding_Informer(c_in=c_in, d_model=d_model, freq=freq, dropout=dropout, **kwargs)
        elif pe_type == "stats":
            return DataEmbedding_Stats(c_in=c_in, d_model=d_model, freq=freq, dropout=dropout, **kwargs)
        elif pe_type == "stats_lags":
            return DataEmbedding_StatsLags(c_in=c_in, d_model=d_model, freq=freq, dropout=dropout, **kwargs)
        elif pe_type == "all_pe_weighted":
            return DataEmbedding_ALLPE_Weighted(c_in=c_in, d_model=d_model, freq=freq, dropout=dropout, **kwargs)
        elif pe_type == "tpe":
            return DataEmbedding_TPE(c_in=c_in, d_model=d_model, freq=freq, dropout=dropout, **kwargs)
        else:
            raise ValueError(f"Tipo de embedding desconocido: {pe_type}")

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(
            dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        dec_out = self.projection(dec_out)

        # dec_out = self.end_conv1(dec_out)
        # dec_out = self.end_conv2(dec_out.transpose(2,1)).transpose(1,2)
        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
