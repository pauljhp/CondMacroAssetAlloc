"""return prediction model that takes the hidden embeddings and predicts asset 
class return or return rankings"""
import torch.nn as nn
import torch
import numpy as np
from torch.utils.data import DataLoader
from typing import Union, Tuple, Optional
from pathlib import Path
import math
from .transformerAE import AutoEncoder
import __main__


setattr(__main__, "AutoEncoder", AutoEncoder)


class ReturnPrediction(nn.Module):
    """Predict the return & stdev of the return of a price time series based on
    historical price and the macro encoding at the time"""
    def __init__(self,
                window_size: int,
                autoencoder_ckpt_path: str,
                num_transformer_layers: int,
                nhead: int=10,
                # num_linear_layers: int=5,
                dim: int=7,
                dropout: float=.1,
                ae_nhead: int=10,
                ae_dim: int=60,
                ae_num_transformer_layers: int=10,
                ae_encoding_dims: int=2,
                ):
        """
        :param window_size: must be the same as the window size in the 
            autoencoder
        :param encoding_dims: must be the same as the autoencoder passed
        :param dim: num of asset classes to predict
        :param nhead: must be factors of dim, can be different from that in the
            autoencoderdd
        """
        super().__init__()
        self.autoencoder = torch.load(autoencoder_ckpt_path)
        self.autoencoder.eval() # disable training
        self.window_size = self.autoencoder.window_size
        self.encoding_dims = self.autoencoder.encoding_dims
        self.nhead = nhead
        self.num_transformer_layers = num_transformer_layers
        encoding_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=nhead, dropout=dropout, batch_first=True)
        decoding_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=nhead, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoding_layer, num_layers=self.num_transformer_layers)
        self.transformer_decoder = nn.TransformerDecoder(decoding_layer, num_layers=self.num_transformer_layers)
        self.linear_transformer = nn.Sequential(
            nn.Flatten(1, -1),
            nn.BatchNorm1d(num_features=dim * self.window_size),
            nn.Linear(self.window_size * dim, self.encoding_dims)
        )
        self.linear = nn.Sequential(
            nn.Flatten(1, -1),
            nn.BatchNorm1d(num_features=self.encoding_dims * 2)
        )
        # cascading_factor = math.floor((self.encoding_dims * 2) ** (1 / num_linear_layers))
        linear_layer = nn.Linear(in_features=self.encoding_dims * 2, out_features=dim)
        self.linear.append(linear_layer)
            # self.register_buffer(name=f"linear_layer_{i}", tensor=linear_layer)

    def forward(self, 
                price_series: torch.tensor, 
                macro_series: torch.tensor,
                macro_padding_mask: torch.tensor,
                price_padding_mask: Optional[torch.tensor]) -> torch.tensor:
        _, macro_encoding = self.autoencoder.encode(macro_series, macro_padding_mask)
        price_ = self.transformer_encoder(price_series, src_key_padding_mask=price_padding_mask)
        # price_ = self.transformer_decoder(price_, memory)
        price_encoding = self.linear_transformer(price_)
        encoding = torch.concat((macro_encoding, price_encoding), dim=-1)
        # return encoding
        prediction = self.linear(encoding)
        return prediction
    
    def __call__(self, price_series: torch.tensor, 
                macro_series: torch.tensor,
                macro_padding_mask: torch.tensor,
                price_padding_mask: torch.tensor) -> torch.tensor:
        return self.forward(price_series, macro_series, macro_padding_mask, price_padding_mask)


# TODO - add std dev of returns in training data and retrain
# FIXME - fix n/a padding


# class ReturnRankingPrediction
# TODO - instead of predicting absolute return, predict rank of returns instead