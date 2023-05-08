import torch
import torch.nn as nn
import numpy as np
from typing import Union, Tuple



import torch.nn as nn
from typing import Tuple


class AutoEncoder(nn.Module):
    def __init__(self, 
                 window_size: int=60,
                 num_transformer_layers: int=3,
                 nhead: int=10,
                 dim: int=60,
                 encoding_dims: int=5):
        super().__init__()
        self.window_size = window_size
        self.dim = dim
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=nhead, batch_first=True)
        decoder_layer = nn.TransformerDecoderLayer(d_model=dim, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=num_transformer_layers)
        self.linear_encoder = nn.Sequential(
            nn.Flatten(1, -1),
            nn.Linear(dim * window_size, dim * window_size // 4),
            nn.Linear(dim * window_size // 4, dim * window_size // 4 ** 2),
            nn.Linear(dim * window_size // 4 ** 2, dim * window_size // 4 ** 3),
            nn.Linear(dim * window_size // 4 ** 3, encoding_dims)
        )
        self.linear_decoder = nn.Sequential(
            nn.Linear(encoding_dims, dim * window_size // 4 ** 3),
            nn.Linear(dim * window_size // 4 ** 3, dim * window_size // 4 ** 2),
            nn.Linear(dim * window_size // 4 ** 2, dim * window_size // 4),
            nn.Linear(dim * window_size // 4, dim * window_size),
            nn.Unflatten(-1, (self.window_size, self.dim)),
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer=decoder_layer, num_layers=num_transformer_layers)
        

    def encode(self, x: torch.tensor) -> Tuple[torch.tensor]:
        x_ = self.transformer_encoder(x)
        # x_ = torch.flatten(x_, 1, 2)
        z = self.linear_encoder(x_)
        return x_, z
    
    def decode(self, z: torch.tensor, memory: torch.tensor) -> torch.tensor:
        y_ = self.linear_decoder(z)
        y = self.transformer_decoder(y_, memory=memory)
        return y

    def forward(self) -> torch.tensor:
        x_, z = self.encode(X)
        y_ = self.decode(z, x_)
        return y_
    
    def __call__(self, x) -> torch.tensor:
        return self.forward(x)
