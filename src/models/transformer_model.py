from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from src.models.components.encoder import TransformerEncoder
from src.models.components.positional_encoding import PositionalEncoding


class StockTransformer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 1024,
        dropout: float = 0.1,
        activation: str = "gelu",
        prediction_horizon: int = 1,
        max_seq_len: int = 5000,
        use_positional_encoding: bool = True,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        self.prediction_horizon = prediction_horizon
        
        self.input_projection = nn.Linear(input_dim, d_model)
        
        if use_positional_encoding:
            self.pos_encoding = PositionalEncoding(d_model, max_seq_len, dropout)
        else:
            self.pos_encoding = None
        
        self.encoder = TransformerEncoder(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            dropout=dropout,
            activation=activation,
        )
        
        self.output_projection = nn.Linear(d_model, prediction_horizon)
        
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.input_projection(x)
        
        if self.pos_encoding is not None:
            x = self.pos_encoding(x)
        
        x = self.encoder(x, mask)
        
        x = x[:, -1, :]
        
        output = self.output_projection(x)
        
        return output
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            return self.forward(x)


__all__ = [
    "StockTransformer",
]
