"""
Enhanced Transformer model with news embeddings support.

This is an improved version of StockTransformer that includes news data.
The base StockTransformer remains unchanged for comparison.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from src.models.transformer_model import StockTransformer
from src.models.components.encoder import TransformerEncoder
from src.models.components.positional_encoding import PositionalEncoding


class StockTransformerWithNews(nn.Module):
    """
    Enhanced Transformer model that combines technical features with news embeddings.
    
    Architecture:
    - Technical features → Transformer Encoder (same as base model)
    - News embeddings → Separate projection → Concatenated with encoder output
    - Combined features → Final prediction
    """
    
    def __init__(
        self,
        input_dim: int,
        news_embedding_dim: int = 768,  # FinBERT embedding dimension
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 1024,
        dropout: float = 0.1,
        activation: str = "gelu",
        prediction_horizon: int = 1,
        max_seq_len: int = 5000,
        use_positional_encoding: bool = True,
        news_fusion_method: str = "concat",  # "concat" or "add"
        news_projection_dim: Optional[int] = None,  # If None, uses d_model
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.news_embedding_dim = news_embedding_dim
        self.d_model = d_model
        self.prediction_horizon = prediction_horizon
        self.news_fusion_method = news_fusion_method
        
        # Technical features processing (same as base model)
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
        
        # News embeddings processing
        news_proj_dim = news_projection_dim if news_projection_dim is not None else d_model
        self.news_projection = nn.Linear(news_embedding_dim, news_proj_dim)
        self.news_dropout = nn.Dropout(p=dropout)
        self.news_proj_dim = news_proj_dim  # Store for use in forward
        
        # Final output projection
        if news_fusion_method == "concat":
            # Concatenate encoder output with news embedding
            output_dim = d_model + news_proj_dim
        elif news_fusion_method == "add":
            # Add news embedding to encoder output (requires same dimension)
            if news_proj_dim != d_model:
                raise ValueError("For 'add' fusion, news_projection_dim must equal d_model")
            output_dim = d_model
        else:
            raise ValueError(f"Unknown fusion method: {news_fusion_method}")
        
        self.output_projection = nn.Linear(output_dim, prediction_horizon)
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        news_embeddings: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Technical features tensor [batch_size, seq_len, input_dim]
            news_embeddings: News embeddings tensor [batch_size, news_embedding_dim]
                           If None, falls back to base model behavior
            mask: Optional attention mask
        
        Returns:
            Predictions [batch_size, prediction_horizon]
        """
        # Process technical features (same as base model)
        x = self.input_projection(x)
        
        if self.pos_encoding is not None:
            x = self.pos_encoding(x)
        
        x = self.encoder(x, mask)
        
        # Get last timestep output
        x = x[:, -1, :]  # [batch_size, d_model]
        
        # Process news embeddings if provided
        if news_embeddings is not None:
            # Project news embeddings
            news_proj = self.news_projection(news_embeddings)  # [batch_size, news_proj_dim]
            news_proj = self.news_dropout(news_proj)
            
            # Fuse with encoder output
            if self.news_fusion_method == "concat":
                combined = torch.cat([x, news_proj], dim=-1)  # [batch_size, d_model + news_proj_dim]
            elif self.news_fusion_method == "add":
                combined = x + news_proj  # [batch_size, d_model]
            else:
                raise ValueError(f"Unknown fusion method: {self.news_fusion_method}")
        else:
            # Fallback to base model (no news)
            # If output_projection expects larger input (concat fusion), we need to pad
            # Otherwise use x directly
            if self.news_fusion_method == "concat":
                # For concat fusion, output_projection expects d_model + news_proj_dim
                # Pad with zeros to match expected size
                zero_padding = torch.zeros(x.shape[0], self.news_proj_dim, device=x.device, dtype=x.dtype)
                combined = torch.cat([x, zero_padding], dim=-1)  # [batch_size, d_model + news_proj_dim]
            else:
                # For add fusion, output_projection expects d_model
                combined = x
        
        # Final prediction
        output = self.output_projection(combined)
        
        return output
    
    def predict(
        self,
        x: torch.Tensor,
        news_embeddings: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Make prediction in eval mode."""
        self.eval()
        with torch.no_grad():
            return self.forward(x, news_embeddings)
    
    def load_base_weights(self, base_model: StockTransformer) -> None:
        """
        Load weights from base StockTransformer model.
        
        This allows transfer learning from the base model.
        """
        # Copy shared weights
        self.input_projection.load_state_dict(base_model.input_projection.state_dict())
        if self.pos_encoding is not None and base_model.pos_encoding is not None:
            self.pos_encoding.load_state_dict(base_model.pos_encoding.state_dict())
        self.encoder.load_state_dict(base_model.encoder.state_dict())
        
        print("✅ Loaded base model weights into enhanced model")


__all__ = [
    "StockTransformerWithNews",
]
