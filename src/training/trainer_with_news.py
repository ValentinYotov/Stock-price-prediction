"""
Enhanced trainer that supports models with news embeddings.

The base Trainer remains unchanged for the base model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.training.trainer import Trainer
from src.training.callbacks import EarlyStopping, ModelCheckpoint
from src.training.losses import get_loss_function
from src.utils.config import Config, PROJECT_ROOT


class TrainerWithNews(Trainer):
    """
    Enhanced trainer that handles models with news embeddings.
    
    Extends the base Trainer to support datasets that return (x, news_emb, y).
    """
    
    def train_epoch(self) -> float:
        """Train one epoch with news embeddings support."""
        import sys
        import gc
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        total_batches = len(self.train_loader)
        
        try:
            for batch_idx, batch in enumerate(self.train_loader):
                try:
                    # Handle both base dataset (x, y) and news dataset (x, news_emb, y)
                    if len(batch) == 2:
                        # Base dataset format
                        batch_x, batch_y = batch
                        news_emb = None
                    elif len(batch) == 3:
                        # News dataset format
                        batch_x, news_emb, batch_y = batch
                    else:
                        raise ValueError(f"Unexpected batch format: {len(batch)} items")
                    
                    batch_x = batch_x.to(self.device)
                    batch_y = batch_y.to(self.device)
                    
                    if news_emb is not None:
                        news_emb = news_emb.to(self.device)
                    
                    self.optimizer.zero_grad()
                    
                    # Forward pass with news embeddings
                    if news_emb is not None:
                        predictions = self.model(batch_x, news_embeddings=news_emb)
                    else:
                        predictions = self.model(batch_x)
                    
                    if predictions.dim() == 1:
                        predictions = predictions.unsqueeze(1)
                    if batch_y.dim() == 1:
                        batch_y = batch_y.unsqueeze(1)
                    
                    loss = self.criterion(predictions, batch_y)
                    
                    loss.backward()
                    
                    if self.config.training.gradient_clip > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.config.training.gradient_clip,
                        )
                    
                    self.optimizer.step()
                    
                    total_loss += loss.item()
                    num_batches += 1
                    
                    # Clean up memory every 10 batches
                    if (batch_idx + 1) % 10 == 0:
                        del predictions, loss
                        gc.collect()
                        
                except Exception as e:
                    print(f"\nГРЕШКА в batch {batch_idx + 1}: {e}", flush=True)
                    import traceback
                    traceback.print_exc()
                    sys.stdout.flush()
                    raise
            
        except Exception as e:
            print(f"\nКРИТИЧНА ГРЕШКА в train_epoch: {e}", flush=True)
            import traceback
            traceback.print_exc()
            sys.stdout.flush()
            raise
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss
    
    def validate(self) -> float:
        """Validate with news embeddings support."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                # Handle both base dataset (x, y) and news dataset (x, news_emb, y)
                if len(batch) == 2:
                    batch_x, batch_y = batch
                    news_emb = None
                elif len(batch) == 3:
                    batch_x, news_emb, batch_y = batch
                else:
                    raise ValueError(f"Unexpected batch format: {len(batch)} items")
                
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                if news_emb is not None:
                    news_emb = news_emb.to(self.device)
                
                # Forward pass with news embeddings
                if news_emb is not None:
                    predictions = self.model(batch_x, news_embeddings=news_emb)
                else:
                    predictions = self.model(batch_x)
                
                if predictions.dim() == 1:
                    predictions = predictions.unsqueeze(1)
                if batch_y.dim() == 1:
                    batch_y = batch_y.unsqueeze(1)
                
                loss = self.criterion(predictions, batch_y)
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss


__all__ = [
    "TrainerWithNews",
]
