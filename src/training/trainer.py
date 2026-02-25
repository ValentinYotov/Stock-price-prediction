from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.training.callbacks import EarlyStopping, ModelCheckpoint
from src.training.losses import get_loss_function
from src.utils.config import Config, PROJECT_ROOT


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        config: Config,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        if device is None:
            self.device = torch.device(config.training.device if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        self.model.to(self.device)
        
        self.criterion = get_loss_function("mse")
        
        if config.training.optimizer.lower() == "adam":
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=config.training.learning_rate,
                betas=config.training.optimizer_params.betas,
                weight_decay=config.training.optimizer_params.weight_decay,
            )
        else:
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=config.training.learning_rate,
            )
        
        if config.training.scheduler.lower() == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=config.training.num_epochs,
            )
        else:
            self.scheduler = None
        
        self.early_stopping = EarlyStopping(
            patience=config.training.early_stopping.patience,
            min_delta=config.training.early_stopping.min_delta,
        )
        
        # Use absolute path from project root (same as in notebooks)
        checkpoint_name = getattr(config.paths, "checkpoint_file", "best_model.pt")
        checkpoint_path = PROJECT_ROOT / config.paths.models_dir / checkpoint_name
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        self.checkpoint = ModelCheckpoint(checkpoint_path, monitor="val_loss", mode="min")
        
        self.train_losses = []
        self.val_losses = []
    
    def train_epoch(self) -> float:
        import sys
        import gc
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        total_batches = len(self.train_loader)
        
        try:
            for batch_idx, (batch_x, batch_y) in enumerate(self.train_loader):
                try:
                    batch_x = batch_x.to(self.device)
                    batch_y = batch_y.to(self.device)
                    
                    self.optimizer.zero_grad()
                    
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
                    import traceback
                    traceback.print_exc()
                    raise
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss
    
    def validate(self) -> float:
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch_x, batch_y in self.val_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
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
    
    def train(self) -> dict:
        best_val_loss = float('inf')
        try:
            for epoch in range(self.config.training.num_epochs):
                try:
                    train_loss = self.train_epoch()
                    val_loss = self.validate()
                    self.train_losses.append(train_loss)
                    self.val_losses.append(val_loss)
                    if self.scheduler is not None:
                        self.scheduler.step()
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                    self.checkpoint(self.model, val_loss, epoch)
                    if self.early_stopping(val_loss):
                        break
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    raise
        except KeyboardInterrupt:
            pass
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise
        
        return {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "best_val_loss": best_val_loss,
        }


__all__ = [
    "Trainer",
]
