from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.training.callbacks import EarlyStopping, ModelCheckpoint
from src.training.losses import get_loss_function
from src.utils.config import Config


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
        
        checkpoint_path = Path(config.paths.models_dir) / "best_model.pt"
        self.checkpoint = ModelCheckpoint(checkpoint_path, monitor="val_loss", mode="min")
        
        self.train_losses = []
        self.val_losses = []
    
    def train_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_x, batch_y in self.train_loader:
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
        
        for epoch in range(self.config.training.num_epochs):
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
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{self.config.training.num_epochs} - "
                      f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        return {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "best_val_loss": best_val_loss,
        }


__all__ = [
    "Trainer",
]
