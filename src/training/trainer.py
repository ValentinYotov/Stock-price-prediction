from __future__ import annotations

import time
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
        
        loss_name = getattr(config.training, "loss", "mse")
        self.criterion = get_loss_function(loss_name)
        
        opt_name = config.training.optimizer.lower()
        if opt_name == "adamw":
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=config.training.learning_rate,
                betas=config.training.optimizer_params.betas,
                weight_decay=config.training.optimizer_params.weight_decay,
            )
        elif opt_name == "adam":
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
        import gc
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        total_batches = len(self.train_loader)
        log_every = max(1, total_batches // 10)

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

                    if (batch_idx + 1) % log_every == 0 and total_batches > 10:
                        running = total_loss / num_batches
                        print(
                            f"   batch {batch_idx + 1:>5}/{total_batches} "
                            f"running_train_loss={running:.6f}",
                            flush=True,
                        )

                    if (batch_idx + 1) % 10 == 0:
                        del predictions, loss
                        gc.collect()

                except Exception:
                    import traceback
                    traceback.print_exc()
                    raise
        except Exception:
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
        total_epochs = self.config.training.num_epochs
        n_train_batches = len(self.train_loader)
        n_val_batches = len(self.val_loader)

        print("=" * 78, flush=True)
        print(
            f"Trainer | device={self.device} | "
            f"loss={getattr(self.config.training, 'loss', 'mse')} | "
            f"optimizer={self.config.training.optimizer} | "
            f"lr={self.config.training.learning_rate} | "
            f"batch_size={self.config.training.batch_size}",
            flush=True,
        )
        print(
            f"Epochs: max={total_epochs}, "
            f"early_stop_patience={self.config.training.early_stopping.patience}, "
            f"train_batches/epoch={n_train_batches}, val_batches/epoch={n_val_batches}",
            flush=True,
        )
        print("=" * 78, flush=True)

        start_total = time.time()

        try:
            for epoch in range(total_epochs):
                try:
                    t0 = time.time()
                    train_loss = self.train_epoch()
                    val_loss = self.validate()
                    epoch_time = time.time() - t0

                    self.train_losses.append(train_loss)
                    self.val_losses.append(val_loss)

                    if self.scheduler is not None:
                        self.scheduler.step()

                    current_lr = self.optimizer.param_groups[0]["lr"]

                    improved = val_loss < best_val_loss
                    if improved:
                        best_val_loss = val_loss

                    self.checkpoint(self.model, val_loss, epoch)

                    marker = " *" if improved else "  "
                    stop_counter = self.early_stopping.counter
                    print(
                        f"Epoch {epoch + 1:>3}/{total_epochs} {marker} | "
                        f"train_loss={train_loss:.6f} | "
                        f"val_loss={val_loss:.6f} | "
                        f"best={best_val_loss:.6f} | "
                        f"lr={current_lr:.2e} | "
                        f"time={epoch_time:5.1f}s | "
                        f"early_stop={stop_counter}/{self.config.training.early_stopping.patience}",
                        flush=True,
                    )

                    if self.early_stopping(val_loss):
                        print(
                            f"\nEarly stopping triggered at epoch {epoch + 1} "
                            f"(no improvement for {self.config.training.early_stopping.patience} epochs).",
                            flush=True,
                        )
                        break
                except Exception:
                    import traceback
                    traceback.print_exc()
                    raise
        except KeyboardInterrupt:
            print("\nTraining interrupted by user.", flush=True)

        total_time = time.time() - start_total
        print("=" * 78, flush=True)
        print(
            f"Training done | epochs run={len(self.train_losses)} | "
            f"best val_loss={best_val_loss:.6f} | total time={total_time / 60:.1f} min",
            flush=True,
        )
        print("=" * 78, flush=True)

        return {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "best_val_loss": best_val_loss,
        }


__all__ = [
    "Trainer",
]
