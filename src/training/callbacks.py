from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch


class EarlyStopping:
    def __init__(self, patience: int = 10, min_delta: float = 0.0, mode: str = "min"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
        elif self._is_better(score, self.best_score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop
    
    def _is_better(self, current: float, best: float) -> bool:
        if self.mode == "min":
            return current < best - self.min_delta
        else:
            return current > best + self.min_delta


class ModelCheckpoint:
    def __init__(self, filepath: Path, monitor: str = "val_loss", mode: str = "min"):
        self.filepath = Path(filepath)
        self.monitor = monitor
        self.mode = mode
        self.best_score = None
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
    
    def __call__(self, model: torch.nn.Module, score: float, epoch: int):
        if self.best_score is None or self._is_better(score, self.best_score):
            self.best_score = score
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'score': score,
            }, self.filepath)
            return True
        return False
    
    def _is_better(self, current: float, best: float) -> bool:
        if self.mode == "min":
            return current < best
        else:
            return current > best


__all__ = [
    "EarlyStopping",
    "ModelCheckpoint",
]
