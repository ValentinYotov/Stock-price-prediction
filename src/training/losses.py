from __future__ import annotations

import torch
import torch.nn as nn


class MSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.MSELoss()
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.criterion(predictions, targets)


class MAELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.L1Loss()
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.criterion(predictions, targets)


class HuberLoss(nn.Module):
    def __init__(self, delta: float = 1.0):
        super().__init__()
        self.criterion = nn.HuberLoss(delta=delta)
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.criterion(predictions, targets)


def get_loss_function(loss_name: str = "mse") -> nn.Module:
    if loss_name.lower() == "mse":
        return MSELoss()
    elif loss_name.lower() == "mae":
        return MAELoss()
    elif loss_name.lower() == "huber":
        return HuberLoss()
    else:
        return MSELoss()


__all__ = [
    "MSELoss",
    "MAELoss",
    "HuberLoss",
    "get_loss_function",
]
