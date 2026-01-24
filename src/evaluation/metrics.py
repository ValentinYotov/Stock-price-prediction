from __future__ import annotations

import numpy as np
import torch


def mae(predictions: np.ndarray | torch.Tensor, targets: np.ndarray | torch.Tensor) -> float:
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()
    
    return float(np.mean(np.abs(predictions - targets)))


def rmse(predictions: np.ndarray | torch.Tensor, targets: np.ndarray | torch.Tensor) -> float:
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()
    
    return float(np.sqrt(np.mean((predictions - targets) ** 2)))


def mape(predictions: np.ndarray | torch.Tensor, targets: np.ndarray | torch.Tensor) -> float:
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()
    
    mask = targets != 0
    if not np.any(mask):
        return float('inf')
    
    return float(np.mean(np.abs((targets[mask] - predictions[mask]) / targets[mask])) * 100)


def r2_score(predictions: np.ndarray | torch.Tensor, targets: np.ndarray | torch.Tensor) -> float:
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()
    
    ss_res = np.sum((targets - predictions) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    
    if ss_tot == 0:
        return 0.0
    
    return float(1 - (ss_res / ss_tot))


def directional_accuracy(
    predictions: np.ndarray | torch.Tensor,
    targets: np.ndarray | torch.Tensor,
) -> float:
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()
    
    if len(predictions) < 2 or len(targets) < 2:
        return 0.0
    
    pred_direction = np.diff(predictions.flatten()) > 0
    target_direction = np.diff(targets.flatten()) > 0
    
    correct = np.sum(pred_direction == target_direction)
    total = len(pred_direction)
    
    return float(correct / total) if total > 0 else 0.0


def calculate_metrics(
    predictions: np.ndarray | torch.Tensor,
    targets: np.ndarray | torch.Tensor,
    metrics: list[str] | None = None,
) -> dict[str, float]:
    if metrics is None:
        metrics = ["mae", "rmse", "mape", "r2", "directional_accuracy"]
    
    results = {}
    
    metric_functions = {
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "r2": r2_score,
        "directional_accuracy": directional_accuracy,
    }
    
    for metric_name in metrics:
        if metric_name.lower() in metric_functions:
            try:
                results[metric_name] = metric_functions[metric_name.lower()](predictions, targets)
            except Exception as e:
                results[metric_name] = float('nan')
    
    return results


__all__ = [
    "mae",
    "rmse",
    "mape",
    "r2_score",
    "directional_accuracy",
    "calculate_metrics",
]
