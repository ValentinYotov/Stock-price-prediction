from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_predictions_vs_actual(
    predictions: np.ndarray | torch.Tensor,
    actual: np.ndarray | torch.Tensor,
    save_path: Path | None = None,
    title: str = "Predictions vs Actual",
) -> None:
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(actual, torch.Tensor):
        actual = actual.cpu().numpy()
    
    predictions = predictions.flatten()
    actual = actual.flatten()
    
    plt.figure(figsize=(12, 6))
    plt.plot(actual, label="Actual", alpha=0.7, linewidth=1.5)
    plt.plot(predictions, label="Predictions", alpha=0.7, linewidth=1.5)
    plt.xlabel("Time Step")
    plt.ylabel("Price")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.close()


def plot_residuals(
    predictions: np.ndarray | torch.Tensor,
    actual: np.ndarray | torch.Tensor,
    save_path: Path | None = None,
) -> None:
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(actual, torch.Tensor):
        actual = actual.cpu().numpy()
    
    residuals = actual.flatten() - predictions.flatten()
    
    plt.figure(figsize=(12, 6))
    plt.plot(residuals, alpha=0.7)
    plt.axhline(y=0, color='r', linestyle='--', linewidth=1)
    plt.xlabel("Time Step")
    plt.ylabel("Residual")
    plt.title("Residuals Plot")
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.close()


def plot_training_curves(
    train_losses: list[float],
    val_losses: list[float],
    save_path: Path | None = None,
) -> None:
    plt.figure(figsize=(12, 6))
    plt.plot(train_losses, label="Train Loss", alpha=0.7)
    plt.plot(val_losses, label="Validation Loss", alpha=0.7)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Curves")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.close()


def plot_scatter_predictions(
    predictions: np.ndarray | torch.Tensor,
    actual: np.ndarray | torch.Tensor,
    save_path: Path | None = None,
) -> None:
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(actual, torch.Tensor):
        actual = actual.cpu().numpy()
    
    predictions = predictions.flatten()
    actual = actual.flatten()
    
    plt.figure(figsize=(8, 8))
    plt.scatter(actual, predictions, alpha=0.5)
    
    min_val = min(np.min(actual), np.min(predictions))
    max_val = max(np.max(actual), np.max(predictions))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label="Perfect Prediction")
    
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Predictions vs Actual (Scatter)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.close()


__all__ = [
    "plot_predictions_vs_actual",
    "plot_residuals",
    "plot_training_curves",
    "plot_scatter_predictions",
]
