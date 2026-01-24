import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from torch.utils.data import DataLoader

from src.data.pipeline import get_datasets
from src.evaluation.metrics import calculate_metrics
from src.evaluation.visualizations import (
    plot_predictions_vs_actual,
    plot_residuals,
    plot_scatter_predictions,
)
from src.models.transformer_model import StockTransformer
from src.utils.config import load_config


def main():
    config = load_config()
    
    print("Зареждане на данни...")
    train_dataset, val_dataset, test_dataset, feature_columns = get_datasets(config)
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
    )
    
    model = StockTransformer(
        input_dim=len(feature_columns),
        d_model=config.model.d_model,
        n_heads=config.model.n_heads,
        n_layers=config.model.n_layers,
        d_ff=config.model.d_ff,
        dropout=config.model.dropout,
        activation=config.model.activation,
        prediction_horizon=config.data.prediction_horizon,
    )
    
    checkpoint_path = Path(config.paths.models_dir) / "best_model.pt"
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=config.training.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Зареден модел от епоха {checkpoint['epoch']} с val loss: {checkpoint['score']:.6f}")
    else:
        print("Не е намерен checkpoint. Използва се ненаучен модел.")
    
    device = torch.device(config.training.device if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            predictions = model(batch_x)
            
            if predictions.dim() == 1:
                predictions = predictions.unsqueeze(1)
            if batch_y.dim() == 1:
                batch_y = batch_y.unsqueeze(1)
            
            all_predictions.append(predictions.cpu())
            all_targets.append(batch_y.cpu())
    
    predictions = torch.cat(all_predictions, dim=0)
    targets = torch.cat(all_targets, dim=0)
    
    metrics = calculate_metrics(predictions, targets, config.evaluation.metrics)
    
    print("\nРезултати на тест set:")
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.6f}")
    
    results_dir = Path(config.paths.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    plot_predictions_vs_actual(
        predictions,
        targets,
        save_path=results_dir / "predictions_vs_actual.png",
        title="Test Set: Predictions vs Actual",
    )
    
    plot_residuals(
        predictions,
        targets,
        save_path=results_dir / "residuals.png",
    )
    
    plot_scatter_predictions(
        predictions,
        targets,
        save_path=results_dir / "scatter_predictions.png",
    )
    
    print(f"\nГрафиките са записани в: {results_dir}")


if __name__ == "__main__":
    main()
