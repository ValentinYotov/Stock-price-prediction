import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from torch.utils.data import DataLoader

from src.data.pipeline import get_datasets
from src.evaluation.metrics import calculate_metrics
from src.evaluation.visualizations import plot_training_curves
from src.models.transformer_model import StockTransformer
from src.training.trainer import Trainer
from src.utils.config import load_config


def main():
    config = load_config()
    
    print("Зареждане на данни...")
    train_dataset, val_dataset, test_dataset, feature_columns = get_datasets(config)
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Feature dimensions: {len(feature_columns)}")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_dataset,
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
    
    trainer = Trainer(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
    )
    
    print("Започване на обучение...")
    history = trainer.train()
    
    results_dir = Path(config.paths.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    plot_training_curves(
        history["train_losses"],
        history["val_losses"],
        save_path=results_dir / "training_curves.png",
    )
    
    print(f"Обучението завърши. Best val loss: {history['best_val_loss']:.6f}")
    print(f"Моделът е записан в: {config.paths.models_dir}/best_model.pt")


if __name__ == "__main__":
    main()
