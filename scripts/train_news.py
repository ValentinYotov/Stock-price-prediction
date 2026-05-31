
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from torch.utils.data import DataLoader

from src.data.pipeline import get_datasets
from src.evaluation.visualizations import plot_training_curves
from src.models.transformer_model import StockTransformer
from src.training.trainer import Trainer
from src.utils.config import load_config


def main() -> None:
    config = load_config()

    # The only difference vs the base model:
    config.data.use_news = True
    config.paths.checkpoint_file = "best_model_news.pt"

    print("Loading data (technical + FinBERT news features)...", flush=True)
    train_dataset, val_dataset, test_dataset, feature_columns = get_datasets(config)

    print(f"Train samples: {len(train_dataset)}", flush=True)
    print(f"Val samples:   {len(val_dataset)}", flush=True)
    print(f"Test samples:  {len(test_dataset)}", flush=True)
    print(f"Feature dims:  {len(feature_columns)}  (incl. news_* features)", flush=True)
    news_cols = [c for c in feature_columns if c.startswith("news_")]
    print(f"News features: {news_cols}", flush=True)

    train_loader = DataLoader(
        train_dataset, batch_size=config.training.batch_size, shuffle=True,
        num_workers=0, pin_memory=False,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.training.batch_size, shuffle=False,
        num_workers=0, pin_memory=False,
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

    trainer = Trainer(model=model, config=config, train_loader=train_loader, val_loader=val_loader)
    history = trainer.train()

    results_dir = Path(config.paths.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    plot_training_curves(
        history["train_losses"],
        history["val_losses"],
        save_path=results_dir / "training_curves_news.png",
    )

    print(f"\nDone. Best val loss: {history['best_val_loss']:.6f}", flush=True)
    print(f"Saved: {config.paths.models_dir}/best_model_news.pt", flush=True)


if __name__ == "__main__":
    main()
