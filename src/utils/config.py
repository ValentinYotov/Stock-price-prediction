
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import yaml


# Път до конфигурационния файл по подразбиране
DEFAULT_CONFIG_PATH = Path("configs") / "default_config.yaml"


@dataclass
class DataFeaturesConfig:
    """Настройки, свързани с feature engineering за входните данни."""

    technical_indicators: bool = True
    lag_features: bool = True
    temporal_features: bool = True
    windows: List[int] = field(default_factory=lambda: [5, 10, 20, 50])

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DataFeaturesConfig":
        return cls(
            technical_indicators=bool(data.get("technical_indicators", True)),
            lag_features=bool(data.get("lag_features", True)),
            temporal_features=bool(data.get("temporal_features", True)),
            windows=list(data.get("windows", [5, 10, 20, 50])),
        )


@dataclass
class DataConfig:
    """Настройки за данните и dataset-а."""

    dataset_name: str = "paperswithbacktest/Stocks-Daily-Price"
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15
    context_length: int = 90
    prediction_horizon: int = 1
    features: DataFeaturesConfig = field(default_factory=DataFeaturesConfig)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DataConfig":
        features_cfg = DataFeaturesConfig.from_dict(data.get("features", {}))

        return cls(
            dataset_name=str(
                data.get("dataset_name", "paperswithbacktest/Stocks-Daily-Price")
            ),
            train_split=float(data.get("train_split", 0.7)),
            val_split=float(data.get("val_split", 0.15)),
            test_split=float(data.get("test_split", 0.15)),
            context_length=int(data.get("context_length", 90)),
            prediction_horizon=int(data.get("prediction_horizon", 1)),
            features=features_cfg,
        )


@dataclass
class ModelConfig:
    """Настройки за модела (архитектура)."""

    type: str = "transformer"  # transformer, lstm, linear
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 1024
    dropout: float = 0.1
    activation: str = "gelu"

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ModelConfig":
        return cls(
            type=str(data.get("type", "transformer")),
            d_model=int(data.get("d_model", 256)),
            n_heads=int(data.get("n_heads", 8)),
            n_layers=int(data.get("n_layers", 6)),
            d_ff=int(data.get("d_ff", 1024)),
            dropout=float(data.get("dropout", 0.1)),
            activation=str(data.get("activation", "gelu")),
        )


@dataclass
class EarlyStoppingConfig:
    """Настройки за early stopping по време на обучение."""

    patience: int = 10
    min_delta: float = 1e-4

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EarlyStoppingConfig":
        return cls(
            patience=int(data.get("patience", 10)),
            min_delta=float(data.get("min_delta", 1e-4)),
        )


@dataclass
class OptimizerParamsConfig:
    """Допълнителни параметри за оптимизатора."""

    betas: List[float] = field(default_factory=lambda: [0.9, 0.999])
    weight_decay: float = 0.0

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "OptimizerParamsConfig":
        return cls(
            betas=list(data.get("betas", [0.9, 0.999])),
            weight_decay=float(data.get("weight_decay", 0.0)),
        )


@dataclass
class TrainingConfig:
    """Настройки за тренировъчния процес."""

    batch_size: int = 32
    learning_rate: float = 1e-4
    num_epochs: int = 100
    optimizer: str = "adam"
    optimizer_params: OptimizerParamsConfig = field(
        default_factory=OptimizerParamsConfig
    )
    scheduler: str = "cosine"
    gradient_clip: float = 1.0
    early_stopping: EarlyStoppingConfig = field(
        default_factory=EarlyStoppingConfig
    )
    device: str = "cuda"

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TrainingConfig":
        optimizer_params = OptimizerParamsConfig.from_dict(
            data.get("optimizer_params", {})
        )
        early_stopping = EarlyStoppingConfig.from_dict(
            data.get("early_stopping", {})
        )

        return cls(
            batch_size=int(data.get("batch_size", 32)),
            learning_rate=float(data.get("learning_rate", 1e-4)),
            num_epochs=int(data.get("num_epochs", 100)),
            optimizer=str(data.get("optimizer", "adam")),
            optimizer_params=optimizer_params,
            scheduler=str(data.get("scheduler", "cosine")),
            gradient_clip=float(data.get("gradient_clip", 1.0)),
            early_stopping=early_stopping,
            device=str(data.get("device", "cuda")),
        )


@dataclass
class EvaluationConfig:
    """Настройки за оценка и визуализация."""

    metrics: List[str] = field(
        default_factory=lambda: ["mae", "rmse", "mape", "r2", "directional_accuracy"]
    )
    save_predictions: bool = True
    visualize: bool = True

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EvaluationConfig":
        return cls(
            metrics=list(
                data.get(
                    "metrics",
                    ["mae", "rmse", "mape", "r2", "directional_accuracy"],
                )
            ),
            save_predictions=bool(data.get("save_predictions", True)),
            visualize=bool(data.get("visualize", True)),
        )


@dataclass
class PathsConfig:
    """Пътища до основните директории в проекта."""

    data_dir: Path = Path("data")
    models_dir: Path = Path("models") / "checkpoints"
    results_dir: Path = Path("results")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PathsConfig":
        return cls(
            data_dir=Path(data.get("data_dir", "data")),
            models_dir=Path(data.get("models_dir", "models/checkpoints")),
            results_dir=Path(data.get("results_dir", "results")),
        )


@dataclass
class Config:
    """Глобален конфигурационен обект за проекта."""

    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    evaluation: EvaluationConfig
    paths: PathsConfig

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Config":
        """Създава Config от суров речник (пр. YAML)."""
        return cls(
            data=DataConfig.from_dict(data.get("data", {})),
            model=ModelConfig.from_dict(data.get("model", {})),
            training=TrainingConfig.from_dict(data.get("training", {})),
            evaluation=EvaluationConfig.from_dict(data.get("evaluation", {})),
            paths=PathsConfig.from_dict(data.get("paths", {})),
        )


def _load_yaml(path: Path) -> Dict[str, Any]:
    """Зарежда YAML файл и връща съдържанието като речник."""
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if data is None:
        # Празен YAML файл
        return {}

    if not isinstance(data, dict):
        raise ValueError(f"Config file {path} must contain a YAML mapping at top-level.")

    return data


def load_config(path: Optional[str] = None) -> Config:
    """
    Зарежда конфигурация от YAML файл и връща `Config` обект.

    :param path: Път към YAML файл. Ако е None, ползва DEFAULT_CONFIG_PATH.
    """
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    raw = _load_yaml(config_path)
    return Config.from_dict(raw)


__all__ = [
    "Config",
    "DataConfig",
    "DataFeaturesConfig",
    "ModelConfig",
    "TrainingConfig",
    "EvaluationConfig",
    "PathsConfig",
    "EarlyStoppingConfig",
    "OptimizerParamsConfig",
    "load_config",
]

