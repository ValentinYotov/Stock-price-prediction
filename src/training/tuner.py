"""
Hyperparameter tuning for the stock-forecasting Transformer.

Two methods are supported, both of which were recommended in the thesis review:

    * Grid search    -- exhaustive cartesian product of `choices` for every
                        hyperparameter. Transparent and deterministic, but the
                        cost grows multiplicatively with the search space.
    * Bayesian (TPE) -- Optuna's Tree-structured Parzen Estimator. Samples
                        configurations using the posterior over past trials,
                        typically reaching the optimum 5-10x faster than grid.

Both methods share one `objective(params)` function that trains a fresh model
with the sampled params (using `fast_overrides` to cap the per-trial budget)
and returns the best validation loss seen during training.

Every trial is logged to a parquet file; the best params are also written to a
ready-to-load YAML so retraining at full strength is a one-liner.
"""
from __future__ import annotations

import copy
import itertools
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from src.data.pipeline import get_datasets
from src.models.transformer_model import StockTransformer
from src.training.trainer import Trainer
from src.utils.config import Config, PROJECT_ROOT, SearchSpaceEntry


# ---------------------------------------------------------------------------
# Apply sampled params into a copied Config
# ---------------------------------------------------------------------------
# Map "flat" tuning parameter names to the nested attribute path on Config.
PARAM_PATHS: Dict[str, Tuple[str, ...]] = {
    "learning_rate": ("training", "learning_rate"),
    "batch_size": ("training", "batch_size"),
    "weight_decay": ("training", "optimizer_params", "weight_decay"),
    "d_model": ("model", "d_model"),
    "n_heads": ("model", "n_heads"),
    "n_layers": ("model", "n_layers"),
    "d_ff": ("model", "d_ff"),
    "dropout": ("model", "dropout"),
    "context_length": ("data", "context_length"),
}


def _set_nested(obj: Any, path: Tuple[str, ...], value: Any) -> None:
    for attr in path[:-1]:
        obj = getattr(obj, attr)
    setattr(obj, path[-1], value)


def _apply_params(config: Config, params: Dict[str, Any]) -> Config:
    """Return a deep copy of `config` with `params` and fast_overrides applied."""
    cfg = copy.deepcopy(config)
    for name, value in params.items():
        if name not in PARAM_PATHS:
            raise KeyError(
                f"Unknown tuning parameter '{name}'. "
                f"Add it to PARAM_PATHS in src/training/tuner.py."
            )
        # Cast ints stored as floats by Optuna's suggest_int back to int.
        if name in {"batch_size", "d_model", "n_heads", "n_layers", "d_ff", "context_length"}:
            value = int(value)
        _set_nested(cfg, PARAM_PATHS[name], value)

    overrides = cfg.tuning.fast_overrides or {}
    if "num_epochs" in overrides:
        cfg.training.num_epochs = int(overrides["num_epochs"])
    if "early_stopping_patience" in overrides:
        cfg.training.early_stopping.patience = int(overrides["early_stopping_patience"])
    if "batch_size" in overrides:
        cfg.training.batch_size = int(overrides["batch_size"])
    return cfg


def _enforce_attention_constraint(params: Dict[str, Any]) -> bool:
    """d_model must be divisible by n_heads (multi-head attention requirement)."""
    if "d_model" in params and "n_heads" in params:
        return int(params["d_model"]) % int(params["n_heads"]) == 0
    return True


# ---------------------------------------------------------------------------
# Per-trial objective: train once with the sampled params, return best val loss
# ---------------------------------------------------------------------------
def _train_once(cfg: Config, trial_checkpoint: str) -> float:
    """Train a fresh model with `cfg` and return the best validation loss."""
    cfg.paths.checkpoint_file = trial_checkpoint

    train_ds, val_ds, _, feature_columns = get_datasets(cfg)
    train_loader = DataLoader(
        train_ds, batch_size=cfg.training.batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.training.batch_size, shuffle=False, num_workers=0
    )

    model = StockTransformer(
        input_dim=len(feature_columns),
        d_model=cfg.model.d_model,
        n_heads=cfg.model.n_heads,
        n_layers=cfg.model.n_layers,
        d_ff=cfg.model.d_ff,
        dropout=cfg.model.dropout,
        activation=cfg.model.activation,
        prediction_horizon=cfg.data.prediction_horizon,
    )
    trainer = Trainer(model=model, config=cfg, train_loader=train_loader, val_loader=val_loader)
    history = trainer.train()
    return float(history["best_val_loss"])


# ---------------------------------------------------------------------------
# Grid search
# ---------------------------------------------------------------------------
def _grid_combinations(space: Dict[str, SearchSpaceEntry]) -> List[Dict[str, Any]]:
    """Cartesian product over `choices` of every search-space entry."""
    names: List[str] = []
    grids: List[List[Any]] = []
    for name, entry in space.items():
        if not entry.choices:
            raise ValueError(
                f"Grid search requires 'choices' for every parameter; "
                f"'{name}' has none. Either add `choices: [...]` or switch to bayesian."
            )
        names.append(name)
        grids.append(list(entry.choices))
    return [dict(zip(names, combo)) for combo in itertools.product(*grids)]


def run_grid_search(
    config: Config,
    output_dir: Path,
    progress_callback: Optional[Callable[[int, int, Dict[str, Any], float], None]] = None,
) -> Dict[str, Any]:
    space = config.tuning.search_space
    if not space:
        raise ValueError("config.tuning.search_space is empty.")

    combos = [c for c in _grid_combinations(space) if _enforce_attention_constraint(c)]
    total = len(combos)
    if total == 0:
        raise ValueError("No valid hyperparameter combinations after constraints.")

    print(f"Grid search: {total} valid combinations", flush=True)
    trials: List[Dict[str, Any]] = []
    best: Optional[Dict[str, Any]] = None

    for i, params in enumerate(combos, 1):
        t0 = time.time()
        ckpt = f"tune_grid_trial_{i:03d}.pt"
        try:
            val_loss = _train_once(_apply_params(config, params), trial_checkpoint=ckpt)
            status = "ok"
        except Exception as exc:
            val_loss = float("inf")
            status = f"failed: {exc}"

        elapsed = time.time() - t0
        record = {"trial": i, "status": status, "val_loss": val_loss,
                  "time_sec": round(elapsed, 1), **params}
        trials.append(record)

        if status == "ok" and (best is None or val_loss < best["val_loss"]):
            best = record

        if progress_callback:
            progress_callback(i, total, params, val_loss)
        print(f"  [{i:>3}/{total}] {params}  val_loss={val_loss:.6f}  ({elapsed:.0f}s)", flush=True)

    return _finalize(trials, best, "grid", output_dir, config)


# ---------------------------------------------------------------------------
# Bayesian search (Optuna TPE)
# ---------------------------------------------------------------------------
def _suggest(trial, name: str, entry: SearchSpaceEntry) -> Any:
    """Sample one parameter from `entry` using Optuna's API."""
    if entry.type == "categorical":
        return trial.suggest_categorical(name, entry.choices)
    if entry.type == "uniform":
        if entry.low is None or entry.high is None:
            raise ValueError(f"'{name}': uniform needs low and high")
        return trial.suggest_float(name, entry.low, entry.high)
    if entry.type == "loguniform":
        if entry.low is None or entry.high is None:
            raise ValueError(f"'{name}': loguniform needs low and high")
        return trial.suggest_float(name, entry.low, entry.high, log=True)
    if entry.type == "int":
        if entry.low is None or entry.high is None:
            raise ValueError(f"'{name}': int needs low and high")
        return trial.suggest_int(name, int(entry.low), int(entry.high))
    raise ValueError(f"'{name}': unknown search-space type '{entry.type}'")


def run_bayesian_search(
    config: Config,
    output_dir: Path,
    progress_callback: Optional[Callable[[int, int, Dict[str, Any], float], None]] = None,
) -> Dict[str, Any]:
    import optuna

    space = config.tuning.search_space
    if not space:
        raise ValueError("config.tuning.search_space is empty.")
    n_trials = config.tuning.n_trials

    print(f"Bayesian (TPE) search: {n_trials} trials", flush=True)
    trials: List[Dict[str, Any]] = []
    best: Dict[str, Any] = {"val_loss": float("inf")}

    def objective(trial: "optuna.trial.Trial") -> float:
        params = {name: _suggest(trial, name, entry) for name, entry in space.items()}
        if not _enforce_attention_constraint(params):
            raise optuna.exceptions.TrialPruned("d_model not divisible by n_heads")

        t0 = time.time()
        ckpt = f"tune_bayes_trial_{trial.number + 1:03d}.pt"
        val_loss = _train_once(_apply_params(config, params), trial_checkpoint=ckpt)
        elapsed = time.time() - t0

        record = {"trial": trial.number + 1, "status": "ok", "val_loss": val_loss,
                  "time_sec": round(elapsed, 1), **params}
        trials.append(record)
        if val_loss < best["val_loss"]:
            best.update(record)

        if progress_callback:
            progress_callback(trial.number + 1, n_trials, params, val_loss)
        print(f"  [trial {trial.number+1:>3}/{n_trials}] {params}  val_loss={val_loss:.6f}  ({elapsed:.0f}s)",
              flush=True)
        return val_loss

    sampler = optuna.samplers.TPESampler(seed=config.tuning.seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, catch=(Exception,))

    if best["val_loss"] == float("inf"):
        best = None  # type: ignore[assignment]
    return _finalize(trials, best, "bayesian", output_dir, config)


# ---------------------------------------------------------------------------
# Persist results
# ---------------------------------------------------------------------------
def _finalize(
    trials: List[Dict[str, Any]],
    best: Optional[Dict[str, Any]],
    method: str,
    output_dir: Path,
    config: Config,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    trials_df = pd.DataFrame(trials)
    trials_df.to_parquet(output_dir / "trials.parquet", index=False)
    trials_df.to_csv(output_dir / "trials.csv", index=False)

    summary = {
        "method": method,
        "n_trials": len(trials),
        "successful_trials": int((trials_df["status"] == "ok").sum()) if not trials_df.empty else 0,
        "best": best,
        "total_time_sec": float(trials_df["time_sec"].sum()) if not trials_df.empty else 0.0,
        "search_space": {k: asdict(v) for k, v in config.tuning.search_space.items()},
        "fast_overrides": dict(config.tuning.fast_overrides),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

    best_params: Dict[str, Any] = {}
    if best is not None:
        best_params = {k: v for k, v in best.items()
                       if k not in {"trial", "status", "val_loss", "time_sec"}}
        (output_dir / "best_params.yaml").write_text(
            yaml.safe_dump({"best_val_loss": best["val_loss"],
                            "method": method,
                            "params": best_params}, sort_keys=False),
            encoding="utf-8",
        )

    print(f"\nResults written to: {output_dir}", flush=True)
    if best is not None:
        print(f"Best val_loss = {best['val_loss']:.6f}", flush=True)
        print(f"Best params   = {best_params}", flush=True)
    return summary


def run_tuning(config: Config) -> Dict[str, Any]:
    """Dispatch entry point: select method from config and run the search."""
    if not config.tuning.enabled:
        raise RuntimeError("config.tuning.enabled is false; nothing to do.")

    torch.manual_seed(config.tuning.seed)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = PROJECT_ROOT / config.paths.results_dir / "tuning" / f"{config.tuning.method}_{timestamp}"

    method = config.tuning.method.lower()
    if method == "grid":
        return run_grid_search(config, output_dir)
    if method == "bayesian":
        return run_bayesian_search(config, output_dir)
    raise ValueError(f"Unknown tuning.method '{method}'. Use 'grid' or 'bayesian'.")


__all__ = [
    "run_tuning",
    "run_grid_search",
    "run_bayesian_search",
    "PARAM_PATHS",
]
