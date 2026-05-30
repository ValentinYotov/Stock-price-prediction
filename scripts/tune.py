"""
Run a hyperparameter search using the settings under `tuning:` in default_config.yaml.

Two methods supported (recommendation #1 from the thesis review):
    --method grid       exhaustive cartesian product over `choices`
    --method bayesian   Optuna TPE sampler (default)

Examples:
    py -3.11 scripts/tune.py                       # uses config.tuning.method
    py -3.11 scripts/tune.py --method bayesian --n-trials 30
    py -3.11 scripts/tune.py --method grid

Results land in results/tuning/<method>_<timestamp>/:
    trials.parquet / trials.csv  -- per-trial params, val_loss, runtime
    summary.json                 -- method, best trial, search space
    best_params.yaml             -- ready-to-merge into default_config.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from src.training.tuner import run_tuning
from src.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Hyperparameter tuning runner")
    parser.add_argument("--method", choices=["grid", "bayesian"], default=None,
                        help="Override config.tuning.method")
    parser.add_argument("--n-trials", type=int, default=None,
                        help="Override config.tuning.n_trials (bayesian only)")
    parser.add_argument("--config", default=None, help="Optional path to a custom YAML config")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.method:
        config.tuning.method = args.method
    if args.n_trials is not None:
        config.tuning.n_trials = args.n_trials
    config.tuning.enabled = True

    print(f"Method        : {config.tuning.method}")
    print(f"N trials      : {config.tuning.n_trials} (bayesian only)")
    print(f"Search space  : {list(config.tuning.search_space.keys())}")
    print(f"Fast overrides: {config.tuning.fast_overrides}\n")

    run_tuning(config)


if __name__ == "__main__":
    main()
