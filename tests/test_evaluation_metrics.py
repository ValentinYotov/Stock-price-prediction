import numpy as np
import pytest
from src.evaluation import metrics as m


def test_mae_numpy():
    preds = np.array([1.0, 2.0, 3.0])
    targets = np.array([2.0, 2.0, 4.0])
    assert m.mae(preds, targets) == pytest.approx(2.0 / 3.0)


def test_rmse_numpy():
    preds = np.array([1.0, 2.0, 3.0])
    targets = np.array([2.0, 2.0, 4.0])
    assert m.rmse(preds, targets) == pytest.approx(np.sqrt(2.0 / 3.0))


def test_mape_all_zero_targets_returns_inf():
    preds = np.array([1.0, 2.0, 3.0])
    targets = np.array([0.0, 0.0, 0.0])
    assert m.mape(preds, targets) == float("inf")


def test_r2_perfect_prediction_is_one():
    targets = np.array([1.0, 2.0, 3.0, 4.0])
    preds = targets.copy()
    assert m.r2_score(preds, targets) == pytest.approx(1.0)


def test_directional_accuracy_basic():  
    targets = np.array([1.0, 2.0, 1.0, 2.0])
    preds = np.array([10.0, 11.0, 10.0, 11.0])
    assert m.directional_accuracy(preds, targets) == pytest.approx(1.0)

