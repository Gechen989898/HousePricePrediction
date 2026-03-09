from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def root_mean_squared_log_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    true_values = np.asarray(y_true)
    pred_values = np.asarray(y_pred)
    log_error = np.log1p(true_values) - np.log1p(np.clip(pred_values, a_min=0, a_max=None))
    return float(np.sqrt(np.mean(np.square(log_error))))


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "rmsle": root_mean_squared_log_error(y_true, y_pred),
    }


def save_metrics(metrics: dict[str, float], output_path: str | Path) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2, sort_keys=True))
