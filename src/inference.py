from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.features import prepare_inference_frame

BASE_DIR = Path(__file__).resolve().parent.parent
NEW_MODEL_PATH = BASE_DIR / "model" / "house_price_stacking.joblib"
LEGACY_MODEL_PATH = BASE_DIR / "model" / "pipeline_stacking.pkl"


def resolve_model_path() -> Path:
    configured_path = os.getenv("MODEL_PATH")
    if configured_path:
        return Path(configured_path)
    if NEW_MODEL_PATH.exists():
        return NEW_MODEL_PATH
    return LEGACY_MODEL_PATH


def load_model(model_path: str | Path | None = None) -> Any:
    path = Path(model_path) if model_path is not None else resolve_model_path()
    return joblib.load(path)


def _predict_bundle(bundle: dict[str, Any], df: pd.DataFrame) -> pd.DataFrame:
    ids, features = prepare_inference_frame(df)
    preds_log = bundle["model"].predict(features)

    if bundle.get("target_transform") == "log1p":
        preds = np.expm1(preds_log)
    else:
        preds = np.exp(preds_log)

    return pd.DataFrame(
        {
            "Id": ids if ids is not None else pd.Series(range(len(preds))),
            "SalePrice": np.round(preds, 2),
        }
    )


def _predict_legacy_model(model: Any, df: pd.DataFrame) -> pd.DataFrame:
    legacy = df.copy()
    ids = legacy["Id"]
    features = legacy.drop(columns=["Id"])
    features["sin_MoSold"] = np.sin(2 * np.pi * (features["MoSold"] - 1) / 12)
    features["cos_MoSold"] = np.cos(2 * np.pi * (features["MoSold"] - 1) / 12)
    preds_log = model.predict(features)
    preds = np.exp(preds_log)
    return pd.DataFrame({"Id": ids, "SalePrice": np.round(preds, 2)})


def predict_dataframe(model: Any, df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(model, dict) and "model" in model:
        return _predict_bundle(model, df)
    return _predict_legacy_model(model, df)
