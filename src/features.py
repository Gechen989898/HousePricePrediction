from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

ID_COLUMN = "Id"
TARGET_COLUMN = "SalePrice"
MONTH_COLUMN = "MoSold"
MONTHS_IN_YEAR = 12

ORDINAL_CATEGORIES = {
    "BsmtCond": ["missing", "Po", "Fa", "TA", "Gd"],
    "BsmtExposure": ["missing", "No", "Mn", "Av", "Gd"],
    "BsmtFinType1": ["missing", "Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"],
    "BsmtFinType2": ["missing", "Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"],
    "BsmtQual": ["missing", "Fa", "TA", "Gd", "Ex"],
    "Electrical": ["missing", "Mix", "FuseP", "FuseF", "FuseA", "SBrkr"],
    "ExterCond": ["missing", "Po", "Fa", "TA", "Gd", "Ex"],
    "ExterQual": ["missing", "Fa", "TA", "Gd", "Ex"],
    "Fence": ["missing", "MnWw", "GdWo", "MnPrv", "GdPrv"],
    "FireplaceQu": ["missing", "Po", "Fa", "TA", "Gd", "Ex"],
    "Functional": ["missing", "Sev", "Maj2", "Maj1", "Mod", "Min2", "Min1", "Typ"],
    "GarageCond": ["missing", "Po", "Fa", "TA", "Gd", "Ex"],
    "GarageFinish": ["missing", "Unf", "RFn", "Fin"],
    "GarageQual": ["missing", "Po", "Fa", "TA", "Gd", "Ex"],
    "HeatingQC": ["missing", "Po", "Fa", "TA", "Gd", "Ex"],
    "KitchenQual": ["missing", "Fa", "TA", "Gd", "Ex"],
    "LandContour": ["missing", "Low", "Bnk", "HLS", "Lvl"],
    "LandSlope": ["missing", "Sev", "Mod", "Gtl"],
    "LotShape": ["missing", "IR3", "IR2", "IR1", "Reg"],
    "PavedDrive": ["missing", "N", "P", "Y"],
    "PoolQC": ["missing", "Fa", "Gd", "Ex"],
}


@dataclass(frozen=True)
class DatasetBundle:
    ids: pd.Series | None
    features: pd.DataFrame
    target: pd.Series | None


def add_cyclical_month_features(df: pd.DataFrame, *, drop_original_month: bool = True) -> pd.DataFrame:
    transformed = df.copy()
    if MONTH_COLUMN in transformed.columns:
        transformed["sin_MoSold"] = np.sin(2 * np.pi * (transformed[MONTH_COLUMN] - 1) / MONTHS_IN_YEAR)
        transformed["cos_MoSold"] = np.cos(2 * np.pi * (transformed[MONTH_COLUMN] - 1) / MONTHS_IN_YEAR)
        if drop_original_month:
            transformed = transformed.drop(columns=[MONTH_COLUMN])
    return transformed


def prepare_dataset(
    df: pd.DataFrame,
    *,
    include_target: bool,
    include_ids: bool = True,
) -> DatasetBundle:
    transformed = add_cyclical_month_features(df)
    ids = transformed[ID_COLUMN].copy() if include_ids and ID_COLUMN in transformed.columns else None
    target = transformed[TARGET_COLUMN].copy() if include_target and TARGET_COLUMN in transformed.columns else None

    drop_columns: list[str] = []
    if ID_COLUMN in transformed.columns:
        drop_columns.append(ID_COLUMN)
    if TARGET_COLUMN in transformed.columns:
        drop_columns.append(TARGET_COLUMN)

    features = transformed.drop(columns=drop_columns)
    return DatasetBundle(ids=ids, features=features, target=target)


def prepare_training_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    bundle = prepare_dataset(df, include_target=True)
    if bundle.target is None:
        raise ValueError(f"Training data must contain '{TARGET_COLUMN}'.")
    return bundle.features, bundle.target


def prepare_inference_frame(df: pd.DataFrame) -> tuple[pd.Series | None, pd.DataFrame]:
    bundle = prepare_dataset(df, include_target=False)
    return bundle.ids, bundle.features


def ordinal_feature_names(columns: Iterable[str]) -> list[str]:
    column_set = set(columns)
    return sorted(name for name in ORDINAL_CATEGORIES if name in column_set)
