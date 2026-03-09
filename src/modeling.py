from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.ensemble import AdaBoostRegressor, StackingRegressor
from sklearn.feature_selection import SelectPercentile, mutual_info_regression
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.features import ORDINAL_CATEGORIES, ordinal_feature_names


def build_preprocessor(features: pd.DataFrame, feature_percentile: int = 25) -> Pipeline:
    ordinal_features = ordinal_feature_names(features.columns)
    numeric_features = sorted(features.select_dtypes(include=["int64", "float64"]).columns)
    nominal_features = sorted(set(features.columns) - set(numeric_features) - set(ordinal_features))

    preproc_numerical = make_pipeline(
        KNNImputer(),
        MinMaxScaler(),
    )

    preproc_ordinal = make_pipeline(
        SimpleImputer(strategy="constant", fill_value="missing"),
        OrdinalEncoder(
            categories=[ORDINAL_CATEGORIES[name] for name in ordinal_features],
            handle_unknown="use_encoded_value",
            unknown_value=-1,
            dtype=np.int64,
        ),
        MinMaxScaler(),
    )

    preproc_nominal = make_pipeline(
        SimpleImputer(strategy="most_frequent"),
        OneHotEncoder(handle_unknown="ignore", drop="if_binary"),
    )

    transformer_steps: list[tuple[object, list[str]]] = []
    if numeric_features:
        transformer_steps.append((preproc_numerical, numeric_features))
    if ordinal_features:
        transformer_steps.append((preproc_ordinal, ordinal_features))
    if nominal_features:
        transformer_steps.append((preproc_nominal, nominal_features))

    preproc_transformer = make_column_transformer(
        *transformer_steps,
        remainder="drop",
    )

    preproc_selector = SelectPercentile(
        score_func=mutual_info_regression,
        percentile=feature_percentile,
    )

    return make_pipeline(preproc_transformer, preproc_selector)


def build_stacking_regressor(*, n_jobs: int = 1) -> StackingRegressor:
    xgboost = XGBRegressor(
        max_depth=10,
        n_estimators=300,
        learning_rate=0.1,
        eval_metric="rmse",
        random_state=42,
        n_jobs=1,
    )
    ridge = Ridge()
    svm = SVR(C=1, epsilon=0.05)
    adaboost = AdaBoostRegressor(estimator=DecisionTreeRegressor(max_depth=None))

    return StackingRegressor(
        estimators=[
            ("xgboost", xgboost),
            ("adaboost", adaboost),
            ("ridge", ridge),
            ("svm_rbf", svm),
        ],
        final_estimator=LinearRegression(),
        cv=5,
        n_jobs=n_jobs,
    )
