from __future__ import annotations

import argparse
import json
import subprocess
from tempfile import mkdtemp
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split

from src.evaluate import regression_metrics, save_metrics
from src.features import prepare_training_frame
from src.modeling import build_preprocessor, build_stacking_regressor

try:
    import mlflow
except ImportError:  # pragma: no cover
    mlflow = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the notebook-aligned stacking model.")
    parser.add_argument("--train-data", required=True, help="Path to the training CSV.")
    parser.add_argument("--model-output", required=True, help="Directory where the trained artifact will be stored.")
    parser.add_argument("--metrics-output", default="outputs/metrics.json", help="Path to write evaluation metrics.")
    parser.add_argument("--predictions-output", default="outputs/validation_predictions.csv", help="Path to write validation predictions.")
    parser.add_argument("--feature-percentile", type=int, default=25, help="Feature selection percentile from the notebook stacking path.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Validation split fraction.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--stacking-n-jobs", type=int, default=1, help="Parallelism for the stacking regressor.")
    parser.add_argument("--experiment-name", default="house-price-stacking")
    return parser.parse_args()


def get_git_sha() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip()


def maybe_configure_mlflow(experiment_name: str) -> None:
    if mlflow is not None:
        mlflow.set_experiment(experiment_name)


def main() -> None:
    args = parse_args()
    train_df = pd.read_csv(args.train_data)
    features, target = prepare_training_frame(train_df)

    X_train, X_valid, y_train, y_valid = train_test_split(
        features,
        target,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    y_train_log = np.log1p(y_train)
    cachedir = mkdtemp()
    preprocessor = build_preprocessor(features, feature_percentile=args.feature_percentile)
    stacking_model = build_stacking_regressor(n_jobs=args.stacking_n_jobs)
    pipeline = make_pipeline(preprocessor, stacking_model, memory=cachedir)

    maybe_configure_mlflow(args.experiment_name)
    active_run = mlflow.start_run() if mlflow is not None else None

    try:
        if mlflow is not None:
            mlflow.log_params(
                {
                    "feature_percentile": args.feature_percentile,
                    "test_size": args.test_size,
                    "random_state": args.random_state,
                    "stacking_n_jobs": args.stacking_n_jobs,
                    "stacking_xgboost_max_depth": 10,
                    "stacking_xgboost_n_estimators": 300,
                    "stacking_xgboost_learning_rate": 0.1,
                    "train_rows": len(X_train),
                    "valid_rows": len(X_valid),
                    "git_sha": get_git_sha(),
                }
            )

        pipeline.fit(X_train, y_train_log)
        valid_pred_log = pipeline.predict(X_valid)
        valid_pred = np.expm1(valid_pred_log)
        metrics = regression_metrics(y_valid.to_numpy(), valid_pred)

        if mlflow is not None:
            mlflow.log_metrics(metrics)

        model_output_dir = Path(args.model_output)
        model_output_dir.mkdir(parents=True, exist_ok=True)
        artifact_path = model_output_dir / "house_price_stacking.joblib"
        artifact = {
            "model": pipeline,
            "target_transform": "log1p",
            "metadata": {
                "model_family": "stacking",
                "feature_percentile": args.feature_percentile,
            },
        }
        joblib.dump(artifact, artifact_path)

        predictions_path = Path(args.predictions_output)
        predictions_path.parent.mkdir(parents=True, exist_ok=True)
        validation_frame = X_valid.copy()
        validation_frame["ActualSalePrice"] = y_valid
        validation_frame["PredictedSalePrice"] = valid_pred
        validation_frame.to_csv(predictions_path, index=False)

        save_metrics(metrics, args.metrics_output)

        run_summary = {
            "metrics": metrics,
            "metrics_path": str(Path(args.metrics_output)),
            "model_path": str(artifact_path),
            "predictions_path": str(predictions_path),
        }
        print(json.dumps(run_summary, indent=2, sort_keys=True))

        # if mlflow is not None:
        #     mlflow.log_artifact(str(Path(args.metrics_output)))
        #     mlflow.log_artifact(str(predictions_path))
        #     mlflow.log_artifact(str(artifact_path))
    finally:
        if active_run is not None:
            mlflow.end_run()


if __name__ == "__main__":
    main()
