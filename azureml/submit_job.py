from __future__ import annotations

import argparse
import os
from pathlib import Path

from azure.ai.ml import Input, MLClient, command
from azure.identity import DefaultAzureCredential


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Submit the stacking training job to Azure ML.")
    parser.add_argument("--subscription-id", default=os.getenv("AZURE_SUBSCRIPTION_ID"))
    parser.add_argument("--resource-group", default=os.getenv("AZURE_RESOURCE_GROUP"))
    parser.add_argument("--workspace-name", default=os.getenv("AZUREML_WORKSPACE_NAME"))
    parser.add_argument("--compute", default=os.getenv("AZUREML_COMPUTE", "cpu-cluster"))
    parser.add_argument("--train-data", default=os.getenv("AZUREML_TRAIN_DATA", "azureml:house-price-train-data:1"))
    parser.add_argument("--environment", default=os.getenv("AZUREML_ENVIRONMENT", "azureml:house-price-train-env:1"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    missing = [
        name
        for name, value in {
            "subscription_id": args.subscription_id,
            "resource_group": args.resource_group,
            "workspace_name": args.workspace_name,
        }.items()
        if not value
    ]
    if missing:
        raise ValueError(f"Missing Azure ML settings: {', '.join(missing)}")

    ml_client = MLClient(
        credential=DefaultAzureCredential(),
        subscription_id=args.subscription_id,
        resource_group_name=args.resource_group,
        workspace_name=args.workspace_name,
    )

    job = command(
        code=str(Path(__file__).resolve().parent.parent),
        command=(
            "python -m src.train "
            "--train-data ${{inputs.train_data}} "
            "--model-output ${{outputs.model_output}} "
            "--metrics-output ${{outputs.metrics_output}}/metrics.json "
            "--predictions-output ${{outputs.metrics_output}}/validation_predictions.csv"
        ),
        inputs={"train_data": Input(type="uri_file", path=args.train_data)},
        outputs={
            "model_output": {"type": "uri_folder"},
            "metrics_output": {"type": "uri_folder"},
        },
        environment=args.environment,
        compute=args.compute,
        experiment_name="house-price-stacking",
        display_name="train-house-price-stacking",
    )

    created_job = ml_client.jobs.create_or_update(job)
    print(f"Submitted Azure ML job: {created_job.name}")
    print(f"Studio URL: {created_job.studio_url}")


if __name__ == "__main__":
    main()
