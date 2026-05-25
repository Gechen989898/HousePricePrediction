# Azure ML scaffold

This project can now submit the notebook-aligned XGBoost training script to Azure ML.

## Files

- `environment-train.yml`: Azure ML training environment
- `train-job.yml`: CLI v2 command job spec
- `submit_job.py`: Python SDK submission script

## Expected assets

- Azure ML workspace
- Compute cluster named `cpu-cluster`
- Data asset named `house-price-train-data`
- Registered environment `house-price-train-env:1`
