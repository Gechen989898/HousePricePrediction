## 🏠 House Price Prediction — End-to-End ML Pipeline & API

### 📌 Overview

This project demonstrates a **complete machine learning lifecycle** — from data exploration and feature engineering through model development, evaluation, and **production-ready API deployment**. The goal is to predict residential house prices using the Ames Housing dataset.

**Key Highlights:**
- 📊 **ML Pipeline**: Leakage-safe preprocessing, feature engineering, and stacking ensemble models
- 🚀 **API Server**: FastAPI-based REST API with file upload functionality
- 🎨 **Web Interface**: User-friendly UI for uploading CSV files and getting predictions
- 🐳 **Containerization**: Docker support for easy deployment
- 📈 **Model Explainability**: SHAP integration for model interpretation
- ✅ **Production Ready**: Follows ML engineering best practices

This project is part of my portfolio, demonstrating expertise in **the full ML lifecycle** and its transition to production systems.

---

## 🎯 Key Features

✨ **Comprehensive ML Pipeline**
- Baseline benchmark → Iterative model improvements → Stacking ensembles
- Cross-validated evaluation (RMSE/MAE metrics)
- Leakage prevention throughout preprocessing
- Multiple regression models: Ridge, KNN, SVR, RF, XGBoost, Gradient Boosting

📡 **Production API**
- FastAPI REST endpoints for predictions
- File upload support for batch predictions
- HTML web interface for easy interaction
- Interactive API documentation (Swagger UI)

🐳 **Deployment Ready**
- Dockerfile for containerized deployment
- Supports local development and cloud deployment
- Environment isolation with requirements.txt
- **☁️ Cloud Deployment**: Deployed to Azure App Service with Azure Container Registry (ACR)

📊 **Model Insights**
- SHAP value analysis for feature importance
- Global and local interpretability
- Visualization support (Matplotlib, Seaborn)

---

## 🧰 Tech Stack

**Backend & API**
- FastAPI 0.115.0
- Uvicorn (ASGI server)
- Python 3.12

**Data & ML**
- Pandas, NumPy
- Scikit-Learn (Pipeline, ColumnTransformer, GridSearchCV, StackingRegressor)
- XGBoost, Gradient Boosting

**Model Explainability**
- SHAP (model interpretation)

**Visualization**
- Matplotlib, Seaborn

**Deployment**
- Docker
- Python-multipart (for file uploads)
- Azure (App Service + Container Registry)

---

## 🗂 Project Structure

```
HousePricePrediction/
│
├── app/
│   └── main.py                 # FastAPI application with endpoints
├── data/
│   ├── train.csv              # Training dataset
│   ├── test.csv               # Test dataset
│   ├── prediction_baseline.csv # Baseline predictions
│   └── prediction_final.csv    # Final model predictions
├── model/
│   └── pipeline_stacking.pkl   # Trained and serialized model
├── notebooks/
│   └── house_price_prediction.ipynb  # Full ML analysis & experimentation
├── Dockerfile                  # Container configuration
├── requirements.txt            # Python dependencies
└── README.md
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.12+ or Docker

### Scripted Stacking Training

The notebook work now has a script-based training entrypoint built around the notebook's stacking path:

- `src/features.py`: shared feature engineering for training and inference
- `src/modeling.py`: notebook-aligned preprocessing plus stacking config
- `src/train.py`: local or Azure ML training entrypoint
- `src/inference.py`: shared model loading and prediction logic

Run local training:

```bash
python -m src.train \
  --train-data data/train.csv \
  --model-output outputs/model \
  --metrics-output outputs/metrics.json \
  --predictions-output outputs/validation_predictions.csv
```

Install training extras for MLflow tracking:

```bash
pip install -r requirements-train.txt
```

Install Azure ML submission extras:

```bash
pip install -r requirements-azureml.txt
```

### Option 1: Local Development

**1. Install dependencies:**
```bash
pip install -r requirements.txt
```

**2. Run the API server:**
```bash
uvicorn app.main:app --reload --port 8000
```

**3. Access the web interface:**
- Open http://localhost:8000/ in your browser
- Upload a CSV file (with the same structure as `data/test.csv`)
- View predictions in real-time

### Option 2: Docker Deployment

**1. Build the Docker image:**
```bash
docker build -t houseprice-api .
```

**2. Run the container:**
```bash
docker run -p 8000:8000 houseprice-api
```

**3. Access the application:**
- Web UI: http://localhost:8000/
- API Docs: http://localhost:8000/docs
- Health Check: http://localhost:8000/health

### Option 3: Azure Cloud Deployment

This project is **live in production** on Azure! 🎉

**Live Application:**
- 🌐 **Web UI**: https://house1244-aufmh6cpbfgabygn.canadacentral-01.azurewebsites.net/
- 📚 **API Docs**: https://house1244-aufmh6cpbfgabygn.canadacentral-01.azurewebsites.net/docs
- 🏥 **Health Check**: https://house1244-aufmh6cpbfgabygn.canadacentral-01.azurewebsites.net/health

**Architecture:**
- **Container Registry**: Azure Container Registry (ACR) stores Docker images
- **App Service**: Azure App Service runs containerized application
- **Region**: Canada Central

**Deployment Pipeline:**

1. **Build Docker image locally:**
```bash
docker build -t houseprice-api:latest .
```

2. **Tag and push to Azure Container Registry:**
```bash
# Login to ACR
az acr login --name <your-acr-name>

# Tag the image
docker tag houseprice-api:latest <your-acr-name>.azurecr.io/houseprice-api:latest

# Push to ACR
docker push <your-acr-name>.azurecr.io/houseprice-api:latest
```

3. **Deploy to App Service:**
```bash
# Create App Service Plan (if needed)
az appservice plan create --name myAppPlan --resource-group myResourceGroup --sku B1 --is-linux

# Create Web App with container
az webapp create --resource-group myResourceGroup --plan myAppPlan \
  --name house1244 \
  --deployment-container-image-name <your-acr-name>.azurecr.io/houseprice-api:latest

# Configure continuous deployment from ACR
az webapp deployment container config --name house1244 --resource-group myResourceGroup \
  --enable-cd true

# Set ACR credentials for App Service
az webapp config container set --name house1244 --resource-group myResourceGroup \
  --docker-custom-image-name <your-acr-name>.azurecr.io/houseprice-api:latest \
  --docker-registry-server-url https://<your-acr-name>.azurecr.io
```

**Key Benefits of Azure Deployment:**
- ✅ Always-on production environment
- ✅ Easy scaling and auto-restart
- ✅ Integration with ACR for container management
- ✅ Built-in monitoring and diagnostics
- ✅ Custom domain support

---

## 📡 API Endpoints

### `GET /` — Web Interface
Interactive HTML interface for uploading CSV files and viewing predictions.

### `POST /predict` — Batch Predictions
Upload a CSV file and receive predictions for all records.

**Request:**
```bash
curl -X POST -F "file=@data/test.csv" http://localhost:8000/predict
```

**Response:**
```json
[
  {"Id": 1461, "SalePrice": 179500.50},
  {"Id": 1462, "SalePrice": 156000.25},
  ...
]
```

### `GET /health` — Health Check
Simple health status endpoint for monitoring.

---

## 💡 Usage Example

### Using the Web Interface (Recommended for Non-Technical Users)
1. Navigate to http://localhost:8000/
2. Click "(Select File)"
3. Choose a CSV file (e.g., `data/test.csv`)
4. Click "(Upload & Predict)"
5. View results in the table below

### Using Python/Requests (Programmatic)
```python
import requests
import pandas as pd

# Prepare your data
with open('data/test.csv', 'rb') as f:
    files = {'file': f}
    response = requests.post('http://localhost:8000/predict', files=files)

# Get predictions
results = response.json()
df = pd.DataFrame(results)
print(df.head())
```

### Using cURL
```bash
curl -X POST -F "file=@data/test.csv" http://localhost:8000/predict | json_pp
```

---

## 📊 Model Performance

The final model uses a **stacking ensemble** combining multiple regressors:
- Base learners: Ridge Regression, KNN, SVR, Random Forest, XGBoost
- Meta-learner: Gradient Boosting

**Evaluation Metrics** (from cross-validation):
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)

*See `notebooks/house_price_prediction.ipynb` for detailed model comparison and performance analysis.*

---

## 🔧 Development

### Running Tests
To test the API locally:
```bash
# Check health endpoint
curl http://localhost:8000/health

# Upload and predict (using test data)
curl -X POST -F "file=@data/test.csv" http://localhost:8000/predict
```

### Interactive API Documentation
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

---

## 📈 Project Learnings & Best Practices

✅ **ML Engineering**
- Cross-validation for robust evaluation
- Pipeline abstraction for reproducibility
- Feature scaling and categorical encoding
- Ensemble methods for improved predictions

✅ **Software Engineering**
- Clean code structure with modular design
- FastAPI for type-safe API development
- Docker for reproducible environments
- Proper dependency management

✅ **ML Deployment**
- Model serialization with joblib
- Web interface for user interactions
- REST API for programmatic access
- Health checks and monitoring endpoints

---

## 🎓 What This Project Demonstrates

For potential employers, this project showcases:

1. **End-to-End ML Capability**: From data exploration to production deployment
2. **Engineering Best Practices**: Clean code, reproducibility, containerization
3. **Full-Stack Skills**: ML development + API design + Frontend interface
4. **Problem-Solving**: Handling class imbalance, feature selection, model evaluation
5. **Communication**: Well-documented code and clear API design

---

## 🔧 Methodology & Workflow

### 1️⃣ Exploratory Data Analysis

* Feature type inspection & missing-value profiling
* Target distribution & log-transform assessment
* Data quality checks to guide preprocessing decisions

---

### 2️⃣ Baseline Model

* Constant-value benchmark based on target distribution
* Used as the **minimum performance bar** future models must beat

---

### 3️⃣ Preprocessing & Feature Engineering

Implemented using **Scikit-Learn Pipelines** to prevent leakage:

* Numerical → KNN imputation + scaling
* Ordinal → ordered encoding
* Nominal → one-hot encoding
* Log-transform on `SalePrice`
* Feature selection using **mutual information**

All steps are applied **consistently in training & inference**.

---

### 4️⃣ Model Development

Multiple regression models were trained and compared:

* Linear / Ridge Regression
* KNN Regressor
* Support Vector Regressor (SVR)
* Tree-based ensemble models
* **Stacking Regressor (meta-ensemble)**

Models were evaluated using **cross-validated RMSE / MAE**
to ensure results reflect **true generalization**.

---

### 5️⃣ Explainability (SHAP)

To ensure transparency and business alignment:

* **Global SHAP** → identifies key price drivers
* **Beeswarm plot** → feature impact and directional effects
* **Local SHAP waterfall** → explains individual predictions

Results confirm that higher **overall quality, living area, and newer construction** generally increase predicted sale price, which aligns with housing-market economics.

---

## 📊 Results (Summary)

* **Baseline RMSE:** 0.21
* **Best Model RMSE:** 0.12
* **Performance Gain:** ~**42.9% improvement vs baseline**

Performance improvements were achieved through:

* structured preprocessing & feature selection
* model comparison and ensemble stacking
* disciplined cross-validated evaluation

The final model demonstrates **strong predictive value and practical interpretability**.

---

## 🚀 Model Export & Inference Readiness

The final pipeline (preprocessing + model) was exported as a deployable object to ensure **consistent preprocessing during inference**, making the project suitable for:

* Streamlit demo apps
* API-based prediction services
* batch scoring workflows

---

## 🔁 Reproducibility

* Pipeline-based preprocessing
* Deterministic execution
* Random seeds fixed

---

## 🔎 Limitations & Future Work

* Extended hyperparameter tuning
* Additional interaction / geospatial features
* Monitoring & drift detection
* MLflow experiment tracking
