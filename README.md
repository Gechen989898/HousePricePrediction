## 🏠 House Price Prediction — End-to-End Machine Learning Pipeline

### 📌 Overview

This project implements a **production-style machine learning pipeline** to predict residential house prices using the Ames Housing dataset. The focus is not only on model accuracy, but on building a workflow that reflects **real-world Data Science practice**, including:

* leakage-safe preprocessing pipelines
* baseline → model iteration → stacking ensembles
* cross-validated evaluation
* explainability using SHAP
* reproducibility and model export

This project is part of my portfolio and demonstrates my ability to work through the **full ML lifecycle — from raw data → feature engineering → modeling → evaluation → interpretation → deployment-readiness**.

---

## 🎯 Objectives

* Build a **clean, modular, and reproducible ML pipeline**
* Establish a **baseline benchmark** and iteratively improve performance
* Apply structured **preprocessing, encoding, scaling, and feature selection**
* Compare multiple regression models and **stacking ensembles**
* Evaluate using **cross-validated RMSE / MAE** to avoid overfitting
* Apply **model explainability (SHAP)** to support interpretation and trust

---

## 🧰 Tech Stack

* Python, NumPy, Pandas
* Scikit-Learn (Pipeline, ColumnTransformer, CV, Stacking)
* Ridge Regression, KNN Regressor, SVR
* Random Forest, Gradient Boosting, XGBoost
* SHAP for global & local interpretability
* Matplotlib / Seaborn for visualization

---

## 🗂 Project Structure

```
house-price-prediction/
│
├── data/                     # dataset files
├── notebooks/                # analysis & experimentation
│   └── house_price_prediction.ipynb
├── models/                   # exported pipelines / artifacts
├── README.md
└── requirements.txt
```

The notebook is designed to be **restart-and-run-all reproducible**.

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
* Web-based prediction interface

---

## 💡 What This Project Demonstrates

This project highlights my ability to:

* Design **industry-style ML workflows**, not just experiments
* Apply **structured feature engineering & validation**
* Work with real-world tabular datasets
* Build models that are **reproducible, interpretable, and scalable**
* Communicate modeling decisions in a **business-aligned** way
