# Credit Risk Probability Model for Alternative Data

### by Rediet Wogayehu

## 📌 Overview

This project builds a credit risk scoring system for a Buy-Now-Pay-Later (BNPL) use case in collaboration with Bati Bank and an eCommerce platform.

Since no historical credit default labels exist, a proxy risk model is built using customer transactional behavior.

The system outputs a risk probability score (0–1) for each customer to support credit decisions.

---

## 🎯 Objectives

* Build a behavioral credit risk model without default labels
* Predict probability of customer financial risk
* Support BNPL approval decisions
* Enable dynamic credit limit assignment
* Deploy a real-time scoring API

---

## 🏦 Business Context

Traditional credit scoring depends on repayment history. In this case:

* No credit bureau data is available
* Risk is inferred from transaction behavior
* RFM (Recency, Frequency, Monetary) is used for segmentation

---

## ⚖️ Regulatory Requirements (Basel II)

* Models must be interpretable
* Feature engineering must be traceable
* Decisions must be explainable to auditors and risk teams

To satisfy these requirements:

* Logistic Regression was used as an interpretable baseline
* MLflow provides experiment tracking and governance
* Feature engineering decisions are documented and reproducible

---

## 📊 Dataset

**Source:** Kaggle (Xente Transaction Dataset)

Contains:

* Customer transactions
* Transaction amounts and categories
* Time-based behavioral information

⚠️ Fraud labels were not used for model training.

---

## 🧠 Approach

### 1. Proxy Target Creation (RFM + KMeans)

* Recency → inactivity level
* Frequency → customer engagement
* Monetary → spending behavior
* KMeans clustering (k = 3)
* Least active customer cluster labeled as high risk

The resulting target variable (`is_high_risk`) acts as a proxy risk indicator rather than a true default label.

---

### 2. Feature Engineering

Customer-level features:

* total_transaction_amount
* avg_transaction_amount
* transaction_count
* std_transaction_amount

Time features:

* hour
* day
* month
* year

---

### 3. Feature Strength (WoE / IV Analysis)

Information Value (IV) was used to evaluate the predictive power of engineered features.

| Feature                  | IV Score |
| ------------------------ | -------: |
| transaction_count        |   0.8740 |
| total_transaction_amount |   0.4847 |
| avg_transaction_amount   |   0.1829 |
| std_transaction_amount   |   0.4784 |

Interpretation:

* IV > 0.50 → Very Strong Predictor
* 0.30–0.50 → Strong Predictor
* 0.10–0.30 → Medium Predictor

These results demonstrate that transaction behavior provides strong discriminatory power for identifying high-risk customers.

Transaction count emerged as the strongest predictor.

---

## 🤖 Model Training

| Model               | Purpose                  |
| ------------------- | ------------------------ |
| Logistic Regression | Baseline (interpretable) |
| Random Forest       | Best performing model    |

Hyperparameter tuning was performed using RandomizedSearchCV.

Final model selection was based on ROC-AUC.

---

## 📊 Model Evaluation

Metrics evaluated:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC

### Figure 1 — Model Performance Comparison

reports/figures/model_comparison.png

Results:

| Model               | ROC-AUC | F1 Score |
| ------------------- | ------: | -------: |
| Logistic Regression |  0.7523 |   0.6031 |
| Random Forest       |  0.7940 |   0.6807 |

Random Forest achieved the strongest predictive performance and was selected as the production model.

---

## 🧪 MLflow Tracking

Tracked items:

* Model parameters
* Evaluation metrics
* Artifacts
* Experiment runs
* Model versions

### MLflow Model Registry

The final model is automatically registered in MLflow Model Registry.

Current versions:

* Logistic Regression → Version 3
* Random Forest → Version 4

This provides model governance, version control, and deployment traceability.

### Figure 2 — MLflow Experiment Tracking UI

reports/figures/mlflow_ui.png

---

## 🚀 API Deployment (FastAPI)

### Endpoint

POST /predict

### Input Example

{
"total_transaction_amount": 10000,
"avg_transaction_amount": 2000,
"transaction_count": 5,
"std_transaction_amount": 500
}

### Output Example

{
"risk_probability": 0.87,
"is_high_risk": 1
}

### Run API

uvicorn src.api.main:app --reload

### Figure 3 — FastAPI Swagger UI

reports/figures/api_swagger.png

Open:

http://127.0.0.1:8000/docs

---

## 📁 Project Structure

credit-risk-model/

├── src/

├── tests/

├── data/

├── notebooks/

├── reports/figures/

├── model/

├── Dockerfile

├── docker-compose.yml

├── requirements.txt

└── README.md

---

## 📈 Key Results

* ~95,000 transactions processed
* Strong behavioral clustering patterns identified
* Clear customer segmentation using RFM
* Information Value analysis confirms predictive feature strength
* Random Forest selected as best model
* ROC-AUC = 0.7940
* F1 Score = 0.6807
* MLflow Model Registry successfully implemented

---

## ⚠️ Limitations

* Proxy target does not represent true default behavior
* No credit bureau or repayment history available
* KMeans clustering introduces labeling assumptions
* Potential class imbalance effects
* Risk of reduced performance on unseen customer populations

---

## 🚀 Future Improvements

* Add SHAP explainability
* Integrate real repayment/default data
* Improve probability calibration
* Implement model drift monitoring
* Add automated promotion workflows in MLflow Registry

---

## 🔁 Reproducibility

pip install -r requirements.txt

python -m src.train

uvicorn src.api.main:app --reload

---

## 🧠 Final Note

This project demonstrates an end-to-end credit risk modeling workflow including behavioral feature engineering, proxy target creation, Information Value analysis, model training, hyperparameter tuning, MLflow experiment tracking, model registry integration, and FastAPI deployment.
