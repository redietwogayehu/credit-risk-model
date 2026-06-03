Credit Risk Probability Model for Alternative Data

by Rediet Wogayehu

📌 Overview

This project builds a credit risk scoring system for a Buy-Now-Pay-Later (BNPL) use case in collaboration with Bati Bank and an eCommerce platform.

Since no historical credit default labels exist, a proxy risk model is built using customer transactional behavior.

The system outputs a risk probability score (0–1) for each customer to support credit decisions.

🎯 Objectives
Build a behavioral credit risk model without default labels
Predict probability of customer financial risk
Support BNPL approval decisions
Enable dynamic credit limit assignment
Deploy a real-time scoring API
🏦 Business Context

Traditional credit scoring depends on repayment history. In this case:

No credit bureau data is available
Risk is inferred from transaction behavior
RFM (Recency, Frequency, Monetary) is used for segmentation
⚖️ Regulatory Requirements (Basel II)
Models must be interpretable
Feature engineering must be traceable
Decisions must be explainable to auditors and risk teams
📊 Dataset

Source: Kaggle (Xente Transaction Dataset)
Link: https://www.kaggle.com/datasets/ealaxi/paysim1

Contains:

Customer transactions
Amounts and categories
Time-based features

⚠️ Fraud labels are NOT used.

🧠 Approach
1. Proxy Target Creation (RFM + KMeans)
Recency → inactivity
Frequency → transaction behavior
Monetary → spending level
KMeans clustering (k=3)
Least active cluster → high risk label
2. Feature Engineering

Customer-level features:

total_transaction_amount
avg_transaction_amount
transaction_count
std_transaction_amount

Time features:

hour, day, month, year
3. Feature Strength (WoE / IV)
Feature	IV Score	Interpretation
transaction_count	0.63	Strong
total_transaction_amount	0.73	Strong
avg_transaction_amount	0.73	Strong

👉 Features show strong predictive power.

🤖 Model Training
Model	Purpose
Logistic Regression	Baseline (interpretable)
Random Forest (Tuned)	Best performing model
Hyperparameter tuning using RandomizedSearchCV
Final selection based on ROC-AUC
📊 Model Evaluation

Metrics:

Accuracy
Precision
Recall
F1-score
ROC-AUC (primary metric)
📌 Figure 1 — Model Performance Comparison

reports/figures/model_comparison.png

👉 Random Forest outperforms Logistic Regression (ROC-AUC ≈ 0.77)

🧪 MLflow Tracking

Tracked:

Model parameters
Metrics
Artifacts
Experiment runs
📌 Model Registry Update (IMPORTANT IMPROVEMENT)

The final model is now registered using MLflow Model Registry:

Logistic Regression → Version 1
Random Forest → Version 2 (Best Model)

👉 Random Forest is automatically promoted as the best-performing model based on ROC-AUC.

📌 Figure 2 — MLflow Experiment Tracking UI

reports/figures/mlflow_ui.png

🚀 API Deployment (FastAPI)
Endpoint

POST /predict

Input
{
  "total_transaction_amount": 10000,
  "avg_transaction_amount": 2000,
  "transaction_count": 5,
  "std_transaction_amount": 500
}
Output
{
  "risk_probability": 0.87,
  "is_high_risk": 1
}
Run API
uvicorn src.api.main:app --reload
📌 Figure 3 — FastAPI Swagger UI

reports/figures/api_swagger.png

Open:
http://127.0.0.1:8000/docs

📁 Project Structure
credit-risk-model/
│
├── src/
├── tests/
├── data/
├── notebooks/
├── reports/figures/
│   ├── model_comparison.png
│   ├── mlflow_ui.png
│   ├── api_swagger.png
│
├── model/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
📈 Key Results
~95,000 transactions processed
Strong behavioral clustering patterns
Clear RFM segmentation
Random Forest best model
ROC-AUC ≈ 0.77
MLflow Model Registry successfully implemented (Versioned Models)
⚠️ Limitations
Proxy target ≠ real default risk
No credit bureau data
KMeans introduces labeling bias
Possible class imbalance effects
Model generalization risk on unseen populations
🚀 Future Improvements
Add SHAP explainability
Add real credit repayment data
Improve model calibration
Add drift monitoring
Enhance MLflow Model Registry with production staging
🔁 Reproducibility
pip install -r requirements.txt
python -m src.train
uvicorn src.api.main:app --reload
🧠 FINAL NOTE
