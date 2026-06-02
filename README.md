Credit Risk Probability Model for Alternative Data
by rediet wogayehu

📌 Project Overview

This project builds a credit risk probability modeling system for a Buy-Now-Pay-Later (BNPL) use case in collaboration between Bati Bank and an eCommerce platform.

The objective is to estimate the probability that a customer belongs to a high-risk segment using transactional behavioral data. Since no explicit default labels exist, the system uses proxy target modeling based on customer behavior.

The system includes:

Exploratory Data Analysis (EDA)
Feature engineering using behavioral signals
RFM-based proxy target generation using clustering
Machine learning models for credit risk prediction
Model evaluation using classification metrics
MLflow experiment tracking
FastAPI deployment for inference
🏦 Business Context

Traditional credit scoring systems rely on historical repayment/default labels. In this project:

No default labels are available
Risk is inferred from transaction behavior
RFM (Recency, Frequency, Monetary) is used for behavioral segmentation
Business Objectives
Reduce credit default risk
Improve BNPL decisioning
Optimize credit limits and pricing strategies
Automate customer risk classification

📌 Final output: risk probability score per customer (proxy credit risk score)

⚠️ Regulatory & Modeling Constraints (Basel II Context)

Financial institutions must ensure:

Model interpretability is required for regulatory compliance
Feature engineering must be documented and traceable
Model behavior must be explainable to auditors

👉 Interpretability is a hard constraint, not optional.

📊 Dataset Information
Dataset: Xente Fraud Detection Dataset
Source: Kaggle
Link: https://www.kaggle.com/datasets/ealaxi/paysim1
Description
Customer transactions
Amount, value, category, channel, timestamps
Fraud labels exist but are NOT used
📁 Project Structure
credit-risk-model/
│
├── src/
├── tests/
├── data/
├── notebooks/
├── reports/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
🧠 Modeling Approach
1. Proxy Target Creation (RFM + KMeans)
Recency → inactivity
Frequency → engagement
Monetary → spending behavior
KMeans clusters customers
Least engaged cluster → high risk label

📌 Output is a risk propensity score

2. Feature Engineering
Customer-level aggregation:
Total transaction amount
Average transaction amount
Transaction count
Std deviation
Time features:
Year, Month, Day, Hour
3. Models Used
Model	Purpose
Logistic Regression	Baseline (interpretable)
Random Forest	Non-linear benchmark
4. Evaluation Metrics
Accuracy
Precision
Recall
F1-score
ROC-AUC (primary)
🧪 MLflow Tracking
Experiment comparison
Metric logging
Model artifact storage
Best model selection based on ROC-AUC
🚀 API Deployment (FastAPI)
Endpoint
POST /predict
Input Example
{
  "total_transaction_amount": 10000,
  "avg_transaction_amount": 2000,
  "transaction_count": 5,
  "std_transaction_amount": 500
}
Response Example
{
  "risk_probability": 0.87,
  "is_high_risk": 1
}
Run API
uvicorn src.api.main:app --reload

Open:

http://127.0.0.1:8000/docs
📈 Key Results
~95,000 transactions
Strong skew in financial features
Clear customer segmentation via RFM
Random Forest outperforms Logistic Regression
ROC-AUC ≈ 0.77
🧪 Engineering Practices
Modular Python structure (src/)
Unit testing with pytest
MLflow experiment tracking
Docker-ready setup
CI/CD pipeline (GitHub Actions)
🚀 Final Summary

This project demonstrates an end-to-end credit risk modeling pipeline:

Data engineering
Proxy target creation
Machine learning training
Model evaluation
MLflow trackingCredit Risk Probability Model for Alternative Data
📌 Project Overview

This project builds a credit risk probability modeling system for a Buy-Now-Pay-Later (BNPL) use case in collaboration between Bati Bank and an eCommerce platform.

The objective is to estimate the probability that a customer belongs to a high-risk segment using transactional behavioral data. Since no explicit default labels exist, the system uses proxy target modeling based on customer behavior.

The system includes:

Exploratory Data Analysis (EDA)
Feature engineering using behavioral signals
RFM-based proxy target generation using clustering
Machine learning models for credit risk prediction
Model evaluation using classification metrics
MLflow experiment tracking
FastAPI deployment for inference
🏦 Business Context

Traditional credit scoring systems rely on historical repayment/default labels. In this project:

No default labels are available
Risk is inferred from transaction behavior
RFM (Recency, Frequency, Monetary) is used for behavioral segmentation
Business Objectives
Reduce credit default risk
Improve BNPL decisioning
Optimize credit limits and pricing strategies
Automate customer risk classification

📌 Final output: risk probability score per customer (proxy credit risk score)

⚠️ Regulatory & Modeling Constraints (Basel II Context)

Financial institutions must ensure:

Model interpretability is required for regulatory compliance
Feature engineering must be documented and traceable
Model behavior must be explainable to auditors

👉 Interpretability is a hard constraint, not optional.

📊 Dataset Information
Dataset: Xente Fraud Detection Dataset
Source: Kaggle
Link: https://www.kaggle.com/datasets/ealaxi/paysim1
Description
Customer transactions
Amount, value, category, channel, timestamps
Fraud labels exist but are NOT used
📁 Project Structure
credit-risk-model/
│
├── src/
├── tests/
├── data/
├── notebooks/
├── reports/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
🧠 Modeling Approach
1. Proxy Target Creation (RFM + KMeans)
Recency → inactivity
Frequency → engagement
Monetary → spending behavior
KMeans clusters customers
Least engaged cluster → high risk label

📌 Output is a risk propensity score

2. Feature Engineering
Customer-level aggregation:
Total transaction amount
Average transaction amount
Transaction count
Std deviation
Time features:
Year, Month, Day, Hour
3. Models Used
Model	Purpose
Logistic Regression	Baseline (interpretable)
Random Forest	Non-linear benchmark
4. Evaluation Metrics
Accuracy
Precision
Recall
F1-score
ROC-AUC (primary)
🧪 MLflow Tracking
Experiment comparison
Metric logging
Model artifact storage
Best model selection based on ROC-AUC
🚀 API Deployment (FastAPI)
Endpoint
POST /predict
Input Example
{
  "total_transaction_amount": 10000,
  "avg_transaction_amount": 2000,
  "transaction_count": 5,
  "std_transaction_amount": 500
}
Response Example
{
  "risk_probability": 0.87,
  "is_high_risk": 1
}
Run API
uvicorn src.api.main:app --reload

Open:

http://127.0.0.1:8000/docs
📈 Key Results
~95,000 transactions
Strong skew in financial features
Clear customer segmentation via RFM
Random Forest outperforms Logistic Regression
ROC-AUC ≈ 0.77
🧪 Engineering Practices
Modular Python structure (src/)
Unit testing with pytest
MLflow experiment tracking
Docker-ready setup
CI/CD pipeline (GitHub Actions)
🚀 Final Summary

This project demonstrates an end-to-end credit risk modeling pipeline:

Data engineering
Proxy target creation
Machine learning training
Model evaluation
MLflow tracking
Credit Risk Probability Model for Alternative Data
📌 Project Overview

This project builds a credit risk probability modeling system for a Buy-Now-Pay-Later (BNPL) use case in collaboration between Bati Bank and an eCommerce platform.

The objective is to estimate the probability that a customer belongs to a high-risk segment using transactional behavioral data. Since no explicit default labels exist, the system uses proxy target modeling based on customer behavior.

The system includes:

Exploratory Data Analysis (EDA)
Feature engineering using behavioral signals
RFM-based proxy target generation using clustering
Machine learning models for credit risk prediction
Model evaluation using classification metrics
MLflow experiment tracking
FastAPI deployment for inference
🏦 Business Context

Traditional credit scoring systems rely on historical repayment/default labels. In this project:

No default labels are available
Risk is inferred from transaction behavior
RFM (Recency, Frequency, Monetary) is used for behavioral segmentation
Business Objectives
Reduce credit default risk
Improve BNPL decisioning
Optimize credit limits and pricing strategies
Automate customer risk classification

📌 Final output: risk probability score per customer (proxy credit risk score)

⚠️ Regulatory & Modeling Constraints (Basel II Context)

Financial institutions must ensure:

Model interpretability is required for regulatory compliance
Feature engineering must be documented and traceable
Model behavior must be explainable to auditors

👉 Interpretability is a hard constraint, not optional.

📊 Dataset Information
Dataset: Xente Fraud Detection Dataset
Source: Kaggle
Link: https://www.kaggle.com/datasets/ealaxi/paysim1
Description
Customer transactions
Amount, value, category, channel, timestamps
Fraud labels exist but are NOT used
📁 Project Structure
credit-risk-model/
│
├── src/
├── tests/
├── data/
├── notebooks/
├── reports/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
🧠 Modeling Approach
1. Proxy Target Creation (RFM + KMeans)
Recency → inactivity
Frequency → engagement
Monetary → spending behavior
KMeans clusters customers
Least engaged cluster → high risk label

📌 Output is a risk propensity score

2. Feature Engineering
Customer-level aggregation:
Total transaction amount
Average transaction amount
Transaction count
Std deviation
Time features:
Year, Month, Day, Hour
3. Models Used
Model	Purpose
Logistic Regression	Baseline (interpretable)
Random Forest	Non-linear benchmark
4. Evaluation Metrics
Accuracy
Precision
Recall
F1-score
ROC-AUC (primary)
🧪 MLflow Tracking
Experiment comparison
Metric logging
Model artifact storage
Best model selection based on ROC-AUC
🚀 API Deployment (FastAPI)
Endpoint
POST /predict
Input Example
{
  "total_transaction_amount": 10000,
  "avg_transaction_amount": 2000,
  "transaction_count": 5,
  "std_transaction_amount": 500
}
Response Example
{
  "risk_probability": 0.87,
  "is_high_risk": 1
}
Run API
uvicorn src.api.main:app --reload

Open:

http://127.0.0.1:8000/docs
📈 Key Results
~95,000 transactions
Strong skew in financial features
Clear customer segmentation via RFM
Random Forest outperforms Logistic Regression
ROC-AUC ≈ 0.77
🧪 Engineering Practices
Modular Python structure (src/)
Unit testing with pytest
MLflow experiment tracking
Docker-ready setup
CI/CD pipeline (GitHub Actions)
🚀 Final Summary

This project demonstrates an end-to-end credit risk modeling pipeline:

Data engineering
Proxy target creation
Machine learning training
Model evaluation
MLflow tracking
Credit Risk Probability Model for Alternative Data
📌 Project Overview

This project builds a credit risk probability modeling system for a Buy-Now-Pay-Later (BNPL) use case in collaboration between Bati Bank and an eCommerce platform.

The objective is to estimate the probability that a customer belongs to a high-risk segment using transactional behavioral data. Since no explicit default labels exist, the system uses proxy target modeling based on customer behavior.

The system includes:

Exploratory Data Analysis (EDA)
Feature engineering using behavioral signals
RFM-based proxy target generation using clustering
Machine learning models for credit risk prediction
Model evaluation using classification metrics
MLflow experiment tracking
FastAPI deployment for inference
🏦 Business Context

Traditional credit scoring systems rely on historical repayment/default labels. In this project:

No default labels are available
Risk is inferred from transaction behavior
RFM (Recency, Frequency, Monetary) is used for behavioral segmentation
Business Objectives
Reduce credit default risk
Improve BNPL decisioning
Optimize credit limits and pricing strategies
Automate customer risk classification

📌 Final output: risk probability score per customer (proxy credit risk score)

⚠️ Regulatory & Modeling Constraints (Basel II Context)

Financial institutions must ensure:

Model interpretability is required for regulatory compliance
Feature engineering must be documented and traceable
Model behavior must be explainable to auditors

👉 Interpretability is a hard constraint, not optional.

📊 Dataset Information
Dataset: Xente Fraud Detection Dataset
Source: Kaggle
Link: https://www.kaggle.com/datasets/ealaxi/paysim1
Description
Customer transactions
Amount, value, category, channel, timestamps
Fraud labels exist but are NOT used
📁 Project Structure
credit-risk-model/
│
├── src/
├── tests/
├── data/
├── notebooks/
├── reports/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
🧠 Modeling Approach
1. Proxy Target Creation (RFM + KMeans)
Recency → inactivity
Frequency → engagement
Monetary → spending behavior
KMeans clusters customers
Least engaged cluster → high risk label

📌 Output is a risk propensity score

2. Feature Engineering
Customer-level aggregation:
Total transaction amount
Average transaction amount
Transaction count
Std deviation
Time features:
Year, Month, Day, Hour
3. Models Used
Model	Purpose
Logistic Regression	Baseline (interpretable)
Random Forest	Non-linear benchmark
4. Evaluation Metrics
Accuracy
Precision
Recall
F1-score
ROC-AUC (primary)
🧪 MLflow Tracking
Experiment comparison
Metric logging
Model artifact storage
Best model selection based on ROC-AUC
🚀 API Deployment (FastAPI)
Endpoint
POST /predict
Input Example
{
  "total_transaction_amount": 10000,
  "avg_transaction_amount": 2000,
  "transaction_count": 5,
  "std_transaction_amount": 500
}
Response Example
{
  "risk_probability": 0.87,
  "is_high_risk": 1
}
Run API
uvicorn src.api.main:app --reload

Open:

http://127.0.0.1:8000/docs
📈 Key Results
~95,000 transactions
Strong skew in financial features
Clear customer segmentation via RFM
Random Forest outperforms Logistic Regression
ROC-AUC ≈ 0.77
🧪 Engineering Practices
Modular Python structure (src/)
Unit testing with pytest
MLflow experiment tracking
Docker-ready setup
CI/CD pipeline (GitHub Actions)
🚀 Final Summary

This project demonstrates an end-to-end credit risk modeling pipeline:

Data engineering
Proxy target creation
Machine learning training
Model evaluation
MLflow tracking
FastAPI deployment