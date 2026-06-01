Credit Risk Probability Model for Alternative Data
 Project Overview

This project develops a credit risk probability model for a Buy-Now-Pay-Later (BNPL) use case in partnership between Bati Bank and an eCommerce platform.

The goal is to estimate the likelihood that a customer belongs to a high-risk segment using transactional behavioral data, since no explicit loan default label exists.

The solution is built using:

Exploratory Data Analysis (EDA)
Behavioral feature engineering (RFM-based logic)
Proxy target generation (K-Means clustering)
Machine learning models for credit scoring
 Business Context

Traditional credit scoring systems rely on historical repayment/default labels.
However, this dataset does not contain direct default outcomes.

To solve this:

A proxy target variable is engineered using customer behavior
Risk is inferred from Recency, Frequency, and Monetary (RFM) patterns
The final output is a risk probability score for BNPL decisioning

This supports:

Loan approval decisions
Credit limit assignment
Risk-based pricing
Automated customer onboarding
 Dataset Information
Source

This project uses the Xente Fraud Detection Dataset, originally released for a data science challenge.

Dataset: Xente Fraud Detection Challenge
Source: Kaggle
https://www.kaggle.com/datasets/ealaxi/paysim1 (or your actual dataset link if different — replace if needed)
Description:
Transaction-level mobile money data including customer transactions, product categories, channels, and fraud labels.
Notes
No direct credit default label exists
Fraud label is not used as a credit risk target
Data is used strictly for behavioral modeling
 Project Structure
credit-risk-model/
├── .github/workflows/ci.yml
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   └── eda.ipynb
├── src/
│   ├── __init__.py
│   ├── data_processing.py
│   ├── train.py
│   ├── predict.py
│   └── api/
│       ├── main.py
│       └── pydantic_models.py
├── tests/
├── reports/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .gitignore
└── README.md
⚙️ Setup Instructions
1. Clone Repository
git clone https://github.com/your-username/credit-risk-model.git
cd credit-risk-model
2. Create Virtual Environment
python -m venv venv
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate     # Windows
3. Install Dependencies
pip install -r requirements.txt
 How to Run EDA

Run the Jupyter notebook:

jupyter notebook notebooks/eda.ipynb

Or use modular EDA functions:

from src.data_processing import load_data, run_basic_eda

df = load_data("data/raw/data.csv")
run_basic_eda(df)
 Key Concepts
1. Proxy Target Creation

Since no default label exists:

RFM (Recency, Frequency, Monetary) analysis is used
K-Means clustering identifies customer segments
Least engaged cluster → labeled as high-risk
2. Modeling Approach

Planned models:

Logistic Regression (WoE-based, interpretable)
Decision Tree
Random Forest
Gradient Boosting
3. Evaluation Metrics
Accuracy
Precision
Recall
F1-score
ROC-AUC
 EDA Highlights
95,662 transactions, 16 features
Highly imbalanced fraud distribution (~0.2%)
Strong skewness in Amount and Value
Significant outliers in financial variables
Strong behavioral signals in ProductCategory and ChannelId
Temporal patterns observed in transaction timestamps
 Next Steps
Feature engineering pipeline (src/data_processing.py)
RFM-based proxy target generation
Model training + MLflow tracking
API deployment using FastAPI
Docker containerization
CI/CD pipeline with GitHub Actions
 Code Quality & Best Practices

This project follows:

Modular Python structure (src/)
Reusable functions for preprocessing
Separation of EDA and production code
Git best practices (.gitignore, branches)
Reproducible ML pipeline design
