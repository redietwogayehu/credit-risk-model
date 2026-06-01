# Credit Risk Probability Model for Alternative Data

## 📌 Project Overview

This project develops a **credit risk probability modeling system** for a Buy-Now-Pay-Later (BNPL) use case in collaboration between **Bati Bank** and an eCommerce platform.

The objective is to estimate the probability that a customer belongs to a **high-risk segment** using transactional behavioral data. The dataset does not contain explicit loan default labels, requiring **behavioral proxy modeling**.

The system includes:

- Exploratory Data Analysis (EDA)
- Feature engineering using behavioral signals
- RFM-based proxy target generation using clustering
- Machine learning models for credit risk prediction
- Model evaluation using standard metrics

---

## 🏦 Business Context

Traditional credit scoring relies on historical repayment/default labels.

In this project:

- No default labels exist
- Risk is inferred from transaction behavior
- RFM (Recency, Frequency, Monetary) is used for segmentation

### Business Objectives

- Reduce credit default risk
- Improve BNPL decisioning
- Optimize credit limits and pricing
- Automate customer risk classification

Final output: **risk probability score per customer**

---

## 📊 Dataset Information

### Source

- Dataset: Xente Fraud Detection Dataset
- Origin: Kaggle Data Science Challenge
- Link: https://www.kaggle.com/datasets/ealaxi/paysim1 *(replace if needed)*

### Description

Includes anonymized transaction data:

- Customer IDs
- Transaction amounts and values
- Product categories
- Channel information
- Fraud labels (NOT used for target)

> ⚠️ Important: This dataset does NOT contain loan default labels. Fraud labels are excluded from modeling.

---

## 📁 Project Structure


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


Credit Scoring Business Understanding (Task 1)
1. Basel II Regulatory Requirements
Model interpretability is mandatory for financial risk systems
Full documentation of features, transformations, and decisions is required
Models must be explainable to regulators and auditors
Continuous monitoring and validation is required

👉 Implication: interpretability is a regulatory constraint, not a choice

2. Proxy Target Requirement and Risk

Since no default labels exist:

RFM (Recency, Frequency, Monetary) is used for behavioral modeling
K-Means clustering segments customers
Least engaged cluster is labeled as high-risk
Risks introduced:
Proxy ≠ true default behavior
Behavioral bias in labeling
Potential mismatch with real credit risk

👉 Output should be treated as risk propensity score, not true default probability.

3. Model Strategy: Interpretability vs Performance
Interpretable Models (Preferred)
Logistic Regression (WoE-based)
Scorecards

Pros:

Fully explainable
Regulatory compliant
Stable and auditable

Cons:

Lower predictive power
High-Performance Models
Random Forest
Gradient Boosting
XGBoost

Pros:

Higher accuracy
Capture nonlinear patterns

Cons:

Low interpretability
Requires explainability tools (e.g., SHAP)
4. Final Modeling Strategy
Primary model: Logistic Regression (WoE)
Benchmark models: Random Forest, Gradient Boosting
All models must include:
Feature importance analysis
Explainability outputs
Stability checks
⚙️ Setup Instructions
1. Clone Repository
git clone https://github.com/redietwogayehu/credit-risk-model
cd credit-risk-model
2. Create Virtual Environment
python -m venv venv

Activate:

Mac/Linux

source venv/bin/activate

Windows

venv\Scripts\activate
3. Install Dependencies
pip install -r requirements.txt
📌 How to Run the Project
Option 1: Run EDA Notebook
jupyter notebook notebooks/eda.ipynb
Option 2: Run Modular Pipeline
from src.data_processing import load_data, run_basic_eda

df = load_data("data/raw/data.csv")
run_basic_eda(df)
Option 3: Train Model (future step)
python src/train.py
Option 4: Run API (future step)
uvicorn src.api.main:app --reload
🧠 Key Concepts
1. Proxy Target Creation
RFM analysis
K-Means clustering
Least engaged segment → high-risk label
2. Modeling Approach
Logistic Regression (interpretable baseline)
Decision Tree
Random Forest
Gradient Boosting
3. Evaluation Metrics
Accuracy
Precision
Recall
F1-score
ROC-AUC
📈 EDA Highlights
95,662 transaction records, 16 features
Highly imbalanced fraud distribution (~0.2%)
Strong skewness in Amount and Value
Significant outliers in financial variables
Strong behavioral signals in ProductCategory and ChannelId
Temporal patterns in TransactionStartTime
🚧 Next Steps
Feature engineering pipeline (src/data_processing.py)
RFM-based proxy target generation
Model training + MLflow tracking
FastAPI deployment
Docker containerization
CI/CD pipeline with GitHub Actions
🧪 Code Quality & Best Practices

This project follows:

Modular Python structure (src/)
Reusable preprocessing functions
Separation of EDA and production code
Git best practices (.gitignore, branching)
Reproducible ML pipeline design
Clean CI/CD-ready structure
