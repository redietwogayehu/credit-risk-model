# credit-risk-model
# Credit Scoring Business Understanding

Basel II and Model Interpretability

Basel II emphasizes accurate risk measurement, transparency, and documentation. Financial institutions must be able to explain how credit decisions are made and demonstrate that risk models are reliable and well governed. Therefore, interpretable models and documented feature engineering processes are important for regulatory compliance and stakeholder trust.

Need for a Proxy Target Variable

The dataset does not contain a direct default label, making supervised credit risk modeling impossible using traditional approaches. To address this limitation, a proxy target variable can be constructed using customer behavioral patterns such as Recency, Frequency, and Monetary (RFM) metrics. Customers exhibiting low engagement may be classified as higher risk. However, proxy variables introduce business risks because they do not represent actual default behavior and may incorrectly classify customers.

Model Trade-offs in a Regulated Environment

Logistic Regression models combined with Weight of Evidence (WoE) transformations provide high interpretability and are easier to explain to regulators and business stakeholders. More advanced models such as Gradient Boosting often achieve better predictive performance but are less transparent. Financial institutions must balance predictive accuracy with regulatory requirements for explainability, fairness, and model governance.