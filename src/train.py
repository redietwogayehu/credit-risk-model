import pandas as pd
import mlflow
import mlflow.sklearn
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from src.data_processing import (
    load_data,
    create_customer_dataset
)

from src.target_engineering import create_rfm_target


# =========================================================
# 1. LOAD DATA
# =========================================================
df = load_data("data/raw/data.csv")


# =========================================================
# 2. FEATURES (CUSTOMER LEVEL)
# =========================================================
features = create_customer_dataset(df)


# =========================================================
# 3. TARGET (RFM PROXY)
# =========================================================
target = create_rfm_target(df)


# =========================================================
# 4. MERGE DATASET
# =========================================================
data = features.merge(target, on="CustomerId")


# =========================================================
# 5. SPLIT DATA
# =========================================================
X = data.drop(columns=["CustomerId", "is_high_risk"])
y = data["is_high_risk"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# =========================================================
# 6. MODELS
# =========================================================
models = {
    "logistic_regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    ),
    "random_forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
}


# =========================================================
# 7. MLFLOW SETUP
# =========================================================
mlflow.set_experiment("credit_risk_model")


# =========================================================
# 8. TRAINING LOOP
# =========================================================

results = {}

for name, model in models.items():

    with mlflow.start_run(run_name=name):

        # Train model
        model.fit(X_train, y_train)

        # Predictions
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob > 0.3).astype(int)

        # Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)

        # Log parameters
        mlflow.log_param("model", name)

        # Log metrics
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1", f1)
        mlflow.log_metric("roc_auc", auc)

        # Log model artifact
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model"
        )

        # Save results for comparison
        results[name] = {
            "model": model,
            "roc_auc": auc,
            "f1": f1
        }

        print(
            f"{name} -> "
            f"AUC: {auc:.4f}, "
            f"F1: {f1:.4f}"
        )


# =========================================================
# 9. SELECT BEST MODEL
# =========================================================

best_model_name = max(
    results,
    key=lambda x: results[x]["roc_auc"]
)

best_model = results[best_model_name]["model"]

print(f"\nBest Model: {best_model_name}")
print(
    f"Best ROC-AUC: "
    f"{results[best_model_name]['roc_auc']:.4f}"
)


# =========================================================
# 10. REGISTER BEST MODEL
# =========================================================

with mlflow.start_run(run_name="best_model_registration"):

    mlflow.log_param(
        "best_model",
        best_model_name
    )

    mlflow.sklearn.log_model(
        sk_model=best_model,
        artifact_path="best_model"
    )

print("Best model saved to MLflow.")
joblib.dump(best_model, "model.pkl")

print("Best model saved locally.")