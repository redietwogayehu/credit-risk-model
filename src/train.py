import pandas as pd
import mlflow
import mlflow.sklearn
import joblib

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score

from src.data_processing import load_data, create_customer_dataset
from src.target_engineering import create_rfm_target
from src.feature_selection import calculate_iv


# LOAD DATA
df = load_data("data/raw/data.csv")

features = create_customer_dataset(df)
target = create_rfm_target(df)

data = features.merge(target, on="CustomerId")


# WOE / IV FEATURE SELECTION
candidate_features = [
    "transaction_count",
    "total_transaction_amount",
    "avg_transaction_amount",
    "std_transaction_amount"
]

selected_features = []
iv_scores = {}

print("\nInformation Value (IV) Analysis")

for col in candidate_features:

    iv = calculate_iv(
        data,
        col,
        "is_high_risk"
    )

    iv_scores[col] = iv

    print(f"{col}: {iv:.4f}")

    if iv > 0.1:
        selected_features.append(col)

print("\nSelected Features:")
print(selected_features)


# FEATURES / TARGET
X = data[selected_features]
y = data["is_high_risk"]


# TRAIN TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# MODELS
models = {
    "logistic_regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    ),

    "random_forest": RandomForestClassifier(
        random_state=42
    )
}


# HYPERPARAMETERS
param_grid = {
    "random_forest": {
        "n_estimators": [100, 200, 300],
        "max_depth": [5, 10, 20, None],
        "min_samples_split": [2, 5, 10]
    }
}


# MLFLOW
mlflow.set_experiment("credit_risk_model")

results = {}


# TRAINING LOOP
for name, model in models.items():

    print(f"\nTraining {name}")

    with mlflow.start_run(run_name=name):

        if name in param_grid:

            model = RandomizedSearchCV(
                model,
                param_grid[name],
                n_iter=5,
                scoring="roc_auc",
                cv=3,
                random_state=42,
                n_jobs=-1
            )

        model.fit(X_train, y_train)

        if hasattr(model, "best_estimator_"):
            model = model.best_estimator_

        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob > 0.3).astype(int)

        auc = roc_auc_score(y_test, y_prob)
        f1 = f1_score(y_test, y_pred)

        mlflow.log_param("model", name)

        mlflow.log_param(
            "selected_features",
            ",".join(selected_features)
        )

        mlflow.log_metric("roc_auc", auc)
        mlflow.log_metric("f1", f1)

        for feature, iv in iv_scores.items():
            mlflow.log_metric(
                f"iv_{feature}",
                iv
            )

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name="credit_risk_model"
        )

        results[name] = {
            "model": model,
            "roc_auc": auc
        }

        print(
            f"{name} | "
            f"AUC={auc:.4f} | "
            f"F1={f1:.4f}"
        )


# BEST MODEL
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

joblib.dump(
    best_model,
    "model/model.pkl"
)

print("Model saved successfully")