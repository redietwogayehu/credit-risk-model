import joblib

from fastapi import FastAPI
import pandas as pd
import mlflow.pyfunc

from src.api.pydantic_models import CustomerData


# =========================
# APP
# =========================
app = FastAPI(title="Credit Risk API")


# =========================
# LOAD MODEL FROM MLFLOW REGISTRY
# =========================
MODEL_NAME = "credit_risk_model"

model = joblib.load("model/model.pkl")  


# =========================
# HEALTH CHECK
# =========================
@app.get("/")
def home():
    return {
        "message": "Credit Risk API Running (MLflow Registry Model)"
    }


# =========================
# PREDICT ENDPOINT
# =========================
@app.post("/predict")
def predict(data: CustomerData):

    if data.transaction_count < 0:
        raise HTTPException(
            status_code=400,
            detail="transaction_count cannot be negative"
        )

    if data.total_transaction_amount < 0:
        raise HTTPException(
            status_code=400,
            detail="total_transaction_amount cannot be negative"
        )

    if data.avg_transaction_amount < 0:
        raise HTTPException(
            status_code=400,
            detail="avg_transaction_amount cannot be negative"
        )

    if data.std_transaction_amount < 0:
        raise HTTPException(
            status_code=400,
            detail="std_transaction_amount cannot be negative"
        )

    input_df = pd.DataFrame(
        [[
            data.transaction_count,
            data.total_transaction_amount,
            data.avg_transaction_amount,
            data.std_transaction_amount
        ]],
        columns=[
            "transaction_count",
            "total_transaction_amount",
            "avg_transaction_amount",
            "std_transaction_amount"
        ]
    )

    try:

        probability = float(
            model.predict(input_df)[0]
        )

        prediction = int(
            probability > 0.30
        )

        return {
            "risk_probability": probability,
            "is_high_risk": prediction
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {e}"
        )