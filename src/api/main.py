import joblib

from fastapi import FastAPI, HTTPException
import pandas as pd

from src.api.pydantic_models import CustomerData


app = FastAPI(title="Credit Risk API")


# =========================
# LOAD MODEL
# =========================
model = joblib.load("model/model.pkl")


# =========================
# HEALTH CHECK
# =========================
@app.get("/")
def home():
    return {
        "message": "Credit Risk API Running"
    }


# =========================
# PREDICT
# =========================
@app.post("/predict")
def predict(data: CustomerData):

    # validation
    if any([
        data.transaction_count < 0,
        data.total_transaction_amount < 0,
        data.avg_transaction_amount < 0,
        data.std_transaction_amount < 0
    ]):
        raise HTTPException(
            status_code=400,
            detail="Input values cannot be negative"
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
        probability = float(model.predict(input_df)[0])
        prediction = int(probability > 0.30)

        return {
            "risk_probability": probability,
            "is_high_risk": prediction
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {e}"
        )