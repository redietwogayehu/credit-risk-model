from fastapi import FastAPI
import pandas as pd
import joblib

from src.api.pydantic_models import CustomerData


app = FastAPI(
    title="Credit Risk API"
)


model = joblib.load("model.pkl")


@app.get("/")
def home():

    return {
        "message": "Credit Risk API Running"
    }


@app.post("/predict")
def predict(data: CustomerData):

    input_df = pd.DataFrame(
        [{
            "total_transaction_amount":
                data.total_transaction_amount,

            "avg_transaction_amount":
                data.avg_transaction_amount,

            "transaction_count":
                data.transaction_count,

            "std_transaction_amount":
                data.std_transaction_amount
        }]
    )

    probability = model.predict_proba(
        input_df
    )[0][1]

    prediction = int(
        probability > 0.30
    )

    return {
        "risk_probability":
            float(probability),

        "is_high_risk":
            prediction
    }