import pandas as pd

from src.data_processing import (
    extract_datetime_features,
    create_customer_dataset
)


def test_extract_datetime_features():

    df = pd.DataFrame({
        "TransactionStartTime": [
            "2025-01-01 10:00:00"
        ]
    })

    result = extract_datetime_features(df)

    assert "transaction_year" in result.columns
    assert "transaction_month" in result.columns
    assert "transaction_day" in result.columns
    assert "transaction_hour" in result.columns


def test_create_customer_dataset():

    df = pd.DataFrame({
        "CustomerId": ["A", "A", "B"],
        "Amount": [100, 200, 300]
    })

    result = create_customer_dataset(df)

    assert len(result) == 2
    assert "total_transaction_amount" in result.columns