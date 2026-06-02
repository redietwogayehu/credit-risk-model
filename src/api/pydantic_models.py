from pydantic import BaseModel


class CustomerData(BaseModel):

    total_transaction_amount: float

    avg_transaction_amount: float

    transaction_count: int

    std_transaction_amount: float