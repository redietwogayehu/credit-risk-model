import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


# =========================================================
# RFM TARGET ENGINEERING
# =========================================================

def create_rfm_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create proxy credit risk label using RFM + clustering.
    """

    df = df.copy()

    # -----------------------------------------------------
    # 1. Ensure datetime format
    # -----------------------------------------------------
    df["TransactionStartTime"] = pd.to_datetime(df["TransactionStartTime"])

    snapshot_date = df["TransactionStartTime"].max()

    # -----------------------------------------------------
    # 2. Build RFM table
    # -----------------------------------------------------
    rfm = df.groupby("CustomerId").agg(
        Recency=("TransactionStartTime", lambda x: (snapshot_date - x.max()).days),
        Frequency=("TransactionId", "count"),
        Monetary=("Amount", "sum")
    ).reset_index()

    # -----------------------------------------------------
    # 3. Feature scaling (VERY IMPORTANT for KMeans)
    # -----------------------------------------------------
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(
        rfm[["Recency", "Frequency", "Monetary"]]
    )

    # -----------------------------------------------------
    # 4. KMeans clustering
    # -----------------------------------------------------
    kmeans = KMeans(
        n_clusters=3,
        random_state=42,
        n_init=10
    )

    rfm["cluster"] = kmeans.fit_predict(rfm_scaled)

    # -----------------------------------------------------
    # 5. Identify worst cluster (high-risk behavior)
    # -----------------------------------------------------
    cluster_profile = rfm.groupby("cluster")[["Recency", "Frequency", "Monetary"]].mean()

    # worst =:
    # - highest recency (inactive)
    # - lowest frequency
    # - lowest monetary
    worst_cluster = cluster_profile["Frequency"].idxmin()

    # -----------------------------------------------------
    # 6. Create binary target
    # -----------------------------------------------------
    rfm["is_high_risk"] = (rfm["cluster"] == worst_cluster).astype(int)

    return rfm[["CustomerId", "is_high_risk"]]

