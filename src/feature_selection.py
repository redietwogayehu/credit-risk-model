import numpy as np
import pandas as pd

def calculate_iv(df, feature, target, bins=10):
    df = df[[feature, target]].copy()

    if df[feature].dtype != "object":
        df[feature] = pd.qcut(df[feature], q=bins, duplicates="drop")

    grouped = df.groupby(feature)[target].agg(["count", "sum"])
    grouped.columns = ["total", "bad"]

    grouped["good"] = grouped["total"] - grouped["bad"]

    grouped["bad_dist"] = grouped["bad"] / grouped["bad"].sum()
    grouped["good_dist"] = grouped["good"] / grouped["good"].sum()

    grouped["woe"] = np.log(grouped["good_dist"] / grouped["bad_dist"])
    grouped["iv"] = (grouped["good_dist"] - grouped["bad_dist"]) * grouped["woe"]

    return grouped["iv"].sum()