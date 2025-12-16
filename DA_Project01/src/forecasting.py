import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np


def prepare_monthly_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepares aggregated monthly data for ML forecasting.
    Returns a DataFrame with:
    - year
    - month
    - month_index (numeric timeline)
    - total_sales
    """

    monthly = (
        df.groupby(["year", "month"])["total_sales"]
        .sum()
        .reset_index()
        .sort_values(["year", "month"])
    )

    # Create a continuous timeline for regression
    monthly["month_index"] = np.arange(len(monthly)) + 1

    return monthly


def train_sales_forecast_model(monthly_df: pd.DataFrame):
    """
    Trains a Linear Regression model to predict future monthly sales.
    """

    X = monthly_df[["month_index"]]
    y = monthly_df["total_sales"]

    model = LinearRegression()
    model.fit(X, y)

    return model


def predict_next_months(model, last_month_index: int, n_months=3):
    """
    Predict sales for the next N months.
    Returns a list of predictions.
    """

    future_indexes = np.arange(last_month_index + 1, last_month_index + n_months + 1).reshape(-1, 1)
    predictions = model.predict(future_indexes)

    return predictions
