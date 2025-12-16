import pandas as pd

def load_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and preprocess the dataset:
    - Standardizes column names
    - Converts date column to datetime
    - Handles missing values
    - Ensures correct data types
    - Creates additional features (month_name, revenue_per_unit)
    """

    # 1. Standardize column names
    df.columns = df.columns.str.lower().str.strip()

    # 2. Convert date column to datetime
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # 3. Handle missing values (if any)
    df = df.dropna(subset=["date", "city", "store", "product_category"])

    # 4. Ensure data types
    df["unit_price"] = df["unit_price"].astype(float)
    df["quantity"] = df["quantity"].astype(int)
    df["total_sales"] = df["total_sales"].astype(float)

    # 5. Create month name for analysis
    df["month_name"] = df["date"].dt.month_name()

    # 6. Create a profit estimate (optional)
    # Assume 30% margin
    df["estimated_profit"] = df["total_sales"] * 0.30

    return df


def load_and_clean(path: str) -> pd.DataFrame:
    """
    Utility function used by Streamlit and analysis scripts.
    Loads and cleans the dataset in one call.
    """
    df = load_dataset(path)
    df = clean_dataset(df)
    return df
