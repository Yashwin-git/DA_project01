import pandas as pd

def load_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = df.columns.str.lower().str.strip()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "city", "store", "product_category"])
    df["unit_price"] = df["unit_price"].astype(float)
    df["quantity"] = df["quantity"].astype(int)
    df["total_sales"] = df["total_sales"].astype(float)
    df["month_name"] = df["date"].dt.month_name()
    
    # 30% margin
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

