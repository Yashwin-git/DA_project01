import pandas as pd

# ---------------------------------------------------------
# CITY-LEVEL ANALYSIS
# ---------------------------------------------------------

def revenue_by_city(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns total revenue grouped by city.
    """
    return (
        df.groupby("city")["total_sales"]
        .sum()
        .sort_values(ascending=False)
        .reset_index()
    )


def top_cities_by_profit(df: pd.DataFrame, n=5) -> pd.DataFrame:
    """
    Returns top N cities by estimated profit.
    """
    return (
        df.groupby("city")["estimated_profit"]
        .sum()
        .sort_values(ascending=False)
        .head(n)
        .reset_index()
    )


# ---------------------------------------------------------
# PRODUCT-LEVEL ANALYSIS
# ---------------------------------------------------------

def best_selling_categories(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns total sales grouped by product category.
    """
    return (
        df.groupby("product_category")["total_sales"]
        .sum()
        .sort_values(ascending=False)
        .reset_index()
    )


def top_selling_products(df: pd.DataFrame, n=5) -> pd.DataFrame:
    """
    Returns the top N products based on total revenue.
    Since this dataset uses only category-level data,
    this returns top N categories.
    """
    return best_selling_categories(df).head(n)


# ---------------------------------------------------------
# STORE-LEVEL ANALYSIS
# ---------------------------------------------------------

def revenue_by_store(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns store-wise total sales.
    """
    return (
        df.groupby("store")["total_sales"]
        .sum()
        .sort_values(ascending=False)
        .reset_index()
    )


# ---------------------------------------------------------
# TIME-BASED ANALYSIS
# ---------------------------------------------------------

def monthly_revenue(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns monthly revenue across all cities/stores.
    """
    return (
        df.groupby(["year", "month"])["total_sales"]
        .sum()
        .reset_index()
        .sort_values(["year", "month"])
    )


def monthly_category_sales(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns category-wise revenue each month.
    """
    return (
        df.groupby(["year", "month", "product_category"])["total_sales"]
        .sum()
        .reset_index()
        .sort_values(["year", "month"])
    )
