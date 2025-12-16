import sys, os
sys.path.append(os.path.abspath(".."))
sys.path.append(os.path.abspath("."))
sys.path.append(os.path.abspath("src"))

import streamlit as st
import pandas as pd
import plotly.express as px

# Correct import paths
from src.data_cleaning import load_and_clean
from src.analysis import (
    revenue_by_city,
    best_selling_categories,
    revenue_by_store,
    monthly_revenue
)
from src.forecasting import (
    prepare_monthly_data,
    train_sales_forecast_model,
    predict_next_months
)

# PAGE CONFIG
st.set_page_config(
    page_title="German Supermarket Sales Dashboard",
    layout="wide",
)

st.title("📊 German Supermarket Sales Analytics Dashboard")

# LOAD DATA
@st.cache_data
def load_data():
    df = load_and_clean("data/supermarket_sales_large.csv")
    return df

df = load_data()

# TABS
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Overview", "City Analysis", "Product Analysis", "Store Analysis", "Forecasting"]
)

# OVERVIEW TAB
with tab1:
    st.header("Dataset Overview")
    st.write("This dashboard analyzes sales across multiple German cities over 2 years.")
    
    st.subheader("Raw Data Sample")
    st.dataframe(df.head())

    st.subheader("Key Statistics")
    col1, col2, col3 = st.columns(3)

    col1.metric("Total Revenue (€)", f"{df['total_sales'].sum():,.2f}")
    col2.metric("Total Quantity Sold", f"{df['quantity'].sum():,}")
    col3.metric("Unique Cities", df["city"].nunique())

# CITY ANALYSIS
with tab2:
    st.header("City Performance")

    city_rev = revenue_by_city(df)
    fig = px.bar(city_rev, x="city", y="total_sales", title="Revenue by City", color="total_sales")
    st.plotly_chart(fig, use_container_width=True)

# PRODUCT ANALYSIS
with tab3:
    st.header("Product Category Analysis")

    category_rev = best_selling_categories(df)
    fig2 = px.bar(category_rev, x="product_category", y="total_sales",
                  title="Best Selling Product Categories", color="total_sales")
    st.plotly_chart(fig2, use_container_width=True)

# STORE ANALYSIS
with tab4:
    st.header("Store Performance")

    store_rev = revenue_by_store(df)
    fig3 = px.bar(store_rev, x="store", y="total_sales", title="Revenue by Store", color="total_sales")
    st.plotly_chart(fig3, use_container_width=True)

# FORECASTING TAB
with tab5:
    st.header("Sales Forecasting (Next 3 Months)")

    monthly_df = prepare_monthly_data(df)
    model = train_sales_forecast_model(monthly_df)
    predictions = predict_next_months(model, last_month_index=monthly_df["month_index"].max(), n_months=3)

    st.subheader("Predicted Sales (€)")
    st.write(pd.DataFrame({
        "Next Month": [predictions[0]],
        "Month +2": [predictions[1]],
        "Month +3": [predictions[2]],
    }))

    # Plot forecast vs history
    future_points = pd.DataFrame({
        "month_index": [monthly_df["month_index"].max() + i for i in range(1, 4)],
        "total_sales": predictions,
        "label": ["Forecast"] * 3
    })

    history = monthly_df[["month_index", "total_sales"]].copy()
    history["label"] = "Historical"

    combined = pd.concat([history, future_points])

    fig4 = px.line(combined, x="month_index", y="total_sales", color="label",
                   title="Historical Revenue vs Forecast")
    st.plotly_chart(fig4, use_container_width=True)
