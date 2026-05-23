import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error, mean_squared_error

st.set_page_config(page_title="Sales Forecasting", page_icon="📈", layout="wide")
st.title("📈 Rossmann Sales Forecasting")
st.markdown("Forecast future sales using Facebook Prophet model.")

@st.cache_data
def load_data():
    df = pd.read_csv("train_small.csv", low_memory=False)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df[(df["Open"] == 1) & (df["Sales"] > 0)]
    return df

@st.cache_resource
def train_model(store_num, _df):
    store_df = _df[_df["Store"] == store_num][["Date", "Sales"]]
    store_df = store_df.rename(columns={"Date": "ds", "Sales": "y"})
    model = Prophet(yearly_seasonality=True, weekly_seasonality=True,
                    daily_seasonality=False, seasonality_mode="additive")
    model.fit(store_df)
    return model, store_df

st.sidebar.header("⚙️ Settings")
mode = st.sidebar.radio("Choose Mode", ["Use Store Number", "Upload CSV"])
forecast_days = st.sidebar.slider("Forecast Days", 30, 180, 90)

if mode == "Use Store Number":
    store_num = st.sidebar.number_input("Store Number", min_value=1, max_value=50, value=1)
    st.subheader(f"🏪 Store {store_num} — Sales Forecast")
    try:
        df = load_data()
        with st.spinner("Training model..."):
            model, store_df = train_model(store_num, df)
            future = model.make_future_dataframe(periods=forecast_days)
            forecast = model.predict(future)
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(store_df["ds"], store_df["y"], color="steelblue", linewidth=0.8, label="Actual")
        ax.plot(forecast["ds"], forecast["yhat"], color="tomato", linewidth=1.2, label="Forecast")
        ax.fill_between(forecast["ds"], forecast["yhat_lower"], forecast["yhat_upper"],
                        alpha=0.2, color="tomato")
        ax.set_title(f"Store {store_num} — Forecast ({forecast_days} days)")
        ax.legend()
        st.pyplot(fig)
        actual = store_df["y"].values
        predicted = forecast["yhat"][:len(actual)].values
        mae = mean_absolute_error(actual, predicted)
        rmse = np.sqrt(mean_squared_error(actual, predicted))
        col1, col2, col3 = st.columns(3)
        col1.metric("MAE", f"{mae:.0f}")
        col2.metric("RMSE", f"{rmse:.0f}")
        col3.metric("Error %", f"{mae/actual.mean()*100:.2f}%")
    except FileNotFoundError:
        st.error("train_small.csv not found.")
else:
    st.subheader("📂 Upload Your CSV")
    uploaded = st.file_uploader("Upload CSV with 'Date' and 'Sales' columns", type="csv")
    if uploaded:
        df_upload = pd.read_csv(uploaded)
        df_upload["Date"] = pd.to_datetime(df_upload["Date"])
        df_upload = df_upload.rename(columns={"Date": "ds", "Sales": "y"})
        st.write("Preview:", df_upload.head())
        with st.spinner("Training model..."):
            model = Prophet(yearly_seasonality=True, weekly_seasonality=True,
                            daily_seasonality=False, seasonality_mode="additive")
            model.fit(df_upload)
            future = model.make_future_dataframe(periods=forecast_days)
            forecast = model.predict(future)
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(df_upload["ds"], df_upload["y"], color="steelblue", linewidth=0.8, label="Actual")
        ax.plot(forecast["ds"], forecast["yhat"], color="tomato", linewidth=1.2, label="Forecast")
        ax.fill_between(forecast["ds"], forecast["yhat_lower"], forecast["yhat_upper"],
                        alpha=0.2, color="tomato")
        ax.set_title("Sales Forecast")
        ax.legend()
        st.pyplot(fig)
        mae = mean_absolute_error(df_upload["y"].values, forecast["yhat"][:len(df_upload)].values)
        rmse = np.sqrt(mean_squared_error(df_upload["y"].values, forecast["yhat"][:len(df_upload)].values))
        col1, col2, col3 = st.columns(3)
        col1.metric("MAE", f"{mae:.0f}")
        col2.metric("RMSE", f"{rmse:.0f}")
        col3.metric("Error %", f"{mae/df_upload['y'].mean()*100:.2f}%")
