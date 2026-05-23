import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ---- Page Config ----
st.set_page_config(page_title="Sales Forecasting", page_icon="📈", layout="wide")
st.title("📈 Rossmann Sales Forecasting")
st.markdown("Forecast future sales using Facebook Prophet model.")

# ---- Sidebar ----
st.sidebar.header("⚙️ Settings")
mode = st.sidebar.radio("Choose Mode", ["Use Store Number", "Upload CSV"])
forecast_days = st.sidebar.slider("Forecast Days", 30, 180, 90)

# ---- Mode 1 — Store Number ----
if mode == "Use Store Number":
    store_num = st.sidebar.number_input("Store Number", min_value=1, max_value=1115, value=1)

    st.subheader(f"🏪 Store {store_num} — Sales Forecast")

    try:
        df_train = pd.read_csv("train.csv")
        df_train["Date"] = pd.to_datetime(df_train["Date"])
        df_clean = df_train[(df_train["Open"] == 1) & (df_train["Sales"] > 0)]

        store_df = df_clean[df_clean["Store"] == store_num][["Date", "Sales"]]
        store_df = store_df.rename(columns={"Date": "ds", "Sales": "y"})

        if len(store_df) < 30:
            st.error("Not enough data for this store.")
        else:
            with st.spinner("Training Prophet model..."):
                model = Prophet(
                    yearly_seasonality=True,
                    weekly_seasonality=True,
                    daily_seasonality=False,
                    seasonality_mode="additive"
                )
                model.fit(store_df)
                future = model.make_future_dataframe(periods=forecast_days)
                forecast = model.predict(future)

            # Plot
            fig, ax = plt.subplots(figsize=(14, 5))
            ax.plot(store_df["ds"], store_df["y"], 
                    color="steelblue", linewidth=0.8, label="Actual Sales")
            ax.plot(forecast["ds"], forecast["yhat"], 
                    color="tomato", linewidth=1.2, label="Forecast")
            ax.fill_between(forecast["ds"], 
                            forecast["yhat_lower"], 
                            forecast["yhat_upper"], 
                            alpha=0.2, color="tomato")
            ax.set_title(f"Store {store_num} — Sales Forecast ({forecast_days} days)")
            ax.set_ylabel("Sales")
            ax.legend()
            st.pyplot(fig)

            # Metrics
            actual = store_df["y"].values
            predicted = forecast["yhat"][:len(actual)].values
            mae = mean_absolute_error(actual, predicted)
            rmse = np.sqrt(mean_squared_error(actual, predicted))
            error_pct = mae / actual.mean() * 100

            col1, col2, col3 = st.columns(3)
            col1.metric("MAE", f"{mae:.0f}")
            col2.metric("RMSE", f"{rmse:.0f}")
            col3.metric("Error %", f"{error_pct:.2f}%")

    except FileNotFoundError:
        st.error("train.csv not found. Please place it in the same folder as app.py")

# ---- Mode 2 — Upload CSV ----
else:
    st.subheader("📂 Upload Your CSV")
    uploaded = st.file_uploader("Upload CSV with 'Date' and 'Sales' columns", type="csv")

    if uploaded:
        df_upload = pd.read_csv(uploaded)
        df_upload["Date"] = pd.to_datetime(df_upload["Date"])
        df_upload = df_upload.rename(columns={"Date": "ds", "Sales": "y"})

        st.write("Preview:", df_upload.head())

        with st.spinner("Training Prophet model..."):
            model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=False,
                seasonality_mode="additive"
            )
            model.fit(df_upload)
            future = model.make_future_dataframe(periods=forecast_days)
            forecast = model.predict(future)

        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(df_upload["ds"], df_upload["y"], 
                color="steelblue", linewidth=0.8, label="Actual Sales")
        ax.plot(forecast["ds"], forecast["yhat"], 
                color="tomato", linewidth=1.2, label="Forecast")
        ax.fill_between(forecast["ds"], 
                        forecast["yhat_lower"], 
                        forecast["yhat_upper"], 
                        alpha=0.2, color="tomato")
        ax.set_title("Sales Forecast")
        ax.legend()
        st.pyplot(fig)

        mae = mean_absolute_error(df_upload["y"].values, 
                                   forecast["yhat"][:len(df_upload)].values)
        rmse = np.sqrt(mean_squared_error(df_upload["y"].values, 
                                           forecast["yhat"][:len(df_upload)].values))
        error_pct = mae / df_upload["y"].mean() * 100

        col1, col2, col3 = st.columns(3)
        col1.metric("MAE", f"{mae:.0f}")
        col2.metric("RMSE", f"{rmse:.0f}")
        col3.metric("Error %", f"{error_pct:.2f}%")
