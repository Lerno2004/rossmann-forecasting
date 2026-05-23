# 📈 Rossmann Sales Forecasting

A time series forecasting web app built with Facebook Prophet and Streamlit.

🔗 **Live App:** [Open App](https://rossmann-forecasting-3vmgimpyxjgvcul2rrfjrt.streamlit.app/)

---

## 📌 Project Overview

This project forecasts daily sales for Rossmann drug stores using historical data from 1,115 stores across Germany (2013–2015).

The goal is to predict future sales to help store managers make better inventory and staffing decisions.

---

## 📊 Dataset

- **Source:** [Kaggle — Rossmann Store Sales](https://www.kaggle.com/competitions/rossmann-store-sales)
- **Size:** 844,338 rows — 1,115 stores — 2.5 years
- **Features:** Date, Sales, Customers, Promo, StateHoliday, StoreType, CompetitionDistance

---

## 🔧 What I Did

### 1. Data Cleaning
- Removed closed store days (Open = 0)
- Removed zero sales when store was open (data errors)
- Fixed missing values using median and zero-fill strategies

### 2. Feature Engineering
- Extracted Year, Month, Day, WeekOfYear from Date
- Added IsWeekend flag
- Merged store metadata (StoreType, Assortment, CompetitionDistance)

### 3. Exploratory Data Analysis
- Visualized daily sales trends (2013–2015)
- Performed time series decomposition (Trend + Seasonality + Residual)
- Confirmed Additive model is appropriate (stable seasonality)

### 4. Forecasting Model — Facebook Prophet
- Trained Prophet model with yearly and weekly seasonality
- Additive seasonality mode
- Train/Test split at 2015-06-01

### 5. Model Evaluation
| Metric | Value |
|--------|-------|
| MAE | 663 |
| RMSE | 754 |
| Error % | 15.04% |

---

## 🌐 Web App Features

- **Store Mode** — Enter any store number (1–1115) and get a sales forecast
- **CSV Mode** — Upload your own CSV with Date and Sales columns
- **Adjustable forecast period** — 30 to 180 days
- **Live metrics** — MAE, RMSE, Error %

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python | Core language |
| Pandas | Data manipulation |
| Prophet | Time series forecasting |
| Matplotlib | Visualization |
| Streamlit | Web app |
| GitHub + Streamlit Cloud | Deployment |

---

## 🚀 Run Locally

```bash
git clone https://github.com/Lerno2004/rossmann-forecasting.git
cd rossmann-forecasting
pip install -r requirements.txt
streamlit run app.py
```

---

## 👤 Author

**Lernik Petrosyan**
[GitHub](https://github.com/Lerno2004)