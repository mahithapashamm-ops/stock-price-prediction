import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Stock Price Prediction", layout="centered")

st.title("📈 Stock Price Direction Prediction")

# ---------- Stock Selection ----------
popular_stocks = {
    "TCS (India)": "TCS.NS",
    "Infosys (India)": "INFY.NS",
    "Reliance (India)": "RELIANCE.NS",
    "Apple (US)": "AAPL",
    "Microsoft (US)": "MSFT",
    "Google (US)": "GOOGL"
}

stock_choice = st.selectbox(
    "Select a popular stock",
    options=list(popular_stocks.keys())
)

symbol = popular_stocks[stock_choice]

# ---------- Date Range ----------
period = st.selectbox(
    "Select historical data range",
    ["1y", "5y", "max"],
    index=1
)

# ---------- Prediction Button ----------
if st.button("Predict"):

    with st.spinner("Fetching data & training model..."):

        data = yf.download(symbol, period=period)

        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        numeric_cols = ["Open", "High", "Low", "Close", "Volume"]
        data = data[numeric_cols].dropna()

        data["Tomorrow_Close"] = data["Close"].shift(-1)
        data["Direction"] = (data["Tomorrow_Close"] > data["Close"]).astype(int)
        data.dropna(inplace=True)

        X = data[numeric_cols]
        y = data["Direction"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        model = SVC(kernel="rbf")
        model.fit(X_train, y_train)

        accuracy = accuracy_score(y_test, model.predict(X_test))
        prediction = model.predict(X_test[-1].reshape(1, -1))[0]

        direction = "📈 UP" if prediction == 1 else "📉 DOWN"

    # ---------- Results ----------
    st.success(f"Prediction for next trading day: **{direction}**")
    st.info(f"Model Accuracy: **{round(accuracy*100, 2)}%**")

    # ---------- Price Trend Graph ----------
    st.subheader("📊 Price Trend (Last Year)")
    last_year = data.tail(252)

    fig, ax = plt.subplots()
    ax.plot(last_year.index, last_year["Close"])
    ax.set_xlabel("Date")
    ax.set_ylabel("Close Price")

    st.pyplot(fig)