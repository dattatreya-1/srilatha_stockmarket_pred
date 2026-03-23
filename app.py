import streamlit as st
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model
import datetime
import matplotlib.pyplot as plt

# ---------------- CONFIG ----------------
st.set_page_config(
    page_title="NIFTY AI Predictor",
    page_icon="📈",
    layout="centered"
)

# ---------------- LOAD FILES ----------------
model = load_model("model/nifty_lstm_model.h5")
scaler = pickle.load(open("model/scaler.pkl","rb"))

df = pd.read_excel("data/NIFTY_50_historical_data.xlsx")
df = df.sort_values("Date")

close_prices = df['Close'].values.reshape(-1,1)

# ---------------- UI ----------------
st.title("📈 NIFTY-50 AI Prediction System")

future_date = st.date_input(
    "Select Future Prediction Date",
    min_value=datetime.date.today()
)

predict_btn = st.button("Predict")

# ---------------- PREDICTION ----------------
if predict_btn:

    today = datetime.date.today()
    days_ahead = (future_date - today).days

    if days_ahead == 0:
        st.warning("Please select future date")
    else:

        last_60 = close_prices[-60:]
        last_60_scaled = scaler.transform(last_60)

        temp_input = list(last_60_scaled.flatten())

        preds = []

        for i in range(days_ahead):

            x_input = np.array(temp_input[-60:])
            x_input = x_input.reshape(1,60,1)

            pred = model.predict(x_input, verbose=0)
            temp_input.append(pred[0][0])

            preds.append(pred[0][0])

        preds = scaler.inverse_transform(
            np.array(preds).reshape(-1,1)
        )

        predicted_price = preds[-1][0]

        last_close = close_prices[-1][0]

        change = predicted_price - last_close
        pct_change = (change / last_close) * 100

        # ---------- OUTPUT ----------
        st.success(f"Predicted NIFTY Value on {future_date} : {predicted_price:.2f}")

        if change > 0:
            st.info(f"📈 Bullish Trend Expected (+{pct_change:.2f}%)")
        else:
            st.info(f"📉 Bearish Trend Expected ({pct_change:.2f}%)")

        # ---------- CHART ----------
        st.subheader("Prediction Chart")

        hist = close_prices[-60:].flatten()

        chart_values = list(hist) + list(preds.flatten())

        plt.figure(figsize=(10,4))
        plt.plot(chart_values)
        plt.axvline(x=59, color='red')
        plt.title("Last 60 Days + Future Prediction")
        st.pyplot(plt)
