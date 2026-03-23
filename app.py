import streamlit as st
import pandas as pd
import numpy as np
import pickle
import datetime
import matplotlib.pyplot as plt

# ---------------- LOAD MODEL ----------------
model = pickle.load(open("nifty_xgb.pkl","rb"))

# ---------------- LOAD DATA ----------------
df = pd.read_excel("NIFTY 50_historical_data_Daily_5_Years.xlsx")
df = df.sort_values("Date")

close_prices = df['Close'].values

# ---------------- UI ----------------
st.title("📈 NIFTY AI Prediction System")

future_date = st.date_input(
    "Select Future Prediction Date",
    min_value=datetime.date.today()
)

predict_btn = st.button("Predict")

# ---------------- PREDICTION ----------------
if predict_btn:

    today = datetime.date.today()
    days_ahead = (future_date - today).days

    if days_ahead <= 0:
        st.warning("Select valid future date")
    else:

        lookback = 10
        temp = list(close_prices[-lookback:])
        preds = []

        for i in range(days_ahead):

            x_input = np.array(temp[-lookback:]).reshape(1,-1)
            pred = model.predict(x_input)[0]

            temp.append(pred)
            preds.append(pred)

        predicted_price = preds[-1]

        last_close = close_prices[-1]
        change = predicted_price - last_close
        pct = (change/last_close)*100

        st.success(f"Predicted NIFTY on {future_date} : {predicted_price:.2f}")

        if change > 0:
            st.info(f"📈 Bullish Trend Expected (+{pct:.2f}%)")
        else:
            st.info(f"📉 Bearish Trend Expected ({pct:.2f}%)")

        # ---------- Chart ----------
        st.subheader("Forecast Chart")

        hist = close_prices[-60:]
        chart_vals = list(hist) + preds

        plt.figure(figsize=(10,4))
        plt.plot(chart_vals)
        plt.axvline(x=59, color='red')
        st.pyplot(plt)
