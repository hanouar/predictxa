import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# Set page config
st.set_page_config(page_title="Gold Price Predictor", page_icon="💰", layout="wide")

# Load models once using cache
@st.cache_resource
def load_models():
    lstm_model = load_model("lstm_gold_model.h5")
    xgb_model = joblib.load("xgb_gold_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return lstm_model, xgb_model, scaler

def get_latest_data():
    gold_latest = yf.download("GC=F", period="5d", interval="1d")
    return gold_latest.tail(1)

def predict_price(lstm_model, xgb_model, scaler):
    # Fetch and prepare data
    gold_latest = get_latest_data()
    current_price = gold_latest['Close'].values[0]
    
    # Preprocess
    latest_features = gold_latest[['Open', 'High', 'Low', 'Close', 'Volume']].values
    latest_features = scaler.transform(latest_features)
    
    # LSTM prediction
    latest_features_lstm = latest_features.reshape((1, 1, latest_features.shape[1]))
    lstm_prediction = lstm_model.predict(latest_features_lstm)
    
    # XGBoost prediction
    final_input = np.hstack((latest_features, lstm_prediction.reshape(1, -1)))
    xgb_prediction = xgb_model.predict(final_input)
    
    # Calculate prices
    predicted_price_change = lstm_prediction[0][0] * (current_price * 0.02)
    predicted_price = current_price + (predicted_price_change if xgb_prediction[0] == 1 else -predicted_price_change)
    breakpoint_price = current_price * 1.01 if xgb_prediction[0] == 1 else current_price * 0.99
    
    return current_price, predicted_price, breakpoint_price

def create_plot(current, breakpoint, predicted):
    fig, ax = plt.subplots(figsize=(10, 6))
    points = ["Current Price", "Breakpoint", "Predicted Price"]
    values = [current, breakpoint, predicted]
    
    ax.plot(points, values, marker='o', linestyle='--', color='blue')
    ax.set_title("Gold Price Prediction", fontsize=16)
    ax.set_ylabel("Price (USD)", fontsize=12)
    ax.grid(True)
    
    # Annotations
    ax.annotate(f"${current:.2f}", (0, current), textcoords="offset points", xytext=(0,10), ha='center')
    ax.annotate(f"${breakpoint:.2f}", (1, breakpoint), textcoords="offset points", xytext=(0,10), ha='center', color='red')
    ax.annotate(f"${predicted:.2f}", (2, predicted), textcoords="offset points", xytext=(0,10), ha='center', color='green')
    
    return fig

# Main app
def main():
    st.title("💰 Gold Price Prediction App")
    
    # Load models
    lstm_model, xgb_model, scaler = load_models()
    
    # Create sidebar
    st.sidebar.header("About")
    st.sidebar.info("This app predicts next-day gold prices using hybrid LSTM-XGBoost model.")
    
    # Prediction section
    if st.button("Refresh Predictions"):
        st.cache_data.clear()
    
    current_price, predicted_price, breakpoint_price = predict_price(lstm_model, xgb_model, scaler)
    
    # Display metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Current Price", f"${current_price:.2f}")
    with col2:
        st.metric("Breakpoint Price", f"${breakpoint_price:.2f}", 
                  delta="1% threshold", delta_color="off")
    with col3:
        st.metric("Predicted Price", f"${predicted_price:.2f}", 
                  delta=f"{(predicted_price - current_price):.2f}")
    
    # Show plot
    st.pyplot(create_plot(current_price, breakpoint_price, predicted_price))
    
    # Show raw data
    if st.checkbox("Show latest market data"):
        st.write(get_latest_data())

if __name__ == "__main__":
    main()
