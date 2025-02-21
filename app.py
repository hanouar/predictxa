import yfinance as yf
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier

# -------------------- Part 1: Training and Saving Models --------------------

# Download Gold Prices
gold_data = yf.download("GC=F", start="2010-01-01", end="2025-02-20", interval="1d")
gold_data.dropna(inplace=True)

# Define target variable (1 if price increases next day, else 0)
gold_data['Target'] = (gold_data['Close'].shift(-1) > gold_data['Close']).astype(int)
gold_data.dropna(inplace=True)

# Prepare dataset
features = gold_data[['Open', 'High', 'Low', 'Close', 'Volume']]
target = gold_data['Target']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Normalize data
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Reshape for LSTM
X_train_lstm = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test_lstm = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

# Build LSTM Model
lstm_model = Sequential([
    LSTM(100, return_sequences=True, input_shape=(1, X_train.shape[1])),
    Dropout(0.3),
    LSTM(100, return_sequences=False),
    Dropout(0.3),
    Dense(50, activation='relu'),
    Dense(1, activation='sigmoid')
])

lstm_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train LSTM
lstm_model.fit(X_train_lstm, y_train, epochs=50, batch_size=32, validation_data=(X_test_lstm, y_test), verbose=1)

# Extract LSTM Features
lstm_features_train = lstm_model.predict(X_train_lstm)
lstm_features_test = lstm_model.predict(X_test_lstm)

# Combine LSTM features with original dataset for XGBoost
X_train_hybrid = np.hstack((X_train, lstm_features_train))
X_test_hybrid = np.hstack((X_test, lstm_features_test))

# Train XGBoost
xgb_model = XGBClassifier(n_estimators=200, learning_rate=0.05)
xgb_model.fit(X_train_hybrid, y_train)

# Save models
lstm_model.save("lstm_gold_model.h5")
joblib.dump(xgb_model, "xgb_gold_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("Models saved: lstm_gold_model.h5, xgb_gold_model.pkl, scaler.pkl")

# -------------------- Part 2: Loading Models and Making Predictions --------------------

# Load models
lstm_model = tf.keras.models.load_model("lstm_gold_model.h5")
xgb_model = joblib.load("xgb_gold_model.pkl")
scaler = joblib.load("scaler.pkl")

# Fetch real-time Gold price
gold_latest = yf.download("GC=F", period="5d", interval="1d")
gold_latest = gold_latest.tail(1)

# Extract current price
current_price = gold_latest['Close'].values[0]

# Preprocess latest data
latest_features = gold_latest[['Open', 'High', 'Low', 'Close', 'Volume']].values
latest_features = scaler.transform(latest_features)

# Reshape for LSTM
latest_features_lstm = latest_features.reshape((1, 1, latest_features.shape[1]))

# Predict price movement with LSTM
lstm_prediction = lstm_model.predict(latest_features_lstm)

# Combine LSTM output with features for XGBoost
final_input = np.hstack((latest_features, lstm_prediction.reshape(1, -1)))

# Predict movement with XGBoost
xgb_prediction = xgb_model.predict(final_input)

# Define predicted price based on movement
predicted_price_change = lstm_prediction[0][0] * (current_price * 0.02)  # Assuming 2% price fluctuation
predicted_price = current_price + (predicted_price_change if xgb_prediction[0] == 1 else -predicted_price_change)

# Define breakpoint price
breakpoint_price = current_price * 1.01 if xgb_prediction[0] == 1 else current_price * 0.99

# Extract scalar values from NumPy arrays
current_price_scalar = current_price.item()  # Convert to scalar
breakpoint_price_scalar = breakpoint_price.item()  # Convert to scalar
predicted_price_scalar = predicted_price.item()  # Convert to scalar

# Create a visualization
plt.figure(figsize=(8, 5))
plt.plot(["Current Price", "Breakpoint", "Predicted Price"],
         [current_price_scalar, breakpoint_price_scalar, predicted_price_scalar], marker='o', linestyle='dashed', color='b')

# Add labels
plt.xlabel("Price Movement Stages")
plt.ylabel("Gold Price (USD)")
plt.title("Updated Predicted Gold Price Movement and Breakpoint")

# Annotate points
plt.text(0, current_price_scalar, f"${current_price_scalar:.2f}", ha='right', fontsize=10)
plt.text(1, breakpoint_price_scalar, f"${breakpoint_price_scalar:.2f}", ha='center', fontsize=10, color='red')
plt.text(2, predicted_price_scalar, f"${predicted_price_scalar:.2f}", ha='left', fontsize=10, color='green')

# Show the plot
plt.grid()
plt.show()

# Output actual values
print(f"📌 Current Gold Price: ${current_price_scalar:.2f}")
print(f"📊 Predicted Gold Price for Tomorrow: ${predicted_price_scalar:.2f}")
print(f"🔴 Breakpoint Price: ${breakpoint_price_scalar:.2f}")