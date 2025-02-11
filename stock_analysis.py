import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import logging
import time
from datetime import datetime
import pytz
import joblib
import pandas_market_calendars as mcal

# Set up logging
logging.basicConfig(level=logging.INFO)

# Check if the US market is open
def is_us_market_open():
    nyse = mcal.get_calendar("NYSE")
    now = datetime.now(pytz.timezone("US/Eastern"))
    today = pd.Timestamp.now(tz="US/Eastern").normalize()
    schedule = nyse.schedule(start_date=today, end_date=today)  # Correctly fetch schedule
    if schedule.empty:
        return False  # Market holiday
    market_open = schedule.iloc[0]["market_open"].astimezone(pytz.timezone("US/Eastern"))
    market_close = schedule.iloc[0]["market_close"].astimezone(pytz.timezone("US/Eastern"))
    return market_open <= now < market_close

# Fetch historical data
def fetch_data(stock_symbol, period="1mo", interval="1d"):
    try:
        data = yf.download(stock_symbol, period=period, interval=interval)
        if data.empty:
            logging.warning(f"No data fetched for {stock_symbol}.")
        else:
            logging.info(f"Fetched data for {stock_symbol}.")
        return data
    except Exception as e:
        logging.error(f"Error fetching data: {e}", exc_info=True)
        return None

# Feature engineering
def create_features(data):
    if len(data) < 10:
        logging.warning("Not enough data for feature engineering.")
        return pd.DataFrame()
    data["MA_5"] = data["Close"].rolling(window=5).mean()
    data["MA_10"] = data["Close"].rolling(window=10).mean()
    data["Signal"] = np.where(data["MA_5"] > data["MA_10"], 1, 0)
    return data.dropna()

# Train the model
def train_model(data):
    if data.empty:
        logging.error("No data available for training the model.")
        return None
    X = data[["MA_5", "MA_10"]]
    y = data["Signal"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    logging.info(classification_report(y_test, predictions))
    joblib.dump(model, "trained_model.pkl")
    logging.info("Model saved successfully.")
    return model

# Load the saved model
def load_model():
    try:
        model = joblib.load("trained_model.pkl")
        logging.info("Model loaded successfully.")
        return model
    except FileNotFoundError:
        logging.error("Trained model file not found. Please train the model first.")
        return None

# Analyze historical data when the market is closed
def analyze_historical_data(stock_symbol):
    logging.info("Analyzing historical data while the market is closed.")
    historical_data = fetch_data(stock_symbol, period="1y", interval="1d")
    if historical_data is not None and not historical_data.empty:
        features_data = create_features(historical_data)
        if not features_data.empty:
            train_model(features_data)
            logging.info("Model retrained with historical data.")
    else:
        logging.warning("No historical data available for analysis.")

# Real-time prediction with periodic retraining
def real_time_prediction(stock_symbol):
    model = load_model()
    if model is None:
        logging.warning("Model not found. Training a new model...")
        historical_data = fetch_data(stock_symbol, period="1y", interval="1d")
        if historical_data is not None:
            features_data = create_features(historical_data)
            model = train_model(features_data)

    fig, ax = plt.subplots(figsize=(14, 7))
    close_line, = ax.plot([], [], label="Close Price", alpha=0.5)
    ma5_line, = ax.plot([], [], label="5-Day Moving Average", alpha=0.75)
    ma10_line, = ax.plot([], [], label="10-Day Moving Average", alpha=0.75)
    ax.legend()
    plt.ion()

    last_retrain = datetime.now()
    while True:
        if is_us_market_open():
            logging.info("Market is open.")
            try:
                data = fetch_data(stock_symbol, interval="1m")
                if data is None or data.empty:
                    logging.warning("No data fetched. Retrying in a minute...")
                    time.sleep(60)
                    continue

                features = create_features(data)
                if features.empty:
                    logging.warning("No features available for prediction.")
                    continue

                latest_data = features.iloc[-1][["MA_5", "MA_10"]].values.reshape(1, -1)
                prediction = model.predict(latest_data)

                close_line.set_data(data.index, data["Close"])
                ma5_line.set_data(data.index, data["MA_5"])
                ma10_line.set_data(data.index, data["MA_10"])

                ax.set_xlim(data.index[0], data.index[-1])
                ax.set_ylim(data["Close"].min() * 0.95, data["Close"].max() * 1.05)
                plt.pause(0.01)

                if (datetime.now() - last_retrain).total_seconds() >= 3600:
                    logging.info("Retraining the model...")
                    features_data = create_features(data)
                    model = train_model(features_data)
                    last_retrain = datetime.now()

            except Exception as e:
                logging.error("Error during real-time prediction", exc_info=True)
        else:
            logging.info("Market is closed. Analyzing historical data...")
            analyze_historical_data(stock_symbol)
            logging.info("Waiting for the next trading session...")
            time.sleep(300)

if __name__ == "__main__":
    stock_symbol = "AAPL"  # Example: Apple Inc.
    real_time_prediction(stock_symbol)
