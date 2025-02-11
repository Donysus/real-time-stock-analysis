# 📈 AI-Powered Real-Time Stock Analysis & Prediction  

This project is an **AI-driven stock trend prediction system** that continuously fetches **real-time stock prices**, applies **technical indicators (Moving Averages)**, and predicts stock price movements using **RandomForestClassifier**. It also **automatically retrains itself periodically** based on live market data.

## 📌 Features  
✅ **Real-Time Market Data** – Fetches live stock prices using Yahoo Finance (`yfinance`).  
✅ **Technical Indicator-Based Predictions** – Uses **5-day & 10-day moving averages** for trend analysis.  
✅ **Machine Learning (RandomForestClassifier)** – Trains a **predictive model** to classify stock movements.  
✅ **Periodic Model Retraining** – Ensures **dynamic adaptation** to market trends.  
✅ **Real-Time Visualization** – Plots stock prices, moving averages & predicted signals dynamically.  

---

## ⚡ Installation  

# Step 1: Install Dependencies  
Run the following command to install required libraries:  
```bash
pip install -r requirements.txt
```
## Step 2: Run the Code
Execute the Python script for real-time stock trend prediction:
```bash
python stock_analysis.py
```
## 🛠 Project Structure
```bash
real-time-stock-analysis/
│── stock_analysis.py         # Main AI-powered stock prediction script
│── requirements.txt          # List of dependencies
│── README.md                 # Documentation
│── .gitignore                # Ignore unnecessary files
│── models/                   # Saved trained models
│── results/                  # Performance visualizations
```

## 📊 How It Works
1️⃣ Fetches real-time stock prices from Yahoo Finance (yfinance).
2️⃣ Computes 5-day & 10-day moving averages for trend analysis.
3️⃣ Trains a RandomForestClassifier model for trend prediction.
4️⃣ Predicts buy/sell signals based on latest data.
5️⃣ Updates live visualization of stock movements.
6️⃣ Retrains the model every 60 minutes for continuous improvement.

If you have any suggestions: Mail me at help.brokenbrains@gmail.com
