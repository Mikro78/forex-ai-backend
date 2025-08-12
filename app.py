import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.interpolate import CubicSpline
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import time
import smtplib
from email.mime.text import MIMEText
import schedule
import requests  # За Twilio/WhatsApp

# --- EMD Implementation (simple version from paper description) ---
def emd_decompose(signal):
    # Simple EMD with cubic spline
    imfs = []
    r = signal.copy()
    while np.std(r) > 1e-6:  # Stop condition
        h = r.copy()
        while True:
            # Find extrema
            max_idx = np.where((h[1:-1] > h[:-2]) & (h[1:-1] > h[2:]))[0] + 1
            min_idx = np.where((h[1:-1] < h[:-2]) & (h[1:-1] < h[2:]))[0] + 1
            if len(max_idx) < 2 or len(min_idx) < 2:
                break
            # Spline envelopes
            upper = CubicSpline(max_idx, h[max_idx])(np.arange(len(h)))
            lower = CubicSpline(min_idx, h[min_idx])(np.arange(len(h)))
            m = (upper + lower) / 2
            h_prev = h
            h = h - m
            if np.std(h - h_prev) < 1e-6:
                break
        imfs.append(h)
        r = r - h
    imfs.append(r)  # Residue
    return imfs

# --- Neural Network Models ---
class NARX(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, delay_input=10, delay_output=10):
        super().__init__()
        self.fc1 = nn.Linear(input_size * delay_input + output_size * delay_output, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.tanh(self.fc1(x))
        return self.fc2(x)

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# Similar definitions for GRU, BiLSTM, AttentionLSTM, Transformer, TCN, NBEATS, TFT, Autoformer...
# (Ограничение на пространството – имплементирай аналогично с PyTorch. За Transformer: nn.Transformer; TCN: nn.Conv1d stacks; NBEATS: stack of blocks with basis expansion.)

# EMD-NARX: Decompose with EMD, train NARX on each IMF, sum predictions.
def train_emd_narx(data, input_cols, target_col):
    imfs = emd_decompose(data[target_col].values)
    predictions = []
    for imf in imfs:
        imf_df = pd.DataFrame({'IMF': imf})
        imf_df[input_cols] = data[input_cols].values[:len(imf)]
        # Train NARX on imf_df...
        # (Use train_model function below)
        pred = model.predict(...)  # Placeholder
        predictions.append(pred)
    return np.sum(predictions, axis=0)

# General train function
def train_model(model, X_train, y_train, epochs=50, batch_size=32):
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    dataset = TensorDataset(torch.tensor(X_train).float(), torch.tensor(y_train).float())
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    start_time = time.time()
    for epoch in range(epochs):
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            output = model(X_batch.unsqueeze(1))  # Adjust for seq dim
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
    train_time = time.time() - start_time
    return model, train_time, scaler

# --- Data Fetch ---
@st.cache_data(ttl=300)  # Cache for 5 min
def fetch_data(interval='5m', years=5):
    ticker = 'EURUSD=X'
    end = datetime.now()
    start = end - timedelta(days=365 * years)
    data = yf.download(ticker, start=start, end=end, interval=interval)
    data['Target'] = data['Close'].shift(-1)
    data.dropna(inplace=True)
    return data

# --- Main App ---
st.title("FOREX AI Trading System: EUR/USD Predictor")
intervals = st.multiselect("Select Intervals", ['5m', '15m', '30m', '1h', '4h', '1d'], default=['5m'])
models_select = st.multiselect("Select Models", ['Combination NARX-EMD', 'NARX', 'EMD-NARX', 'LSTM', 'GRU', 'Transformer', 'Bi-LSTM', 'Attention-LSTM', 'TCN', 'NBEATS', 'Prophet', 'TFT', 'Autoformer', 'NARX-CNN'], default=['NARX', 'EMD-NARX', 'LSTM'])

data_5m = fetch_data('5m', 5)  # Main data
st.write("Last 5 rows of data:", data_5m.tail())

# Training and Prediction
results = {}
times = {}
scalers = {}
for model_name in models_select:
    # Example for LSTM
    if model_name == 'LSTM':
        model = LSTMModel(input_size=3, hidden_size=50, output_size=1)  # OHL input
        X = data_5m[['Open', 'High', 'Low']].values
        y = data_5m['Target'].values.reshape(-1, 1)
        model, train_time, scaler = train_model(model, X[:-1], y[:-1])  # Train on all but last
        pred = model(torch.tensor(scaler.transform(X[-1].reshape(1, -1))).float().unsqueeze(1)).item()
        results[model_name] = pred
        times[model_name] = train_time
        scalers[model_name] = scaler
    # Add logic for other models similarly...

# Multi-interval consensus
consensus = []
for intv in intervals:
    data_intv = fetch_data(intv, 5)
    # Predict direction (up/down)
    pred_intv = results['LSTM']  # Example, use average
    direction = 'Buy' if pred_intv > data_intv['Close'].iloc[-1] else 'Sell'
    consensus.append(direction)
if all(d == 'Buy' for d in consensus) or consensus.count('Buy') > len(intervals)/2:
    st.success("Strong Buy Signal!")

# Dashboard
st.subheader("Model Performance")
perf_df = pd.DataFrame({'Model': list(results.keys()), 'Predicted Close': list(results.values()), 'Train Time (s)': list(times.values())})
st.table(perf_df)

# Best model
best_model = perf_df.loc[perf_df['Train Time (s)'].idxmin()]  # Or by accuracy
st.write(f"Best Model: {best_model['Model']} with Pred: {best_model['Predicted Close']}")

# Last 10 Predictions (simulate history)
history = pd.DataFrame({'Actual': np.random.rand(10), 'Pred': np.random.rand(10)})  # Replace with real
history['Success'] = np.where(np.abs(history['Actual'] - history['Pred']) < 0.001, '✓', '✗')
st.table(history)

# Charts
fig, ax = plt.subplots()
ax.plot(data_5m['Close'][-100:], label='Actual')
for model, pred in results.items():
    ax.scatter(len(data_5m), pred, label=model)
st.pyplot(fig)

# Backtesting
def backtest(models, data):
    bt_results = {}
    for model_name in models:
        preds = []
        for i in range(len(data)-100, len(data)):
            # Simulate prediction
            pred = np.mean(data['Close'].iloc[i-10:i])  # Dummy
            preds.append(pred)
        actual = data['Target'].iloc[-100:]
        mse = mean_squared_error(actual, preds)
        r2 = r2_score(actual, preds)
        dir_acc = np.mean(np.sign(np.diff(preds)) == np.sign(np.diff(actual)))
        bt_results[model_name] = {'MSE': mse, 'R2': r2, 'Dir Acc %': dir_acc*100}
    return pd.DataFrame(bt_results)

bt_df = backtest(models_select, data_5m)
st.table(bt_df)
fig_bt = plt.figure()
bt_df.plot(kind='bar')
st.pyplot(fig_bt)

# Notifications
email = st.text_input("Your Email")
other_emails = st.text_input("Additional Emails (comma-separated)")
def send_notification(signal):
    msg = MIMEText(f"FOREX Signal: {signal} for EUR/USD")
    msg['Subject'] = 'AI Forex Alert'
    server = smtplib.SMTP('smtp.gmail.com', 587)  # Configure your SMTP
    server.login('your@gmail.com', 'pass')
    server.sendmail('from@gmail.com', [email] + other_emails.split(','), msg.as_string())
    # For WhatsApp: Use Twilio
    twilio_url = 'https://api.twilio.com/2010-04-01/Accounts/YOUR_SID/Messages.json'
    data = {'To': 'whatsapp:+your_number', 'From': 'whatsapp:+twilio_number', 'Body': signal}
    requests.post(twilio_url, data=data, auth=('YOUR_SID', 'YOUR_TOKEN'))

# Scheduler for every 5 min
def job():
    # Refetch, retrain on new data, predict, send
    new_data = fetch_data('5m', 5)
    # Fine-tune models on last 100 rows...
    signal = f"Predicted Close: {results['LSTM']}"  # Example
    send_notification(signal)

schedule.every(5).minutes.do(job)

# Live Trading (placeholder - integrate OANDA API)
if st.button("Enable Live Trading"):
    # Use oandapyV20 for auto trades
    pass

# Run scheduler in background
while True:
    schedule.run_pending()
    time.sleep(1)