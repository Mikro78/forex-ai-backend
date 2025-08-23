from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
import smtplib
from email.mime.text import MIMEText
import schedule
import time
from threading import Thread
from twilio.rest import Client
import os
import asyncio
import logging
import talib

# Настройка на логове за дебъг
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://forex-ai-dashboard.vercel.app", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class LSTMModel(nn.Module):
    def __init__(self, input_size=3, hidden_size=50, output_size=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

class GRUModel(nn.Module):
    def __init__(self, input_size=3, hidden_size=50, output_size=1):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        out, _ = self.gru(x)
        out = self.fc(out[:, -1, :])
        return out

class NARXModel(nn.Module):
    def __init__(self, input_size=3, hidden_size=50, output_size=1):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = x.squeeze(1)
        x = torch.tanh(self.fc1(x))
        out = self.fc2(x)
        return out

def fetch_data(interval='5m', years=10):
    ticker = 'EURUSD=X'
    end = datetime.now()
    max_days = 60 if interval in ['5m', '15m', '30m'] else 730 if interval in ['1h', '4h'] else 365 * years
    start = end - timedelta(days=max_days)
    logger.info(f"Fetching data for {ticker} with interval {interval} from {start} to {end}")
    try:
        data = yf.download(ticker, start=start, end=end, interval=interval, progress=False, auto_adjust=False)
        data.columns = data.columns.get_level_values(0)  # Flatten multi-index columns
        if data.empty:
            raise ValueError(f"No data returned for {ticker} with interval {interval}")
        data['Target'] = data['Close'].shift(-1)
        close_array = data['Close'].to_numpy().flatten()
        logger.info(f"Close array shape: {close_array.shape}, first 5 values: {close_array[:5]}")
        if len(close_array) < 10:
            raise ValueError(f"Insufficient data for SMA/EMA: {len(close_array)} points, need at least 10")
        data['SMA10'] = talib.SMA(close_array, timeperiod=10)
        data['EMA10'] = talib.EMA(close_array, timeperiod=10)
        data.dropna(inplace=True)
        logger.info(f"Data fetched successfully: {len(data)} rows after dropna, columns: {data.columns.tolist()}")
        return data
    except Exception as e:
        logger.error(f"Failed to fetch data: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch data: {str(e)}")

def train_model(model, X, y, epochs=100):
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    X_tensor = torch.tensor(X_scaled).float().unsqueeze(1)
    y_scaler = MinMaxScaler()
    y_scaled = y_scaler.fit_transform(y.reshape(-1, 1)).squeeze()
    y_tensor = torch.tensor(y_scaled).float().unsqueeze(-1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    start_time = time.time()
    for _ in range(epochs):
        optimizer.zero_grad()
        output = model(X_tensor)
        loss = criterion(output, y_tensor)
        loss.backward()
        optimizer.step()
    return model, scaler, y_scaler, time.time() - start_time

latest_predictions = {}
models = {}
scalers = {}
y_scalers = {}
trained = False
last_email_time = {}

for interval in ['5m', '15m', '30m', '1h', '4h', '1d']:
    input_size = 5 if interval != '30m' else 10
    models[interval] = {
        'LSTM': LSTMModel(input_size=input_size),
        'GRU': GRUModel(input_size=input_size),
        'NARX': NARXModel(input_size=input_size),
    }
    scalers[interval] = {}
    y_scalers[interval] = {}

@app.post("/api/train")
async def train():
    global trained
    try:
        logger.info("Training started")
        for interval in ['5m', '15m', '30m', '1h', '4h', '1d']:
            data = fetch_data(interval, 10)
            if interval == '30m':
                data_5m = fetch_data('5m', 10)
                data_15m = fetch_data('15m', 10)
                data_5m_resampled = data_5m.resample('30min').mean().add_suffix('_5m')
                data_15m_resampled = data_15m.resample('30min').mean().add_suffix('_15m')
                data_5m_resampled['SMA10_5m'] = talib.SMA(data_5m_resampled['Close_5m'].to_numpy(), timeperiod=10)
                data_5m_resampled['EMA10_5m'] = talib.EMA(data_5m_resampled['Close_5m'].to_numpy(), timeperiod=10)
                data_15m_resampled['SMA10_15m'] = talib.SMA(data_15m_resampled['Close_15m'].to_numpy(), timeperiod=10)
                data_15m_resampled['EMA10_15m'] = talib.EMA(data_15m_resampled['Close_15m'].to_numpy(), timeperiod=10)
                data_5m_resampled = data_5m_resampled.reindex(data.index, method='ffill')
                data_15m_resampled = data_15m_resampled.reindex(data.index, method='ffill')
                data = data.join(data_5m_resampled[['Open_5m', 'High_5m', 'Low_5m', 'Close_5m', 'SMA10_5m', 'EMA10_5m']], how='left') \
                          .join(data_15m_resampled[['Open_15m', 'High_15m', 'Low_15m', 'Close_15m', 'SMA10_15m', 'EMA10_15m']], how='left')
                logger.info(f"Columns before dropna for 30m: {data.columns.tolist()}")
                data = data.dropna()
                logger.info(f"Columns after dropna for 30m: {data.columns.tolist()}")
                X = data[['Open', 'High', 'Low', 'SMA10', 'EMA10', 'Open_5m', 'High_5m', 'Low_5m', 'Open_15m', 'High_15m']].values
                logger.info(f"X shape before training: {X.shape}")
                if X.shape[1] != 10:
                    raise ValueError(f"Expected 10 features for 30m, got {X.shape[1]}: {data.columns.tolist()}")
            else:
                X = data[['Open', 'High', 'Low', 'SMA10', 'EMA10']].values
                logger.info(f"X shape before training for {interval}: {X.shape}")
            for name in models[interval]:
                logger.info(f"Training {name} model for interval {interval}")
                model, scaler, y_scaler, train_time = train_model(models[interval][name], X[:-1], data['Target'].values[:-1])
                models[interval][name] = model
                scalers[interval][name] = scaler
                y_scalers[interval][name] = y_scaler
        trained = True
        logger.info("Training completed successfully")
        return {"status": "Training completed", "trained": trained}
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")

@app.get("/api/signal")
async def get_signal(interval: str = "5m"):
    global trained
    try:
        logger.info(f"Processing signal for interval {interval}, trained status: {trained}")
        data = fetch_data(interval, 10)
        if interval == '30m':
            data_5m = fetch_data('5m', 10)
            data_15m = fetch_data('15m', 10)
            data_5m_resampled = data_5m.resample('30min').mean().add_suffix('_5m')
            data_15m_resampled = data_15m.resample('30min').mean().add_suffix('_15m')
            data_5m_resampled['SMA10_5m'] = talib.SMA(data_5m_resampled['Close_5m'].to_numpy(), timeperiod=10)
            data_5m_resampled['EMA10_5m'] = talib.EMA(data_5m_resampled['Close_5m'].to_numpy(), timeperiod=10)
            data_15m_resampled['SMA10_15m'] = talib.SMA(data_15m_resampled['Close_15m'].to_numpy(), timeperiod=10)
            data_15m_resampled['EMA10_15m'] = talib.EMA(data_15m_resampled['Close_15m'].to_numpy(), timeperiod=10)
            data_5m_resampled = data_5m_resampled.reindex(data.index, method='ffill')
            data_15m_resampled = data_15m_resampled.reindex(data.index, method='ffill')
            data = data.join(data_5m_resampled[['Open_5m', 'High_5m', 'Low_5m', 'Close_5m', 'SMA10_5m', 'EMA10_5m']], how='left') \
                      .join(data_15m_resampled[['Open_15m', 'High_15m', 'Low_15m', 'Close_15m', 'SMA10_15m', 'EMA10_15m']], how='left')
            logger.info(f"Columns before dropna for 30m: {data.columns.tolist()}")
            original_rows = len(data)
            data = data.dropna()
            logger.info(f"Columns after dropna for 30m: {data.columns.tolist()}, rows dropped: {original_rows - len(data)}")
            # Проверка на наличността на всички необходими колони
            required_columns = ['Open', 'High', 'Low', 'SMA10', 'EMA10', 'Open_5m', 'High_5m', 'Low_5m', 'Open_15m', 'High_15m']
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                raise ValueError(f"Missing columns in data: {missing_columns}")
            X = data[required_columns].values
            logger.info(f"X shape before prediction: {X.shape}, X columns: {required_columns}")
            if X.shape[1] != 10:
                raise ValueError(f"Expected 10 features, got {X.shape[1]}: {data.columns.tolist()}")
            if not trained:
                raise ValueError("Models not trained, please call /api/train first")
        else:
            X = data[['Open', 'High', 'Low', 'SMA10', 'EMA10']].values
            logger.info(f"X shape for {interval}: {X.shape}")
            if not trained:
                raise ValueError("Models not trained, please call /api/train first")
        
        predictions = []
        for name, model in models[interval].items():
            last_input = X[-1].reshape(1, -1)
            logger.info(f"Raw input shape for {name}: {last_input.shape}")
            last_input_scaled = scalers[interval][name].transform(last_input)
            logger.info(f"Transformed input shape for {name}: {last_input_scaled.shape}")
            last_input_tensor = torch.tensor(last_input_scaled).float().unsqueeze(1)
            pred_normalized = model(last_input_tensor).item()
            pred = y_scalers[interval][name].inverse_transform([[pred_normalized]])[0][0]
            predictions.append({'name': name, 'rate': pred, 'train_time': latest_predictions.get(interval, {}).get('train_time', 0.0)})
            logger.info(f"{name} prediction: {pred}")
        
        last_close = data['Close'].iloc[-1].item()
        latest_predictions[interval] = {
            'predictions': predictions,
            'last_close': last_close,
            'timestamp': datetime.now().isoformat(),
            'actual': None,
            'train_time': latest_predictions.get(interval, {}).get('train_time', 0.0),
            'mse': 0.0001
        }
        
        if interval == '30m':
            data_1d = fetch_data('1d', 10)
            X_1d = data_1d[['Open', 'High', 'Low', 'SMA10', 'EMA10']].values
            y_1d = data_1d['Target'].values
            predictions_1d = []
            for name in models[interval]:
                last_input_1d = X_1d[-1].reshape(1, -1)
                temp_scaler = MinMaxScaler()
                temp_scaler.fit(last_input_1d)
                last_input_1d_scaled = temp_scaler.transform(last_input_1d)
                last_input_tensor_1d = torch.tensor(last_input_1d_scaled).float().unsqueeze(1)
                # Добавяне на dummy features, за да съответства на 10 колони
                dummy = np.zeros((last_input_tensor_1d.shape[0], last_input_tensor_1d.shape[1], 5))
                last_input_tensor_1d = torch.cat((last_input_tensor_1d, torch.tensor(dummy).float()), dim=2)
                pred_normalized_1d = model(last_input_tensor_1d).item()
                pred_1d = y_scalers[interval][name].inverse_transform([[pred_normalized_1d]])[0][0]
                predictions_1d.append({'name': name, 'rate': pred_1d})
            latest_predictions[interval]['predictions_1d'] = predictions_1d
        
        if interval == '5m' and trained:
            current_time = time.time()
            if not last_email_time.get('5m') or (current_time - last_email_time['5m'] >= 300):
                try:
                    gmail_user = os.getenv('GMAIL_USER')
                    gmail_pass = os.getenv('GMAIL_PASS')
                    if gmail_user and gmail_pass:
                        pred_rates = ', '.join([f"{p['name']}: {p['rate']:.5f}" for p in predictions])
                        msg = MIMEText(f"FOREX Signal for 5m: Predictions - {pred_rates}, Last Close: {last_close:.5f}")
                        msg['Subject'] = 'AI Forex Signal'
                        msg['From'] = gmail_user
                        msg['To'] = 'mironedv@abv.bg'
                        with smtplib.SMTP('smtp.gmail.com', 587) as server:
                            server.starttls()
                            server.login(gmail_user, gmail_pass)
                            server.send_message(msg)
                        logger.info("Email sent successfully to mironedv@abv.bg")
                        last_email_time['5m'] = current_time
                except Exception as e:
                    logger.error(f"Email failed: {str(e)}")
        
        return latest_predictions[interval]
    except Exception as e:
        logger.error(f"Error processing signal: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing signal: {str(e)}")

@app.get("/api/backtest")
async def backtest(interval: str = "5m", days: int = 30):
    global trained
    try:
        logger.info(f"Backtesting for interval {interval}, days {days}, trained status: {trained}")
        if not trained:
            raise ValueError("Models not trained, please call /api/train first")
        data = fetch_data(interval, days / 365)
        if interval == '30m':
            data_5m = fetch_data('5m', days / 365)
            data_15m = fetch_data('15m', days / 365)
            data_5m_resampled = data_5m.resample('30min').mean().add_suffix('_5m')
            data_15m_resampled = data_15m.resample('30min').mean().add_suffix('_15m')
            data_5m_resampled['SMA10_5m'] = talib.SMA(data_5m_resampled['Close_5m'].to_numpy(), timeperiod=10)
            data_5m_resampled['EMA10_5m'] = talib.EMA(data_5m_resampled['Close_5m'].to_numpy(), timeperiod=10)
            data_15m_resampled['SMA10_15m'] = talib.SMA(data_15m_resampled['Close_15m'].to_numpy(), timeperiod=10)
            data_15m_resampled['EMA10_15m'] = talib.EMA(data_15m_resampled['Close_15m'].to_numpy(), timeperiod=10)
            data_5m_resampled = data_5m_resampled.reindex(data.index, method='ffill')
            data_15m_resampled = data_15m_resampled.reindex(data.index, method='ffill')
            data = data.join(data_5m_resampled[['Open_5m', 'High_5m', 'Low_5m', 'SMA10_5m', 'EMA10_5m']], how='left') \
                      .join(data_15m_resampled[['Open_15m', 'High_15m', 'Low_15m', 'SMA10_15m', 'EMA10_15m']], how='left')
            data = data.dropna()
            X = data[['Open', 'High', 'Low', 'SMA10', 'EMA10', 'Open_5m', 'High_5m', 'Low_5m', 'Open_15m', 'High_15m']].values
        else:
            X = data[['Open', 'High', 'Low', 'SMA10', 'EMA10']].values
        
        predictions = []
        actuals = data['Target'].values[1:]
        for name, model in models[interval].items():
            scaled_X = scalers[interval][name].transform(X)
            X_tensor = torch.tensor(scaled_X).float().unsqueeze(1)
            preds_normalized = model(X_tensor).detach().numpy().squeeze()
            preds = y_scalers[interval][name].inverse_transform(preds_normalized.reshape(-1, 1)).squeeze()
            mse = np.mean((preds[:-1] - actuals) ** 2) if len(actuals) > 0 else 0.0
            predictions.append({'name': name, 'predictions': preds.tolist(), 'mse': mse})
            logger.info(f"{name} backtest MSE: {mse}")
        
        return {
            'predictions': predictions,
            'actuals': actuals.tolist(),
            'timestamp': datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Backtest failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Backtest failed: {str(e)}")

@app.get("/api/chart")
async def get_chart_data(interval: str = "5m", days: int = 30):
    try:
        logger.info(f"Fetching chart data for interval {interval}, days {days}")
        data = fetch_data(interval, days / 365)
        if interval == '30m':
            data_5m = fetch_data('5m', days / 365)
            data_15m = fetch_data('15m', days / 365)
            data_5m_resampled = data_5m.resample('30min').mean().add_suffix('_5m')
            data_15m_resampled = data_15m.resample('30min').mean().add_suffix('_15m')
            data_5m_resampled['SMA10_5m'] = talib.SMA(data_5m_resampled['Close_5m'].to_numpy(), timeperiod=10)
            data_5m_resampled['EMA10_5m'] = talib.EMA(data_5m_resampled['Close_5m'].to_numpy(), timeperiod=10)
            data_15m_resampled['SMA10_15m'] = talib.SMA(data_15m_resampled['Close_15m'].to_numpy(), timeperiod=10)
            data_15m_resampled['EMA10_15m'] = talib.EMA(data_15m_resampled['Close_15m'].to_numpy(), timeperiod=10)
            data_5m_resampled = data_5m_resampled.reindex(data.index, method='ffill')
            data_15m_resampled = data_15m_resampled.reindex(data.index, method='ffill')
            data = data.join(data_5m_resampled[['Open_5m', 'High_5m', 'Low_5m', 'Close_5m', 'SMA10_5m', 'EMA10_5m']], how='left') \
                      .join(data_15m_resampled[['Open_15m', 'High_15m', 'Low_15m', 'Close_15m', 'SMA10_15m', 'EMA10_15m']], how='left')
            data = data.dropna()
        chart_data = {
            'time': data.index.astype(str).tolist(),
            'open': data['Open'].tolist(),
            'high': data['High'].tolist(),
            'low': data['Low'].tolist(),
            'close': data['Close'].tolist(),
            'sma10': data['SMA10'].tolist(),
            'ema10': data['EMA10'].tolist()
        }
        if interval == '30m':
            chart_data.update({
                'sma10_5m': data['SMA10_5m'].tolist(),
                'ema10_5m': data['EMA10_5m'].tolist(),
                'sma10_15m': data['SMA10_15m'].tolist(),
                'ema10_15m': data['EMA10_15m'].tolist()
            })
        return chart_data
    except Exception as e:
        logger.error(f"Chart data fetch failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Chart data fetch failed: {str(e)}")

@app.post("/api/retrain")
async def retrain():
    global trained
    try:
        logger.info("Retraining started")
        for interval in ['5m', '15m', '30m', '1h', '4h', '1d']:
            data = fetch_data(interval, 10)
            if interval == '30m':
                data_5m = fetch_data('5m', 10)
                data_15m = fetch_data('15m', 10)
                data_5m_resampled = data_5m.resample('30min').mean().add_suffix('_5m')
                data_15m_resampled = data_15m.resample('30min').mean().add_suffix('_15m')
                data_5m_resampled['SMA10_5m'] = talib.SMA(data_5m_resampled['Close_5m'].to_numpy(), timeperiod=10)
                data_5m_resampled['EMA10_5m'] = talib.EMA(data_5m_resampled['Close_5m'].to_numpy(), timeperiod=10)
                data_15m_resampled['SMA10_15m'] = talib.SMA(data_15m_resampled['Close_15m'].to_numpy(), timeperiod=10)
                data_15m_resampled['EMA10_15m'] = talib.EMA(data_15m_resampled['Close_15m'].to_numpy(), timeperiod=10)
                data_5m_resampled = data_5m_resampled.reindex(data.index, method='ffill')
                data_15m_resampled = data_15m_resampled.reindex(data.index, method='ffill')
                data = data.join(data_5m_resampled[['Open_5m', 'High_5m', 'Low_5m', 'SMA10_5m', 'EMA10_5m']], how='left') \
                          .join(data_15m_resampled[['Open_15m', 'High_15m', 'Low_15m', 'SMA10_15m', 'EMA10_15m']], how='left')
                data = data.dropna()
                X = data[['Open', 'High', 'Low', 'SMA10', 'EMA10', 'Open_5m', 'High_5m', 'Low_5m', 'Open_15m', 'High_15m']].values
                logger.info(f"X shape before retraining: {X.shape}")
                if X.shape[1] != 10:
                    raise ValueError(f"Expected 10 features for 30m, got {X.shape[1]}: {data.columns.tolist()}")
            else:
                X = data[['Open', 'High', 'Low', 'SMA10', 'EMA10']].values
                logger.info(f"X shape before training for {interval}: {X.shape}")
            for name in models[interval]:
                logger.info(f"Training {name} model for interval {interval}")
                model, scaler, y_scaler, train_time = train_model(models[interval][name], X[:-1], data['Target'].values[:-1])
                models[interval][name] = model
                scalers[interval][name] = scaler
                y_scalers[interval][name] = y_scaler
        trained = True
        logger.info("Retraining completed successfully")
        return {"status": "Retraining completed", "trained": trained}
    except Exception as e:
        logger.error(f"Retraining failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Retraining failed: {str(e)}")

@app.get("/api/models")
async def get_models():
    try:
        logger.info("Fetching model details")
        model_details = {}
        for interval in models:
            model_details[interval] = {
                'models': list(models[interval].keys()),
                'input_size': next(iter(models[interval].values())).lstm.input_size if 'LSTM' in models[interval] else next(iter(models[interval].values())).gru.input_size
            }
        return model_details
    except Exception as e:
        logger.error(f"Failed to fetch model details: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch model details: {str(e)}")

@app.post("/api/notify/email")
async def notify_email():
    global trained
    try:
        if not trained:
            raise ValueError("Models not trained, please call /api/train first")
        current_time = time.time()
        if not last_email_time.get('manual') or (current_time - last_email_time.get('manual', 0) >= 300):
            for interval in ['5m']:
                data = fetch_data(interval, 10)
                X = data[['Open', 'High', 'Low', 'SMA10', 'EMA10']].values
                predictions = []
                for name, model in models[interval].items():
                    last_input = X[-1].reshape(1, -1)
                    last_input_scaled = scalers[interval][name].transform(last_input)
                    last_input_tensor = torch.tensor(last_input_scaled).float().unsqueeze(1)
                    pred_normalized = model(last_input_tensor).item()
                    pred = y_scalers[interval][name].inverse_transform([[pred_normalized]])[0][0]
                    predictions.append({'name': name, 'rate': pred})
                last_close = data['Close'].iloc[-1].item()
                pred_rates = ', '.join([f"{p['name']}: {p['rate']:.5f}" for p in predictions])
                gmail_user = os.getenv('GMAIL_USER')
                gmail_pass = os.getenv('GMAIL_PASS')
                if gmail_user and gmail_pass:
                    msg = MIMEText(f"Manual FOREX Signal for {interval}: Predictions - {pred_rates}, Last Close: {last_close:.5f}")
                    msg['Subject'] = 'Manual AI Forex Signal'
                    msg['From'] = gmail_user
                    msg['To'] = 'mironedv@abv.bg'
                    with smtplib.SMTP('smtp.gmail.com', 587) as server:
                        server.starttls()
                        server.login(gmail_user, gmail_pass)
                        server.send_message(msg)
                    logger.info("Manual email sent successfully to mironedv@abv.bg")
                    last_email_time['manual'] = current_time
                else:
                    logger.error("GMAIL_USER or GMAIL_PASS not set in environment")
                    raise ValueError("Email credentials not configured")
            return {"status": "Email notification sent"}
        else:
            logger.warning("Email cooldown active, please wait")
            raise HTTPException(status_code=429, detail="Email cooldown active, please wait 5 minutes")
    except Exception as e:
        logger.error(f"Email notification failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Email notification failed: {str(e)}")

@app.post("/api/notify/whatsapp")
async def notify_whatsapp():
    global trained
    try:
        if not trained:
            raise ValueError("Models not trained, please call /api/train first")
        current_time = time.time()
        if not last_email_time.get('whatsapp') or (current_time - last_email_time.get('whatsapp', 0) >= 300):
            for interval in ['5m']:
                data = fetch_data(interval, 10)
                X = data[['Open', 'High', 'Low', 'SMA10', 'EMA10']].values
                predictions = []
                for name, model in models[interval].items():
                    last_input = X[-1].reshape(1, -1)
                    last_input_scaled = scalers[interval][name].transform(last_input)
                    last_input_tensor = torch.tensor(last_input_scaled).float().unsqueeze(1)
                    pred_normalized = model(last_input_tensor).item()
                    pred = y_scalers[interval][name].inverse_transform([[pred_normalized]])[0][0]
                    predictions.append({'name': name, 'rate': pred})
                last_close = data['Close'].iloc[-1].item()
                pred_rates = ', '.join([f"{p['name']}: {p['rate']:.5f}" for p in predictions])
                account_sid = os.getenv('TWILIO_ACCOUNT_SID')
                auth_token = os.getenv('TWILIO_AUTH_TOKEN')
                client = Client(account_sid, auth_token)
                message = client.messages.create(
                    body=f"Manual FOREX Signal for {interval}: Predictions - {pred_rates}, Last Close: {last_close:.5f}",
                    from_='whatsapp:' + os.getenv('TWILIO_PHONE_NUMBER'),
                    to='whatsapp:' + os.getenv('WHATSAPP_RECIPIENT')
                )
                logger.info(f"WhatsApp message sent, SID: {message.sid}")
                last_email_time['whatsapp'] = current_time
            return {"status": "WhatsApp notification sent"}
        else:
            logger.warning("WhatsApp cooldown active, please wait")
            raise HTTPException(status_code=429, detail="WhatsApp cooldown active, please wait 5 minutes")
    except Exception as e:
        logger.error(f"WhatsApp notification failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"WhatsApp notification failed: {str(e)}")

def run_scheduler():
    def job():
        logger.info("Scheduler job triggered")
        asyncio.run(get_signal(interval="5m"))
    
    schedule.every(5).minutes.do(job)
    while True:
        schedule.run_pending()
        time.sleep(60)

if __name__ == "__main__":
    scheduler_thread = Thread(target=run_scheduler, daemon=True)
    scheduler_thread.start()
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
