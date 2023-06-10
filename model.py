"""
This module trains a random forest regressor model on historical price data and makes predictions on the next day's BTC price.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from binance.client import Client

API_KEY = 'hvkXktWV7JwZYRZRIbodO7ZFoBnuAcCOceosOE0FTufksIvQafO2yPLcL3jdW7oP'
API_SECRET = '2z2tnZicu844s8YBdREGiT7OBRGDOxFFJlxbqOVQDTmO18vrkZLNkKlP9Vyog8PC'

client = Client(API_KEY, API_SECRET)
SYMBOL = 'BTCUSDT'
INTERVAL = Client.KLINE_INTERVAL_1DAY
LIMIT = 500
WINDOW_SIZE = 7

# Fetch live price data from the Binance API
klines = client.futures_klines(symbol=SYMBOL, interval=INTERVAL, limit=LIMIT)
data = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume',
                                     'close_time', 'quote_asset_volume', 'trades',
                                     'taker_buy_base', 'taker_buy_quote', 'ignored'])
data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
data.set_index('timestamp', inplace=True)
data['close'] = pd.to_numeric(data['close'])

# Create dataset for training model
data['target'] = data['close'].shift(-1)
data.dropna(inplace=True)
X = np.array([data['close'].iloc[i:i+WINDOW_SIZE].values for i in range(len(data) - WINDOW_SIZE)])
y = data.iloc[WINDOW_SIZE:]['target'].values

# Initialize and train a random forest regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# Fetch the latest price data for making predictions
latest_klines = client.futures_klines(symbol=SYMBOL, interval=INTERVAL, limit=WINDOW_SIZE)
latest_data = pd.DataFrame(latest_klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume',
                                                   'close_time', 'quote_asset_volume', 'trades',
                                                   'taker_buy_base', 'taker_buy_quote', 'ignored'])
latest_data['timestamp'] = pd.to_datetime(latest_data['timestamp'], unit='ms')
latest_data.set_index('timestamp', inplace=True)
latest_data['close'] = pd.to_numeric(latest_data['close'])

# Prepare input data for making predictions
input_data = np.array([latest_data['close'].values])

# Make predictions using the trained model
predictions = model.predict(input_data)

# Print the predicted next day's BTC price
predicted_price = predictions[0]
print(f'Predicted next day BTC price: {predicted_price:.2f}')
