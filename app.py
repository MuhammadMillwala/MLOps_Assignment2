import numpy as np
import pandas as pd
from binance.client import Client
from flask import Flask, request, render_template
from joblib import load

app = Flask(__name__)

# Load trained model from disk
MODEL_FILENAME = 'btc_model.joblib'
model = load(MODEL_FILENAME)

# Initialize Binance API client
API_KEY = 'hvkXktWV7JwZYRZRIbodO7ZFoBnuAcCOceosOE0FTufksIvQafO2yPLcL3jdW7oP'
API_SECRET = '2z2tnZicu844s8YBdREGiT7OBRGDOxFFJlxbqOVQDTmO18vrkZLNkKlP9Vyog8PC'
client = Client(API_KEY, API_SECRET)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get last 7 days of BTC price data from form
        last_7_days_data = [float(request.form[f'day_{i}']) for i in range(7)]
        last_7_days_data = np.array(last_7_days_data).reshape(1, -1)

        # Make prediction for next day's BTC price
        prediction = model.predict(last_7_days_data)[0]
        prediction = f'{prediction:.2f}'

        return render_template('index.html', prediction=prediction)

    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        SYMBOL = 'BTCUSDT'
        INTERVAL = Client.KLINE_INTERVAL_1DAY
        WINDOW_SIZE = 7

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

        return render_template('predict.html', predicted_price=predicted_price)

    return render_template('predict.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
