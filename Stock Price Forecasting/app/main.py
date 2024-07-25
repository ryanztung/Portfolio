import numpy as np
import pandas as pd
from flask import Flask, request, render_template
from keras.models import load_model
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import webbrowser
import joblib

import matplotlib
matplotlib.use('Agg')

# Initialize Flask application
app = Flask(__name__)

# Import LSTM
model = load_model('./models/functional_LSTM.keras')

# Import scalers
price_scaler = joblib.load('./models/price_scaler.pkl')
date_scaler = joblib.load('./models/date_scaler.pkl')

# Import dates vector
dates = np.load('./data/dates.npy')

# Import close price data
df = pd.read_csv('./data/stock_data.csv')
close_prices = df['Close'][30:].values

# Define function to obtain close price predictions
def predict_prices(days_ahead, close_prices, dates, date_scaler, model):
    # Loop through each day prior to the prediction date
    for day in range(days_ahead):
        # Extract sequential features
        lagged_prices = close_prices[-30:].reshape(1, 30, 1)

        # Extract rolling features
        rolling_mean_5 = close_prices[-5:].mean()
        rolling_mean_10 = close_prices[-10:].mean()
        rolling_mean_30 = close_prices[-30:].mean()

        # Increment dates vector
        dates = np.append(dates, dates[-1] + np.timedelta64(1,'D'))

        # Set current date to last item in dates vector
        current_date = dates[-1].astype('datetime64[D]').item()

        # Extract day, month, and year features
        day = current_date.day
        month = current_date.month
        year = current_date.year

        # Combine static features
        static_features = np.array([rolling_mean_5, rolling_mean_10, rolling_mean_30, day, month, year])

        # Scale day, month, and year features
        static_features[3:] = date_scaler.transform(static_features[3:].reshape(1, 3))
        static_features = static_features.reshape(1, 6)

        # Append prediction to close_prices vector
        close_prices = np.append(close_prices, model.predict([lagged_prices, static_features]))
    
    return close_prices, dates

# Define function to create price visualization
def plot_predictions(close_prices, dates, days_ahead):
    plt.rcParams['figure.dpi'] = 300
    plt.figure(figsize=(5, 3))
    plt.plot(dates[:-days_ahead], close_prices[:-days_ahead], label='Historical Observations', linewidth=0.5)
    plt.plot(dates[-days_ahead:], close_prices[-days_ahead:], label='Predictions', linewidth=0.5)
    plt.xlabel('Date', fontsize=8)
    plt.ylabel('Close Price', fontsize=8)
    plt.xticks(fontsize=6)
    plt.yticks(fontsize=6)
    plt.legend(fontsize=8, prop={'size': 6})
    plt.tight_layout()
    plt.savefig('app/static/prediction_plot.png')
    plt.close()

# Prediction route
@app.route('/', methods=['GET', 'POST'])
def predict():
    prediction = None
    date = None

    if request.method == 'POST':
        # Process input date
        date = np.datetime64(request.form['date'])
        days_ahead = (date - dates[-1].astype('datetime64[D]')).astype(int)

        # Retrieve predictions
        expanded_close_prices, expanded_dates = predict_prices(days_ahead, close_prices, dates, date_scaler, model)
        expanded_close_prices = price_scaler.inverse_transform(expanded_close_prices.reshape(-1, 1))
        prediction = f'{ expanded_close_prices[-1][0]:.2f}'
    
        # Generate prediction plot
        plot_predictions(expanded_close_prices, expanded_dates, days_ahead)

    return render_template('index.html', prediction=prediction, date=date)

if __name__ == '__main__':
    webbrowser.open_new('http://127.0.0.1:5000/')
    app.run(debug=True)


