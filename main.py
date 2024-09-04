import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
from keras.models import load_model
import matplotlib.pyplot as plt

# Load the trained LSTM model
model = load_model('stock_price_prediction_model.h5')

# Title and description
st.title('Stock Price Prediction')
st.write('This app predicts the future stock price of a given stock based on historical data.')

# Sidebar for stock selection and date range input
st.sidebar.header('User Input Parameters')
stock_symbol = st.sidebar.text_input('Stock Ticker', 'NFLX')
start_date = st.sidebar.date_input('Start Date', value=pd.to_datetime('2010-01-01'))
end_date = st.sidebar.date_input('End Date', value=pd.to_datetime('2023-08-01'))

# Fetch historical stock data from yfinance
st.write(f'Fetching data for {stock_symbol} from {start_date} to {end_date}...')
stock_data = yf.download(stock_symbol, start=start_date, end=end_date)

# Show the historical stock data
st.subheader('Historical Stock Data')
st.write(stock_data.tail())

# Plot the closing price history
#st.subheader('Closing Price History')
#plt.figure(figsize=(14, 5))
#plt.plot(stock_data['Close'], color='blue', label=f'{stock_symbol} Closing Price')
#plt.title(f'{stock_symbol} Closing Price History')
#plt.xlabel('Date')
#plt.ylabel('Closing Price')
#plt.legend()
#st.pyplot(plt)

# Data Preprocessing
close_prices = stock_data['Close'].values.reshape(-1, 1)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(close_prices)

# Prepare the dataset for prediction
time_step = 60

def create_dataset(dataset, time_step=60):
    X = []
    for i in range(len(dataset) - time_step):
        X.append(dataset[i:(i + time_step), 0])
    return np.array(X)

X_input = create_dataset(scaled_data, time_step)
X_input = X_input.reshape(X_input.shape[0], X_input.shape[1], 1)

# Making Predictions
st.subheader('Stock Price Prediction')
predicted_prices = model.predict(X_input)
predicted_prices = scaler.inverse_transform(predicted_prices)

# Plot the predictions
st.subheader('Predicted Stock Prices')
plt.figure(figsize=(14, 5))
plt.plot(close_prices[time_step:], color='blue', label='Actual Stock Price')
plt.plot(predicted_prices, color='red', label='Predicted Stock Price')
plt.title(f'{stock_symbol} Stock Price Prediction')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()
st.pyplot(plt)

# Conclusion
st.write('This model predicts stock prices based on historical data using an LSTM model. The red line represents the predicted prices, while the blue line represents the actual prices.')
