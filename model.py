from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt
# import mplfinace as mpl
import numpy as np
import pandas as pd
from statsmodels.tsa.ar_model import AutoReg
from datetime import timedelta

def hypothesis_testing(df):
    from scipy import stats
    result_dict = {}
    result = stats.ttest_1samp(df['daily_return'], 0)
    result_dict['AAPL'] = result
    result_dict


def model(df):
    df_close = df['close']
    df_close.plot(kind='kde')
    result = seasonal_decompose(df_close, model='multiplicative',period=30, extrapolate_trend=30)
    fig = plt.figure()
    fig = result.plot()
    st.pyplot(fig.set_size_inches(16, 9))
    df_log = np.log(df_close)
    train_data, test_data = df_log[3:int(len(df_log)*0.9)], df_log[int(len(df_log)*0.9):]

    model_autoARIMA = auto_arima(train_data, start_p=0, start_q=0,
                      test='adf',       # use adftest to find optimal 'd'
                      max_p=3, max_q=3, # maximum p and q
                      m=1,              # frequency of series
                      d=None,           # let model determine 'd'
                      seasonal=False,   # No Seasonality
                      start_P=0,
                      D=0,
                      trace=True,
                      error_action='ignore',
                      suppress_warnings=True,
                      stepwise=True)
    print(model_autoARIMA.summary())
    model_autoARIMA.plot_diagnostics(figsize=(15,8))
    plt.show()

def yule_walker_prediction(df):
    """
    Predict future stock prices using Yule-Walker equations.

    Parameters:
        stock_prices (list or array): Historical stock prices.
        lag (int): The order of the autoregressive model (AR lag).
        steps (int): Number of future steps to predict.

    Returns:
        list: Predicted stock prices.
    """
    stock_prices = df['close']

    # Parameters
    lag = 100   # AR(3) model
    steps = 100 #changed this to 4

    # Convert stock prices to a NumPy array
    stock_prices = np.asarray(stock_prices)

    # Fit an autoregressive model using Yule-Walker equations
    model = AutoReg(stock_prices, lags=lag).fit()

    # Generate future predictions
    future_predictions = model.predict(start=len(stock_prices), end=len(stock_prices) + steps - 1)

    predictions = future_predictions.tolist()
    fig, ax = plt.subplots()
    plt.plot(df.index, df['close'])
    #changed the range value to prediction_steps to ensure the x and y dimension match.
    x = [df.index.iloc[-1] + timedelta(days=i) for i in range(1, steps + 1)]
    plt.plot(x,  predictions)



def sarimax(df):
  # If needed, difference the closing price for stationarity
    df['Close_Diff'] = df['close'].diff().dropna()

    # Handle missing sentiment scores or scale them
    df['Sentiment_Score'] = df['mean_sentiment_raw'].fillna(0)
    train_size = int(len(df) * 0.8)
    train, test = df.iloc[:train_size], df.iloc[train_size:]
    exog_train = train[['open', 'mean_sentiment_raw', 'day_of_week']]
    exog_test = test[['open', 'mean_sentiment_raw', 'day_of_week']]

    # Define the ARIMA order (p, d, q)
    p, d, q = 1, 1, 1

    # Fit ARIMAX Model
    model = SARIMAX(train['close'], exog=exog_train, order=(p, d, q), enforce_stationarity=False, enforce_invertibility=False)
    results = model.fit(disp=False)

    st.write(results.summary())

    # Predict
    forecast = results.get_forecast(steps=len(test), exog=exog_test)

    # Extract forecasted mean
    predicted_mean = forecast.predicted_mean

    # Confidence intervals
    conf_int = forecast.conf_int()

    # Plot Results
    st.title("Stock Price Prediction with ARIMAX")

    # Plot Results
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(train['close'], label='Train')
    ax.plot(test['close'], label='Test')
    ax.plot(predicted_mean, label='Predicted', color='red')
    ax.fill_between(conf_int.index, conf_int.iloc[:, 0], conf_int.iloc[:, 1], color='pink', alpha=0.3)
    ax.legend()

