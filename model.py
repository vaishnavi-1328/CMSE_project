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
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import traceback


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
    # stock_prices = df['close']

    # # Parameters
    # lag = 50   # AR(3) model
    # steps = 50 #changed this to 4

    # # Convert stock prices to a NumPy array
    # stock_prices = np.asarray(stock_prices)

    # # Fit an autoregressive model using Yule-Walker equations
    # model = AutoReg(stock_prices, lags=lag).fit()

    # # Generate future predictions
    # future_predictions = model.predict(start=len(stock_prices), end=len(stock_prices) + steps - 1)

    # predictions = future_predictions.tolist()
    # fig, ax = plt.subplots()
    # plt.plot(df.index, df['close'])
    # #changed the range value to prediction_steps to ensure the x and y dimension match.
    # x = [df.index.iloc[-1] + timedelta(days=i) for i in range(1, steps + 1)]
    # plt.plot(x,  predictions)

    from statsmodels.tsa.ar_model import AutoReg
    if 'date' in df.columns:  # Check if a 'date' column exists
        df['date'] = pd.to_datetime(df['date'])
    lag = st.slider("Select AR lag (order)", min_value=1, max_value=100, value=50)
    steps = st.slider("Select number of steps to predict", min_value=1, max_value=50, value=10)

        # Predict future stock prices
    stock_prices = np.asarray(df['close'])
    lag = min(lag, len(stock_prices) // 2)  # Ensure lag is appropriate for data size

        # Fit an autoregressive model
    model = AutoReg(stock_prices, lags=lag).fit()

        # Generate future predictions
    future_predictions = model.predict(start=len(stock_prices), end=len(stock_prices) + steps - 1)
    predictions = future_predictions.tolist()

        # Generate future dates for predictions
    st.write(df.head())
    future_dates = [df.index[-1] + timedelta(days=i) for i in range(1, steps + 1)]

        # Plot results
    fig, ax = plt.subplots()
    ax.plot(df.index, df['close'], label="Historical Prices")
    ax.plot(future_dates, predictions, label="Predicted Prices", linestyle="--")
    ax.set_xlabel("Date")
    ax.set_ylabel("Stock Price")
    ax.set_title("Stock Price Prediction")
    ax.legend()
    st.pyplot(fig)

        # Display prediction values
    st.subheader("Future Predictions")

    prediction_df = pd.DataFrame({
        "Date": future_dates,
        "Predicted Price": predictions
    })
# Assuming `df` contains historical stock prices and `prediction_df` contains predicted prices
    # Convert the index to datetime and set it
    prediction_df.index = pd.to_datetime(prediction_df['Date'])

    st.write("### Stock Price Prediction")

    # Display the prediction DataFrame
    st.write("Predicted Prices")
    st.write(prediction_df)

    # Input: Number of stocks
    st.write("How many stocks do you want to buy?")
    stocks = st.number_input("Number of stocks", min_value=1, max_value=200)

    # Input: Investment duration (in days)
    st.write("How long do you want to invest (in days)?")
    days = st.number_input("Investment duration (days)", min_value=1, max_value=50)

    # Calculate the predicted date - keeping it as datetime
    predicted_date = pd.to_datetime(df.index[-1]) + pd.Timedelta(days=days)

    # For comparison purposes, format the date to match your prediction_df index
    predicted_date = predicted_date.normalize()  # This removes time component

    # Debug prints
    st.write("Debug - Predicted date:", predicted_date)
    st.write("Debug - Available dates in prediction_df:", prediction_df.index)

    # Find the closest available date
    closest_date = prediction_df.index[prediction_df.index.get_indexer([predicted_date], method='nearest')[0]]

    # Find the predicted price for the selected date
    predicted_price = prediction_df.loc[closest_date, 'Predicted Price']
    current_price = df.iloc[-1]['close']  # Get the most recent closing price

    # Calculate profit or loss
    profit_or_loss = (predicted_price - current_price) * stocks

    # Display results
    st.write(f"Predicted price on {closest_date.date()}: ${predicted_price:.2f}")
    st.write(f"Current price: ${current_price:.2f}")

    if profit_or_loss > 0:
        st.success(f"You might make a profit of ${profit_or_loss:.2f}")
    elif profit_or_loss < 0:
        st.error(f"You might incur a loss of ${abs(profit_or_loss):.2f}")
    else:
        st.info("No profit, no loss.")


def sarimax(df):
    try:
        st.subheader("SARIMAX Model Analysis")
        
        # Create copy to avoid modifying original data
        df_model = df.copy()
        
        # Ensure data is sorted by date
        df_model = df_model.sort_index()
        
        # Calculate differenced series
        df_model['Close_Diff'] = df_model['close'].diff().dropna()
        
        # Handle missing sentiment scores
        df_model['Sentiment_Score'] = df_model['mean_sentiment_raw'].fillna(0)
        
        # Create train-test split
        train_size = int(len(df_model) * 0.8)
        train = df_model.iloc[:train_size]
        test = df_model.iloc[train_size:]
        
        # Display training and testing data info
        col1, col2 = st.columns(2)
        with col1:
            st.write("Training Data Sample")
            st.write(train.head())
        with col2:
            st.write("Testing Data Sample")
            st.write(test.head())
        
        # Prepare exogenous variables
        exog_columns = ['mean_sentiment_raw', 'day_of_week']
        exog_train = train[exog_columns]
        exog_test = test[exog_columns]
        
        # Define SARIMAX parameters
        p, d, q = 1, 1, 1  # ARIMA order
        P, D, Q, s = 1, 1, 1, 5  # Seasonal order
        
        # Create and fit SARIMAX model
        with st.spinner('Fitting SARIMAX model...'):
            model = SARIMAX(train['close'],
                          exog=exog_train,
                          order=(p, d, q),
                          seasonal_order=(P, D, Q, s),
                          enforce_stationarity=False,
                          enforce_invertibility=False)
            results = model.fit(disp=False)
        
        # Get predictions
        forecast = results.get_forecast(steps=len(test), exog=exog_test)
        predicted_mean = forecast.predicted_mean
        conf_int = forecast.conf_int()
        
        # Create figure with secondary axis
        fig = go.Figure()
        
        # Add traces for actual prices
        fig.add_trace(
            go.Scatter(
                x=train.index,
                y=train['close'],
                name='Training Data',
                line=dict(color='blue')
            )
        )
        
        fig.add_trace(
            go.Scatter(
                x=test.index,
                y=test['close'],
                name='Testing Data',
                line=dict(color='green')
            )
        )
        
        # Add prediction trace
        fig.add_trace(
            go.Scatter(
                x=test.index,
                y=predicted_mean,
                name='Predictions',
                line=dict(color='red', dash='dash')
            )
        )
        
        # Add confidence interval
        fig.add_trace(
            go.Scatter(
                x=test.index.tolist() + test.index.tolist()[::-1],
                y=conf_int.iloc[:, 0].tolist() + conf_int.iloc[:, 1].tolist()[::-1],
                fill='toself',
                fillcolor='rgba(255,0,0,0.1)',
                line=dict(color='rgba(255,0,0,0)'),
                name='95% Confidence Interval'
            )
        )
        
        # Update layout
        fig.update_layout(
            title='SARIMAX Model Predictions',
            xaxis_title='Date',
            yaxis_title='Stock Price',
            hovermode='x unified',
            showlegend=True,
            height=600,
            template='plotly_white'
        )
        
        # Display the plot
        st.plotly_chart(fig, use_container_width=True)
        
        # Calculate and display metrics
        mse = mean_squared_error(test['close'], predicted_mean)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(test['close'], predicted_mean)
        
        # Display metrics in columns
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Mean Squared Error", f"{mse:.2f}")
        with col2:
            st.metric("Root Mean Squared Error", f"{rmse:.2f}")
        with col3:
            st.metric("Mean Absolute Error", f"{mae:.2f}")
        
        # Display model summary in expander
        with st.expander("View Model Summary"):
            st.text(results.summary())
        
        # Optional: Add download buttons for predictions
        prediction_df = pd.DataFrame({
            'Actual': test['close'],
            'Predicted': predicted_mean,
            'Lower_CI': conf_int.iloc[:, 0],
            'Upper_CI': conf_int.iloc[:, 1]
        })
        
        st.download_button(
            label="Download Predictions",
            data=prediction_df.to_csv().encode('utf-8'),
            file_name='sarimax_predictions.csv',
            mime='text/csv'
        )
        
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.write("Error details:", traceback.format_exc())