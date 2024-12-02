import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

class MultiFeatureStockPredictor:
    def __init__(self, lookback=60):
        self.lookback = lookback
        self.model = None
        self.scalers = {}
    
    def scale_data(self, df):
        """
        Scale all features using MinMaxScaler
        """
        scaled_data = pd.DataFrame()
        
        # Scale each feature independently
        for column in df.columns:
            self.scalers[column] = MinMaxScaler()
            scaled_data[column] = self.scalers[column].fit_transform(df[column].values.reshape(-1, 1)).flatten()
            
        return scaled_data
    
    def create_sequences(self, data):
        """
        Create sequences for LSTM model
        """
        X, y = [], []
        
        for i in range(self.lookback, len(data)):
            X.append(data[i-self.lookback:i])
            y.append(data.iloc[i]['close'])  # Predicting close price
            
        return np.array(X), np.array(y)
    
    def build_model(self, input_shape):
        """
        Build and compile LSTM model
        """
        model = Sequential([
            LSTM(100, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(100, return_sequences=False),
            Dropout(0.2),
            Dense(10),
            Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        self.model = model
        return model
    
    def train_predict(self, df, train_split=0.8):
        """
        Complete training pipeline
        """
        
        # Scale data
        df = df[['close',"volume","log_return","volatility",'rsi',"sentiment_count_raw"]]
        scaled_data = self.scale_data(df)
        
        # Create sequences
        X, y = self.create_sequences(scaled_data)
        
        # Split data
        train_size = int(len(X) * train_split)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        # st.write("X_train shape:", X_train.shape)
        # st.write("X_test shape:", X_test.shape)
        # pca = PCA(n_components=1)
        # X_train = pca.fit_transform(X_train)
        # X_test = pca.transform(X_test)
        # Build and train model
        self.build_model(input_shape=(X.shape[1], X.shape[2]))
        
        # Train model
        history = self.model.fit(
            X_train, y_train,
            epochs=10,
            batch_size=32,
            validation_split=0.1,
            verbose=1
        )
        
        # Make predictions
        train_pred = self.model.predict(X_train)
        test_pred = self.model.predict(X_test)
        
        # Inverse transform predictions
        train_pred = self.scalers['close'].inverse_transform(train_pred)
        test_pred = self.scalers['close'].inverse_transform(test_pred)
        
        return train_pred, test_pred, history

# Example usage
def main(df):
    
    # Initialize and train predictor
    predictor = MultiFeatureStockPredictor(lookback=30)
    train_pred, test_pred, history = predictor.train_predict(df)

    train_pred = train_pred.flatten()  # Converts (n, 1) -> (n,)
    test_pred = test_pred.flatten()
    total_samples = len(df)
    n_train = len(train_pred)
    n_test = len(test_pred)
    
    st.write(f"Total samples in df: {total_samples}")
    st.write(f"Training predictions: {n_train}")
    st.write(f"Test predictions: {n_test}")
    
    # Calculate correct indices based on actual prediction lengths
    train_index = df.index[predictor.lookback:predictor.lookback + n_train]
    test_index = df.index[predictor.lookback + n_train:predictor.lookback + n_train + n_test]
    results_df = pd.DataFrame({'Actual': df['close']})  # Start with actual values
    
    # Add predictions only where we have them
    results_df.loc[train_index, 'Train Predictions'] = train_pred
    results_df.loc[test_index, 'Test Predictions'] = test_pred

    # Plot the results using Plotly
    fig = px.line(results_df, title="Stock Predictions vs Actual Values")
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Stock Value",
        legend_title="Legend"
    )

    # Add MSE details to Streamlit
    st.write(f"Training MSE: {history.history['loss'][-1]:.4f}")
    st.write(f"Validation MSE: {history.history['val_loss'][-1]:.4f}")

    # Display the Plotly chart in Streamlit
    st.plotly_chart(fig)