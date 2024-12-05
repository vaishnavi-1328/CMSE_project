import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import streamlit as st

class FinancialDataIntegrator:
    def __init__(self):
        self.price_data = None
        self.sentiment_data = None

    def preprocess_price_data(self, price_df):
        """
        Preprocess price data and calculate technical indicators
        """
        df = price_df.copy()

        # Calculate returns
        df['daily_return'] = df['close'].pct_change()
        df['log_return'] = np.log(df['close']) - np.log(df['close'].shift(1))

        # Add technical indicators
        df['volatility'] = df['daily_return'].rolling(window=20).std()
        df['ma5'] = df['close'].rolling(window=5).mean()
        df['ma20'] = df['close'].rolling(window=20).mean()
        df['rsi'] = self._calculate_rsi(df['close'])

        return df

    def preprocess_sentiment_data(self, sentiment_df):
        """
        Preprocess and aggregate sentiment data
        """
        df = sentiment_df.copy()
        # Convert time to datetime if not already
        df['time_published'] = pd.to_datetime(df['time_published'])

        # Set time as index
        df.set_index('time_published', inplace=True)

        # Resample to daily frequency and calculate various sentiment metrics
        daily_sentiment = df.resample('D').agg({
            'sentiment_score': [
                ('mean_sentiment', 'mean'),
                ('sentiment_std', 'std'),
                ('sentiment_count', 'count')
            ]
        }).droplevel(0, axis=1)
        # Fill missing values with forward fill, then backward fill
        daily_sentiment = daily_sentiment.fillna(method='ffill').fillna(method='bfill')
        
        return daily_sentiment

    def integrate_data(self, price_df, sentiment_df, sentiment_window=3):
        """
        Integrate price and sentiment data with rolling sentiment features

        Args:
            price_df: DataFrame with price data
            sentiment_df: DataFrame with sentiment data
            sentiment_window: Number of days for sentiment rolling features
        """
        # Preprocess both datasets
        price_processed = self.preprocess_price_data(price_df)
        sentiment_processed = self.preprocess_sentiment_data(sentiment_df)

        # Ensure both datasets have datetime index
        price_processed.index = pd.to_datetime(price_processed.index)

        # Create rolling sentiment features
        sentiment_features = pd.DataFrame()

        # Calculate rolling sentiment metrics
        for col in sentiment_processed.columns:
            sentiment_features[f'{col}_raw'] = sentiment_processed[col]
            sentiment_features[f'{col}_ma{sentiment_window}'] = (
                sentiment_processed[col].rolling(window=sentiment_window).mean()
            )
            if col == 'mean_sentiment':
                sentiment_features[f'{col}_momentum'] = (
                    sentiment_processed[col].diff()
                )

        # Merge price and sentiment data
        integrated_data = price_processed.join(sentiment_features, how='left')

        # Handle missing values
        integrated_data = self._handle_missing_values(integrated_data)

        # Add time-based features
        integrated_data = self._add_time_features(integrated_data)

        return integrated_data

    def _calculate_rsi(self, prices, period=14):
        """Calculate RSI technical indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _handle_missing_values(self, df):
        """Handle missing values in the integrated dataset"""
        # Forward fill sentiment features (carry forward last known sentiment)
        sentiment_cols = [col for col in df.columns if 'sentiment' in col]
        df[sentiment_cols] = df[sentiment_cols].fillna(method='ffill')

        # For any remaining NaN values, use backward fill
        df = df.fillna(method='bfill')

        return df

    def _add_time_features(self, df):
        """Add time-based features"""
        df['day_of_week'] = df.index.dayofweek
        df['month'] = df.index.month
        df['is_month_end'] = df.index.is_month_end.astype(int)

        return df

# Example usage
def main():
    # Example data loading (replace with your actual data)
    price_data = pd.read_csv('stock_data.csv', index_col=0, parse_dates=True)
    sentiment_data = pd.read_csv('sentiment_data.csv', parse_dates=['time_published'])

    # Create integrator instance
    integrator = FinancialDataIntegrator()

    # Integrate data
    integrated_data = integrator.integrate_data(price_data, sentiment_data)

    # Create feature sets for modeling
    features = integrated_data[[
        'daily_return', 'volatility', 'ma5', 'ma20', 'rsi',
        'mean_sentiment_raw', 'mean_sentiment_ma3', 'mean_sentiment_momentum',
        'sentiment_std_raw', 'sentiment_count_raw',
        'day_of_week', 'month', 'is_month_end'
    ]].copy()

    # Create target variable (next day's return)
    target = integrated_data['daily_return'].shift(-1)

    integrated_data.to_csv('integrated_data.csv')
    features.to_csv('features.csv')
    target.to_csv('target.csv')

if __name__ == "__main__":
    main()