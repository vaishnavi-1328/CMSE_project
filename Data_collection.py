import requests
import json
import streamlit as st

# replace the "demo" apikey below with your own key from https://www.alphavantage.co/support/#api-key
url = 'https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers=AAPL&apikey=H8CHNTUNAGZKISTS'
r = requests.get(url)
data = r.json()
import requests
import pandas as pd
from datetime import datetime
import time

class AlphaVantageAPI:
    def __init__(self, api_key = "H8CHNTUNAGZKISTS"):
        """
        Initialize the Alpha Vantage API wrapper
        Args:
            api_key (str): Your Alpha Vantage API key
        """
        self.api_key = api_key
        self.base_url = "https://www.alphavantage.co/query"

    def get_stock_time_series(self, symbol, interval='daily', outputsize='full'):
        """
        Get historical stock data
        Args:
            symbol (str): Stock symbol (e.g., 'AAPL')
            interval (str): 'daily', 'weekly', or 'monthly'
            outputsize (str): 'full' for 5+ years of data, 'compact' for last 100 data points
        """
        function = f"TIME_SERIES_{interval.upper()}"

        params = {
            'function': function,
            'symbol': symbol,
            'apikey': self.api_key,
            'outputsize': outputsize
        }

        response = requests.get(self.base_url, params=params)
        data = response.json()
        # Handle API errors
        if "Error Message" in data:
            raise Exception(f"API Error: {data['Error Message']}")

        # Get the time series data
        time_series_key = f"Time Series ({interval.title()})"
        if time_series_key not in data:
            raise Exception("No time series data found in response")
        print(time_series_key)
        # Convert to DataFrame
        df = pd.DataFrame.from_dict(data[time_series_key], orient='index')
        st.write("Initial time series stock data:")
        st.write(df.head())

        # Clean column names
        df.columns = [col.split('. ')[1] for col in df.columns]

        # Convert to numeric
        for col in df.columns:
            df[col] = pd.to_numeric(df[col])

        # Sort index by date
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        return df

    def get_news_sentiment(self, symbol, topics='blockchain,earnings'):
        """
        Get news sentiment data
        Args:
            symbol (str): Stock symbol
            topics (str): Comma-separated topics to filter news
        """
        params = {
            'function': 'NEWS_SENTIMENT',
            'symbol': symbol,
            'topics': topics,
            'apikey': self.api_key
        }

        response = requests.get(self.base_url, params=params)
        data = response.json()

        if "Error Message" in data:
            raise Exception(f"API Error: {data['Error Message']}")

        if 'feed' not in data:
            return pd.DataFrame()

        # Extract relevant information from news items
        news_data = []
        for item in data['feed']:
            news_data.append({
                'time_published': item['time_published'],
                'title': item['title'],
                'sentiment_score': float(item['overall_sentiment_score']),
                'sentiment_label': item['overall_sentiment_label']
            })
        news= pd.DataFrame(news_data)
        st.write("initial news data from alphavantage")
        st.write(news.head())
        return news
    
    def fetch_data(self):
    # Replace with your API key
        try:
            # Get historical stock data
            stock_data = self.get_stock_time_series('NDAQ', interval='daily')
            print("\nStock Data:")
            print(stock_data.head())
            
            # Get news sentiment
            sentiment_data = self.get_news_sentiment('NDAQ')
            print("\nSentiment Data:")
            print(sentiment_data.tail())
            
            sentiment_data, stock_data = self.clean_data(sentiment_data, stock_data)
            
            return stock_data, sentiment_data
            
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            return None, None
    
        
    
    def clean_data(self, sentiment_data, stock_data):
        def filter_data_yrs(date):
            if date.year >= 2020:
                return True
            else:
                return False
        stock_data["Date"] = pd.to_datetime(stock_data["Unnamed: 0"], format='%Y-%m-%d')
        stock_data['yrs'] = (stock_data.apply(lambda x: filter_data_yrs(x["Date"]), axis=1))
        stock_data=stock_data[stock_data['yrs']==True]
        sentiment_data = pd.read_csv('sentiment_data.csv')
        stock_data = filter_data_yrs(stock_data)
        sentiment_data['time_published'] = pd.to_datetime(sentiment_data['time_published'])
        return sentiment_data, stock_data




