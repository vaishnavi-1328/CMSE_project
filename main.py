import streamlit as st
import pandas as pd
import requests
import model
from Data_collection import AlphaVantageAPI
import json
import Data_collection
import Finance_data_integrator
import EDA
import requests
import pandas as pd
from datetime import datetime
import time
import numpy as np
import streamlit as st
import base64
import LSTM

def set_bg_hack_url():
    '''
    A function to unpack an image from url and set as bg.
    Returns
    -------
    The background.
    '''
        
    st.markdown(
         f"""
         <style>
         .stApp {{
             background: url("https://www.shutterstock.com/shutterstock/videos/3435572275/thumb/1.jpg?ip=x480");
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

# Sidebar for navigation
st.markdown("""
    <style>
    [data-testid="stSidebar"] {
        background-color: white;
    }
    [data-testid="stSidebar"] .css-1d391kg {
        color: white; /* Adjust text color for visibility */
    }
    </style>
    """, unsafe_allow_html=True)

# Sidebar con[nn]t
st.sidebar.title("Navigation")
page = st.sidebar.selectbox(
    "Go to", ["Home", "Data", "EDA, Visualization", "Model"])
# Route to the selected page
if page=='home':
    st.write("HELLO, WELCOME TO MY PROJECT")

elif page == "Data":
    # with st.spinner('Collecting data...'):
    #     obj = AlphaVantageAPI()
    #     stock_data, sentiment_data = obj.fetch_data()
            
    # if stock_data is not None and sentiment_data is not None:
    if(True):
        with st.spinner('Saving data...'):
        # stock_data.to_csv("stock_data1.csv")
        # sentiment_data.to_csv("sentiment_data1.csv")
            
            st.success('Data collected and saved successfully!')
            
        st.write("Stock Data:")
        stock_data=pd.read_csv('stock_data.csv')
        sentiment_data = pd.read_csv('sentiment_data.csv')
        st.write(stock_data.head())
        st.write("Sentiment Data:") 
        st.write(sentiment_data.head())
        st.write("Data cleaning steps:")
        st.markdown(f'<p style="font-size:16px;">{"1. Converted from Date time str to datetime type and extracted the date column. <br> 2. The dataset was available from 2005, Since we are \
                                                  intereted in data from 2020, have filtered only that required data. <br> 3. The dataset does not have any null values because it was clollected from API.  "}</p>', unsafe_allow_html=True)



        fin = Finance_data_integrator.main()

elif(page=="EDA, Visualization"):
    with st.spinner('Collecting data...'):
        stock_data=pd.read_csv('stock_data.csv')
        sentiment_data = pd.read_csv('sentiment_data.csv')
        fin = Finance_data_integrator.main()
        integrated_data = pd.read_csv('integrated_data.csv')
        df_numeric = integrated_data.select_dtypes(include=[np.number])
        EDA.corr_matrix(df_numeric)
        EDA.PCA_visual(df_numeric)
        EDA.times_series_plot(integrated_data)


elif(page=='Model'):
    integrated_data = pd.read_csv('integrated_data.csv')
    model.sarimax(integrated_data)      
    LSTM.main(integrated_data)


        
else:
    set_bg_hack_url()
    st.markdown(f'<h1 style="color:#edf0ee;font-size:40px;">{"STOCK MARKET ANALYSIS"}</h1>', unsafe_allow_html=True)
            
    st.markdown(f'<h3 style="color:#edf0ee;font-size:20px;">{"As and always, Stock market is a field of calculated gambling. The amusing fact about stock market is the fact that there are so many \
                                                             factors that play to make the stock prices increase or decrease. As data scientists, we hope to make this gambling more \
                                                             precise and less risky as possible. With the amount of data that we have till date, the trends in the \
                                                             market and the most important factors that impact the stock prices. In this project, we aim to forecast the NASDAQ prices.\
                                                              The NASDAQ data was collected \
                 from Alphavantage. This data is combined with the sentiment data that was collected from the same source. <br> \
                 Steps that was taken throughout the project <br>\
                1.Data collection through API <br> \
                S2. Data cleaning <br>\ 3. Data preprocessing <br> 4. IDA and EDA <br> 5. Modelling and Forecasting"}</h3>', unsafe_allow_html=True)



