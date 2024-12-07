import streamlit as st
import pandas as pd
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
import Seasonality

def set_bg_hack_url(link):
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
             background: url("{link}");
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
    "Go to", ["Home","Project description", "Data", "EDA, Visualization", "Model"])
# Route to the selected page
if page=='Project description':
     set_bg_hack_url("https://static.vecteezy.com/system/resources/thumbnails/001/925/614/small_2x/business-graph-chart-of-stock-market-investment-on-blue-background-vector.jpg")
     st.markdown(f'<h5 style="color:#edf0ee;font-size:20px;">{"As and always, Stock market is a field of calculated gambling. The amusing fact about stock market is the fact that there are so many \
                                                             factors that play to make the stock prices increase or decrease. As data scientists, we hope to make this gambling more \
                                                             precise and less risky as possible. With the amount of data that we have till date, the trends in the \
                                                             market and the most important factors that impact the stock prices. In this project, we aim to forecast the NASDAQ prices.\
                                                              The NASDAQ data was collected \
                 from Alphavantage. This data is combined with the sentiment data that was collected from the same source. <br> \
                 Steps that was taken throughout the project <br>\
                1.Data collection through API <br> \
                2. Data cleaning <br> 3. Data preprocessing <br> 4. IDA and EDA <br> 5. Modelling and Forecasting"}</h5>', unsafe_allow_html=True)


    

elif page == "Data":
    # with st.spinner('Collecting data...'):
        # obj = AlphaVantageAPI()
        # stock_data, sentiment_data = obj.fetch_data()
            
    # if stock_data is not None and sentiment_data is not None:
    
    if(True):
        
        st.title("ALPHAVANTAGE")
        st.markdown(" The data was collected from a website called Alphavantage that streams live news and the stock values. Here is the link for the website: [link](%s), for this project, I used Nasdaq stock values." %  'https://www.alphavantage.co/')
        
        with st.spinner('Saving data...'):
        # stock_data.to_csv("stock_data1.csv")
        # sentiment_data.to_csv("sentiment_data1.csv")
            
            st.success('Data collected and saved successfully!')
            
        st.write("Stock Data:")
        st.write("The columns that get returned when we hit the time series API \
                 response = requests.get(self.base_url, params=params). In params, we set the function as TIME_SERIES ,interval as 'daily' and the symbol for the stock we are interested in. For NASDAQ, symbol is set to NDAQ.")
        stock_data=pd.read_csv('stock_data.csv')
        sentiment_data = pd.read_csv('sentiment_data.csv')
        st.write(stock_data.head())
        st.write("Sentiment Data:") 
        st.write("Similar to colection of the stock values data, the Alphavantage provides another function that we can use to collect the news data. For the same API, we can set the function as NEWS_SENTIMENT and hece returns the data columns as shown below. The documentation for the API can be [found here]( %s)" %'https://www.alphavantage.co/documentation/')
        
        st.write(sentiment_data.head())
        st.write("Data cleaning steps:")
        st.markdown(f'<p style="font-size:16px;">{"1. Converted from Date time str to datetime type and extracted the date column. <br> 2. The dataset was available from 2005, Since we are \
                                                  intereted in data from 2020, have filtered only that required data. <br> 3. The missing data was handled using fillna, method='ffill'\
                                                  <br> 4. Dropped unecssary columns like the time published, Unnamed: 0, set the index to date. The cleaned and integrated dataset is shown under EDA and Visualization page.  "}</p>', unsafe_allow_html=True)



        fin = Finance_data_integrator.main()

elif(page=="EDA, Visualization"):
    import seaborn as sns
    import matplotlib.pyplot as plt
    with st.spinner('Collecting data...'):
        
        stock_data=pd.read_csv('stock_data.csv')
        sentiment_data = pd.read_csv('sentiment_data.csv')

        fin = Finance_data_integrator.main()
        integrated_data = pd.read_csv('integrated_data.csv')
        integrated_data['date'] = pd.to_datetime(integrated_data['Unnamed: 0'])
        integrated_data=integrated_data.set_index('date')
        df2=integrated_data.copy()
        integrated_data = integrated_data[integrated_data.index.year > 2020]
        df_numeric = integrated_data.select_dtypes(include=[np.number])
        integrated_data.drop(['Unnamed: 0'],axis=1, inplace=True)
        st.title("Data processing, Feature engineering")
        st.markdown(f'<p> {"For the integration of both the datasets, the following steps have been performed:<br> 1. Converted string type date to datetime, converted datetime to date in sentiment data. <br> \
                           2. Handled missing values and merged both the datasets by setting the date as the index."} </p>',unsafe_allow_html=True)
        st.markdown(f'<p> {"For Feature engineering, the following steps have been performed:<br> 1. Preprocessed both the stock prices and the sentiment data for more information the same like the RSI, RSI=100 - (100/1+ gain_of_window/loss_of_window), moving average, etc. <br> 3. Added time features like the day of week, month, etc \
                           2. Handled missing values and merged both the datasets by setting the date as the index."} </p>',unsafe_allow_html=True)
        
        st.write(integrated_data.head())
        selected_column = st.selectbox("Select Column to Analyze", integrated_data.columns)
        analysis_type = st.radio("Select Analysis Type", ["Univariate", "Bivariate"])

        if analysis_type == "Univariate":
            st.subheader(f"Univariate Analysis of {selected_column}")
            if integrated_data[selected_column].dtype in ['float64', 'int64']:
                # Histogram
                fig, ax = plt.subplots(figsize=(4,4))
                sns.histplot(integrated_data[selected_column], kde=True, ax=ax, bins=30)
                ax.set_title(f"Histogram of {selected_column}")
                st.pyplot(fig)

                # Boxplot
                fig, ax = plt.subplots(figsize=(10,10))
                sns.boxplot(x=integrated_data[selected_column], ax=ax)
                ax.set_title(f"Boxplot of {selected_column}")
                st.pyplot(fig)
            else:
                # Bar plot for categorical data
                fig, ax = plt.subplots()
                integrated_data[selected_column].value_counts().plot(kind="bar", ax=ax)
                ax.set_title(f"Bar Plot of {selected_column}")
                st.pyplot(fig)

        # Bivariate Analysis
        elif analysis_type == "Bivariate":
            st.subheader(f"Bivariate Analysis of {selected_column} vs Other Columns")
            if integrated_data[selected_column].dtype in ['float64', 'int64']:
                # Scatter Plot
                other_column = st.selectbox("Select Column for Bivariate Analysis", integrated_data.select_dtypes(['float64', 'int64']).columns)
                fig, ax = plt.subplots(figsize=(10,10))
                sns.scatterplot(x=integrated_data[selected_column], y=integrated_data[other_column], ax=ax)
                ax.set_title(f"Scatter Plot: {selected_column} vs {other_column}")
                st.pyplot(fig)

            else:
                st.write(f"Bivariate analysis for {selected_column} is not supported for non-numeric data.")

        EDA.corr_matrix(df_numeric)
        EDA.PCA_visual(df_numeric)
        EDA.times_series_plot(integrated_data)
        Seasonality.main(integrated_data)


elif(page=='Model'):
    integrated_data = pd.read_csv('integrated_data.csv')
    integrated_data['date'] = pd.to_datetime(integrated_data['Unnamed: 0'])    
    integrated_data=integrated_data.set_index('date')
    integrated_data = integrated_data[integrated_data.index.year > 2020]
    df_numeric = integrated_data.select_dtypes(include=[np.number])
    integrated_data.drop(['Unnamed: 0'],axis=1, inplace=True)
    tab1, tab2, tab3 = st.tabs([
        "Yule walker",
        "LSTM",
        "SARIMAX"
    ])
    with tab1:
        model.yule_walker_prediction(integrated_data)
    with tab3:
        model.sarimax(integrated_data)   
    with tab2:
        df_numeric = integrated_data.select_dtypes(include=["number"])
        st.write("Disclaimer: This model takes a couple of minutes to load,")   
        st.title("LSTM model, with 10 epochs")
        LSTM.main(integrated_data)


        
else:
    set_bg_hack_url("https://t4.ftcdn.net/jpg/02/99/97/81/360_F_299978128_VrSxbrMdiTHh4Zx51SGnpMtI24SZs8MD.jpg")
    st.markdown(f'<h1 style="color:#edf0ee;font-size:60px;">{"STOCK MARKET ANALYSIS"}</h1>', unsafe_allow_html=True)
            
   