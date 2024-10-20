import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import warnings
from sklearn.linear_model import BayesianRidge, LinearRegression
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.experimental import enable_iterative_imputer
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

from sklearn.preprocessing import LabelEncoder

from sklearn.compose import ColumnTransformer

warnings.filterwarnings("ignore")

# Train the logistic regression model


nltk.download('vader_lexicon')



def intro(data,df_tweet,df_cleaned,numerical_cols,df_imputed,df_combined):

    st.header("US Stock data analysis")
    st.sidebar.header("Please select the demo")

    st.markdown(
        """
        Hi all, here is a sumary of my project and the aim for this project:
        
        Brief Introduction to The Stock Market¶
        The stock market is one of the most fundamental aspects in global financial 
        systems where investors and brokers are highly concerned with how certain 
        companies and assets flow in respect to time. This assignment gives us an approach to compare 
        various stocks and assets and provide detailed findings so that investors can adjust their buying 
        and selling strategies accordingly to maximize profit and minimize risk of loss.""")

    st.subheader("Stocks and Assets")

    st.markdown("""Stocks

        We categorize stocks as shares of publicly-traded companies. They represent some form of ownership of the company. Buying shares of that company makes you one of the owners partially. Stock prices can fluctuate based on corporate related factors.""")

    st.subheader("Assets/Commodities")

    st.markdown("""Assets, or also known as commodities, hold value that represents some form of wealth that is tradeable as well. Investors alternatively trade assets to diversify their portfolios. Our data set includes assets like gold, platinum, as well as cryptocurrencies like bitcoin. Note that assets generally have higher volatility and are prone to sudden and massive fluctuations, increasing risks altogether.

        Research Objectives
        Optimize stock/asset selection for investment purposes
        Utilize visualization techniques to compare and assess leading stocks
        Analyze interrelationships among various stock options
    """
    )
    st.subheader("Companies dataset:")
    st.write(data.head())
    st.subheader("Tweets dataset:")
    st.write(df_tweet.head())
    st.subheader("numerical columns:")
    st.write(numerical_cols)
    st.subheader('After handing null values, data cleaning, the dataset is combined as shown below:')
    st.markdown("""
                1. Null values were handled using MICE imputation
                2. Data cleaning steps:
                    - Some numerical columns had . and , which had to be reomoved to convert from str and object to float datatype. 
                    - Both datasets had columns of Date which had different formats like - and / which had to be reformated to convert to datetime format. 
                    - Dropped duplicates.
                3. Data integration was done by transforming the stock dataset by convertinig the columns into rows with the comany names and stock value as the columns, To this dataset,tweet dataset is merged over company and date columns. """)
    st.write(df_combined.head())

def Visualization(df_imputed,df_encoded,df_scores):
    st.write("Have chosen microsoft, tesla, google and Apple for the visualization:")
    for i in df_encoded['companies'].unique():
        # Filter data for the current company
        company_data = df_encoded[df_encoded['companies'] == i]
        
        # Calculate the rolling average
        company_data['rolling_avg'] = company_data['values'].rolling(window=7).mean()
        
        # Create subplot with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Add original data trace
        fig.add_trace(
            go.Scatter(x=company_data['Date'], y=company_data['values'], name='Original'),
            secondary_y=False,
        )
        
        # Add rolling average trace
        fig.add_trace(
            go.Scatter(x=company_data['Date'], y=company_data['rolling_avg'], 
                    name=f'{7}-day Rolling Average', line=dict(color='red')),
            secondary_y=True,
        )
        
        # Update layout
        fig.update_layout(
            title=f'Time Series Chart for {i}',
            xaxis_title="Date",
            yaxis_title=i,
            legend_title="Legend",
            hovermode="x unified"
        )
        
        # Add range slider and buttons
        fig.update_xaxes(
            rangeslider_visible=True,
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="YTD", step="year", stepmode="todate"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(step="all")
                ])
            )
        )
        
        # Display the plot in Streamlit
        st.plotly_chart(fig, use_container_width=True)

    # sns.histplot(data=df_combined, x='values', kde=True, ax=ax[0])
    companies = df_combined['companies'].unique()
    fig, axes = plt.subplots(len(companies), 1, figsize=(12, 5*len(companies)))
    fig.tight_layout(pad=3.0)

    for count, company in enumerate(companies):
        company_data = df_combined[df_combined['companies'] == company]
        
        ax = axes[count] if len(companies) > 1 else axes
        
        sns.lineplot(data=company_data, x='Date', y='pos', ax=ax, label='Positive')
        sns.lineplot(data=company_data, x='Date', y='neg', ax=ax, label='negative')
        sns.lineplot(data=company_data, x='Date', y='values', ax=ax, label='Values')
        
        ax.set_title(f'Sentiment Analysis for {company}')
        ax.set_xlabel('Date')
        ax.set_ylabel('Sentiment Score')
        ax.legend()

    st.pyplot(fig)
    fig_sentiment = px.pie(df_combined, names=['Positive', 'Neutral', 'Negative'], 
                       values=[df_combined['pos'].mean(), df_combined['neu'].mean(), df_combined['neg'].mean()],
                       title='Overall Sentiment Distribution')
    st.plotly_chart(fig_sentiment)

    correlation = df_combined[['values', 'sent_score', 'pos', 'neu', 'neg']].corr()
    fig_corr = px.imshow(correlation, text_auto=True, aspect="auto",
                        title="Correlation Heatmap of Numeric Variables")
    st.plotly_chart(fig_corr)



def EDA_IDA(df_clean,df_std, df_outlier,df_imputed,df_encoded):
    st.subheader("Standardization")
    st.write("Before standardization:")
    st.write(df_clean.describe())
    st.write("After standardization:")
    st.write(df_std.describe())
    st.write("Before removing outlier")
    fig = px.scatter(x=df_std.index,y=df_std['Netflix_Vol.'])
    st.plotly_chart(fig, theme=None)
    st.write("After removing outlier")
    fig = px.scatter(x=df_outlier.index,y=df_outlier['Netflix_Vol.'])
    st.write(df_outlier.describe())
    st.plotly_chart(fig, theme=None)
    st.subheader("Missing values handling")
    st.write("Before imputation:")
    st.write(df_clean.isna().sum())
    MCAR_test(df_clean)
    st.write("After MICE imputation")
    st.write(df_imputed.isna().sum())
    st.subheader("Correlation")
    fig, ax = plt.subplots()
    correlation= df_clean[numerical_cols].corr()
    print(correlation)
    #print(correlation.sort_values(by=ascending=False))
    fig = px.imshow(correlation)
    st.plotly_chart(fig, theme=None)
    st.markdown("""good correlation :

    natural gas and crude oil
    natural gas and tesla
    crude oil and copper, tesla, google
    nasdaq is not particularly correlated to any stock
    apple with copper price, crude oil price, tesla , microsoft price, google and nvidea
    tesla with google, apple
    nvidea with apple, microsoft, meta and google""")
    st.subheader("Data encoding after combining:")
    st.write(df_encoded.head())
    ##outliers visualization show it
    st.write(df_encoded.describe())
    




def Modelling(df_combined):
    order=3
    poly = PolynomialFeatures(degree=order) 
    X=df_combined[['pos','neg','neu']]
    st.subheader('Independent features:')
    st.write(X.head())
    st.subheader('Dependent features:')
    y=df_combined['values']
    st.write(y.head())
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train_poly = poly.fit_transform(X_train) #fit both the training and test data to a polynomial function of degree n
    X_test_poly = poly.transform(X_test)

    # # 3. Initialize and fit the model (Linear Regression in this example)
    model = LinearRegression()
    model.fit(X_train_poly, y_train)

    # 4. Make predictions on the test set
    y_pred = model.predict(X_test_poly)

    # 5. Evaluate the model (e.g., using Mean Squared Error)
    mse = mean_squared_error(y_test, y_pred)
    st.write("Mean Squared Error:", mse)



def data_cleaning(data):
    data["Date"] = data["Date"].str.replace('/', '-')
    data["Date"] = pd.to_datetime(data["Date"], format='%d-%m-%Y')
    data=data.set_index(data['Date'])
    data=data.drop(columns=['Date'])
    numerical_columns=data.select_dtypes(include='number')
    cols=data[list(set(data.columns) - set(numerical_columns))].columns
    for i in cols:
        data[i] = data[i].str.replace(',', '')  # Remove commas
        data[i] = data[i].astype(float)         # Convert to float to preserve decimals

    data.drop_duplicates()
    return data

def outlier_removal(df):

    float_cols = df.select_dtypes(include='number').columns
    df_filtered = df[df[float_cols].apply(lambda x: x.between(-3, 3)).all(axis=1)]
    return df_filtered.dropna()

def imputation(df):
    df_numeric = df.reset_index(drop=True)  
    imputer = IterativeImputer(estimator=BayesianRidge(), n_nearest_features=None, 
                            imputation_order='ascending', random_state=42)
    numerical_cols = df.select_dtypes(include='number').columns
    imputed_data = imputer.fit_transform(df[numerical_cols])
    df_imputed = pd.DataFrame(imputed_data, columns=numerical_cols, index=df.index)

    return df_imputed




def MCAR_test(data):
    def littles_test(data):
        # Dropping rows with no missing data to focus on rows with missing values
        data_complete = data.dropna()
        
        # Calculating chi-squared statistic for Little's test
        # The null hypothesis for Little's test: Data is missing completely at random (MCAR)
        n_obs = data.shape[0]
        chi_square_stat = 0
        
        for column in data.columns:
            observed_freqs = data_complete.groupby(column).size()
            expected_freqs = observed_freqs.mean()
            chi_square_stat += stats.chisquare(observed_freqs, f_exp=expected_freqs)[0]
        
        # Calculate the degrees of freedom
        df = len(data.columns) - 1
        
        # Calculate p-value from chi-squared distribution
        p_value = 1 - stats.chi2.cdf(chi_square_stat, df)
        
        return chi_square_stat, p_value

    chi_square_stat, p_value = littles_test(data)

    print(f"Chi-Squared Statistic: {chi_square_stat}")
    print(f"P-Value: {p_value}")

    if p_value > 0.05:
        st.write(print("Fail to reject the null hypothesis: Data is missing completely at random (MCAR)"))
    else:
        st.write(print("Reject the null hypothesis: Data is not missing completely at random"))


def data_intergration(df,df_tweet):
    df_tweet_removed=df_tweet[df_tweet['Company Name'].isin(['Apple Inc.','Microsoft Corporation','Tesla, Inc.','Alphabet Inc.'])]
    companies_mapping={'Apple_Price':'Apple Inc.','Microsoft_Price':'Microsoft Corporation','Google_Price':'Alphabet Inc.','Tesla_Price':'Tesla, Inc.'}
    
    numerical_cols = df.select_dtypes(include='number').columns

    df1_reset = df.reset_index()
        # Step 2: Map the company names in df1 using the dictionary (if needed)
    df1_reset.rename(columns=companies_mapping, inplace=True)

        # Step 3: Melt df1 after mapping
    df1_melted = df1_reset.melt(id_vars='Date', var_name='companies', value_name='values')


        # Step 5: Clean 'Date' columns in both datasets (convert to str and ensure consistent format)
    df1_melted['Date'] = pd.to_datetime(df1_melted['Date']).dt.strftime('%Y-%m-%d')
    df_tweet_removed['Date'] = pd.to_datetime(df_tweet_removed['Date']).dt.strftime('%Y-%m-%d')

        # Step 6: Remove leading/trailing whitespaces and make company names lowercase for consistency
    df1_melted['companies'] = df1_melted['companies'].str.strip().str.lower()
    df_tweet_removed['Company Name'] = df_tweet_removed['Company Name'].str.strip().str.lower()

        # Step 7: Merge df1_melted with df2 on both 'Company name' and 'Date'
    df_combined = pd.merge(df_tweet_removed, df1_melted, left_on=['Company Name', 'Date'], right_on=['companies', 'Date'], how='inner')
    return df_combined

def sentiment_analyser(df_combined):
    df_combined["sent_score"] = ''
    df_combined["pos"] = ''
    df_combined["neu"] = ''
    df_combined["neg"] = ''
    sent_analyze = SentimentIntensityAnalyzer()
    for ind,row in df_combined.T.items():
        sent_sent = sent_analyze.polarity_scores(df_combined.loc[ind,'Tweet'])
        df_combined.at[ind,"sent_score"] = sent_sent["compound"]
        df_combined.at[ind,"pos"] = sent_sent["pos"]
        df_combined.at[ind,"neu"] = sent_sent["neu"]
        df_combined.at[ind,"neg"] = sent_sent["neg"]
    df_std=standarization(df_combined)
    return df_std

def data_encoding(df):
    le = LabelEncoder()
    df['companies_encoded'] = le.fit_transform(df['companies'])
    return df

def standarization(df):
    scaler = StandardScaler()
    numerical_columns=df.select_dtypes(include='number').columns
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_columns)
        ])
    transformed_data = preprocessor.fit_transform(df)
    standardized_df = pd.DataFrame(transformed_data, columns=numerical_columns)
    z=scaler.fit_transform(df.select_dtypes(include='number'))
    transformed_columns = numerical_columns
    z= pd.DataFrame(z, columns=transformed_columns)
    z.index=df.index
    return z


data = pd.read_csv('US_Stock_Data_vs.csv')
df_tweet = pd.read_csv('stock_tweets.csv')
df_cleaned = data.drop(columns=['Unnamed: 0'])
df_tweet['Date']=pd.to_datetime(df_tweet['Date']).dt.date
numerical_cols= data[data.select_dtypes(include='float').columns].columns

df_clean = data_cleaning(df_cleaned)
df_std=standarization(df_clean)
df_outlier=outlier_removal(df_std)
df_imputed=imputation(df_outlier)
df_combined = data_intergration(df_imputed,df_tweet)
df_encoded = data_encoding(df_combined)
df_scores=sentiment_analyser(df_encoded)



functions=['Intro, Data collection and preparation','Visualizations','IDA and EDA','Modelling']
demo_name = st.sidebar.selectbox("Choose a demo", functions)
if(demo_name=='Intro, Data collection and preparation'):
    intro(data,df_tweet,df_cleaned,numerical_cols,df_imputed,df_combined)
elif(demo_name=='IDA and EDA'):
    EDA_IDA(df_clean,df_std, df_outlier,df_imputed,df_encoded)
elif(demo_name=='Visualizations'):
    Visualization(df_imputed,df_encoded,df_scores)
elif(demo_name=='Modelling'):
    Modelling(df_combined)