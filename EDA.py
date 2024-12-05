import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from transformers import pipeline

def corr_matrix(df_numeric):
    # plt.figure(figsize=(20, 16))
    # sns.heatmap(df_numeric.corr(), annot=True)
    fig_corr = px.imshow(df_numeric.corr(), text_auto=True, aspect="auto",
                        title="Correlation Heatmap of Numeric Variables")
    st.plotly_chart(fig_corr)
    st.write(" In the correlation matrix we can see that there is high correlation between the features from the stock data. It is obvious because, the difference between these values in a day \
             will be minimal as compared to the data collected over a few years. Example, the open on the 1st of Jan could be 20 and the close could be 18 but in the month of November, the stock could have increased such that the ope could be 40 and the close as 45.\
             The other highly correlated are the sentiment data features.")


def PCA_visual(df_numeric):
    scaling=StandardScaler()

    # Use fit and transform method
    scaling.fit(df_numeric)
    Scaled_data=scaling.transform(df_numeric)

    # Set the n_components=3
    principal=PCA(n_components=3)
    principal.fit(Scaled_data)
    x=principal.transform(Scaled_data)

    # Streamlit application
    st.title("3D Scatter Plot of PCA")

    # Create the figure and plot
    fig = plt.figure(figsize=(10, 10))

    # Choose projection 3d for creating a 3d graph
    axis = fig.add_subplot(111, projection='3d')

    # x[:,0] is PC1, x[:,1] is PC2, and x[:,2] is PC3
    axis.scatter(x[:, 0], x[:, 1], x[:, 2], c=df_numeric['ma5'], cmap="Accent")

    # Set labels for the axes
    axis.set_xlabel("PC1", fontsize=10)
    axis.set_ylabel("PC2", fontsize=10)
    axis.set_zlabel("PC3", fontsize=10)

    # Display the plot in Streamlit
    st.pyplot(fig)
    
    principal_components = principal.components_

    # Step 3: Create a DataFrame for the contributions of each stock
    contributions = pd.DataFrame(principal_components.T, 
                                index=df_numeric.columns, 
                                columns=[f'PC{i+1}' for i in range(3)])

    # Step 4: Plot the contributions
    fig, ax = plt.subplots(figsize=(10, 6))
    contributions.plot(kind='bar', ax=ax)
    ax.set_title('Contributions of Each Stock to Top-3 Principal Components')
    ax.set_ylabel('Contribution')
    ax.set_xlabel('Stocks')
    ax.legend(title='Principal Components')
    plt.tight_layout()
    # Display the plot in Streamlit
    st.pyplot(fig)
    st.write(" The above graph shows the contribution of each feature towards the pricipal components")


    # Step 5: Explained Variance (Optional: to show the importance of each component)
    explained_variance = principal.explained_variance_ratio_
    st.write(f'Explained Variance by PC1: {explained_variance[0]:.2f}')
    st.write(f'Explained Variance by PC2: {explained_variance[1]:.2f}')
    st.write(f'Explained Variance by PC3: {explained_variance[2]:.2f}')




def times_series_plot(df):
    st.write("select the y variable you are interested in")
    st.title("Time series plots")
    y_variable = st.selectbox("Select Y Variable", df.columns[1:], index=0)

    fig1 = px.line(df, x=df.index, y=y_variable, color_discrete_sequence=["#0514C0"], labels={'y': 'Stock'})
    # fig.add_scatter(x=df['ds'], y=prediction['y'], mode='lines', name='Prediction', line=dict(color='#4CC005'))
    fig1.update_layout(title='Stock trend with the 5 day moving average', xaxis_title='Date', yaxis_title='Value')
    st.plotly_chart(fig1, use_container_width=True)

    fig2 = px.line(df, x = df.index, y='volatility')
    fig2.update_layout(title='Volatility trend', xaxis_title='Date', yaxis_title='Volatility')
    st.plotly_chart(fig2, use_container_width=True)

    fig3 = make_subplots(rows=2, cols=1,
                        vertical_spacing=0.03, row_heights=[0.7, 0.3])

    # Candlestick chart
    fig3.add_trace(go.Candlestick(x=df.index,
                                open=df['open'],
                                high=df['high'],
                                low=df['low'],
                                close=df['close'],
                                name='OHLC'), row=1, col=1)

    # Volume bar chart
    fig3.add_trace(go.Bar(x=df.index, y=df['volume'], name='Volume'),
                row=2, col=1)

    fig3.update_layout(title='Price and Volume Analysis',
                    yaxis_title='Price',
                    yaxis2_title='Volume')
    st.plotly_chart(fig3, use_container_width=True)



    fig4 = go.Figure(data=[go.Candlestick(x=df.index,
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'])])

    fig4.add_trace(go.Bar(x=df.index, y=df['volume'], name='Volume', yaxis='y2'))

    fig4.add_trace(go.Scatter(x=df.index, y=df['mean_sentiment_ma3'],
                            mode='lines', name='Sentiment (MA3)', yaxis='y3'))

    fig4.update_layout(
        yaxis=dict(title='Price'),
        yaxis2=dict(title='Volume', overlaying='y', side='right'),
        yaxis3=dict(title='Sentiment', overlaying='y', side='right', position=0.95)
    )

    st.plotly_chart(fig4, use_container_width=True)

    mean = df['log_return'].mean()
    std = df['log_return'].std()
    ouliers_removed = df[(df['log_return'] > mean - 2 * std) & (df['log_return'] < mean + 2 * std)]
    st.write("outliers are removed based on the log_return values:")
    st.write(ouliers_removed.head())
    st.write("The plot of the daily returns traces a normal distribution indicating that the central limit theorem")
    fig = px.histogram(ouliers_removed, x="daily_return", y="close" )
    st.plotly_chart(fig)

