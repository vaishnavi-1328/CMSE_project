import streamlit as st
from bokeh.plotting import figure
from matplotlib import pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
def PCA(data):
    df_pca = df.copy()
    df_numeric = df_pca.select_dtypes(include=[np.number])
    df_numeric.drop(columns=['day_of_week','month','is_month_end'], inplace=True)
    u,v,w = np.linalg.svd(df_numeric)
    print(u)
    print(v)
    print(w)
    figure = plt.figure(figsize=(20, 16))
    sns.heatmap(df_numeric.corr(), annot=True)

    st.bokeh_chart(figure, use_container_width=False)


