import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from PIL import Image

st.set_page_config(
    page_title= 'Sleep Disorder - EDA',
    layout= 'wide',
    initial_sidebar_state= 'expanded'
)

def run():

    # Page Title
    st.title('Sleep Disorder EDA')

    # Sub Header
    st.subheader('EDA untuk Analisa Sleep Disorder')

    # Menambahkan Text
    st.write('This page is created by Mehdi')

    # Membuat Garis Lurus
    st.markdown('---')

    # Magic Syntax
    '''
    Page ini, merupakan explorasi sederhana
    Dataset yang digunakan adalah dataset Sleep Disorder
    Dataset ini berasal dari kaggle
    '''

    # Show Data Frame
    data = pd.read_csv('h8dsft_P1M2_mehdi.csv')
    st.dataframe(data)

    # Membuat Barplot
    st.write('#### Plot Age')
    fig = plt.figure(figsize=(15,5))
    sns.countplot(x='Age', data=data)
    st.pyplot(fig)

    # Membuat Histogram
    st.write('#### Histogram of Sleep Disorder')
    fig = plt.figure(figsize=(15,5))
    sns.histplot(data['Sleep Disorder'], bins=30, kde=True)
    st.pyplot(fig)

    # Membuat Histogram Berdasarkan Input User
    st.write('#### Histogram Berdasarkan Input User')
    pilihan = st.radio('Pilih kolom: ', ('Age', 'Gender', 'Sleep Duration', 'Stress Level'))
    fig = plt.figure(figsize=(15,5))
    sns.histplot(data[pilihan], bins=30, kde=True)
    st.pyplot(fig)


if __name__ == '__main__':
    run()