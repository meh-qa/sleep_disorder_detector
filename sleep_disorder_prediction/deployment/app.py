import streamlit as st
import eda
import prediction

navigation = st.sidebar.selectbox('Pilih Halaman : ', ('EDA', 'Sleep Disorder Predict'))

if navigation == 'EDA':
    eda.run()
elif navigation == 'Sleep Disorder Predict':
    prediction.run()