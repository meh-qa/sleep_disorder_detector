import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json

# Load Model
with open('best_model_rf.pkl', 'rb') as file_1:
  rf_best_model = pickle.load(file_1)
  
with open('best_model_logreg.pkl', 'rb') as file_2:
  logreg_best_model = pickle.load(file_2)

with open('model_scaler.pkl', 'rb') as file_3:
  model_scaler = pickle.load(file_3)

with open('model_encoder.pkl','rb') as file_4:
  model_encoder = pickle.load(file_4)

with open('list_num_cols.txt', 'r') as file_5:
  list_num_cols = json.load(file_5)

with open('list_cat_cols.txt', 'r') as file_6:
  list_cat_cols = json.load(file_6)

def run():
    # Create Form Input
    with st.form('key=form_sleep_disorder'):
        gender = st.radio('Gender', ('Male', 'Female'))
        age = st.number_input('Age', min_value=0, max_value=200, value=25)
        occupation = st.radio('Occupation', ('Doctor', 'Nurse', 'Software Engineer', 'Sales Representative', 'Teacher', 'Engineer', 'Scientist', 'Accountant', 'Lawyer', 'Salesperson'))
        sleep_duration = st.number_input('Sleep Duration', min_value=0, max_value=24, value=0)
        qos = st.number_input('Quality of Sleep', min_value=0, max_value=10, value=0)
        pal = st.number_input('Physical Activity Level', min_value=0, max_value=100, value=0)
        stress_level = st.number_input('Stress Level', min_value=0, max_value=10, value=0)
        bmi_cat = st.radio('BMI Category', ('Normal', 'Overweight', 'Obese'))
        blood_pressure = st.text_input('Blood Pressure',value='0/0')
        heart_rate = st.number_input('Heart Rate', min_value=0, max_value=100, value=0)
        daily_steps = st.number_input('Daily Steps', min_value=0, max_value=99999, value=0)

        submitted = st.form_submit_button('Predict')

    data_inf = {
        "Gender":gender,
        "Age":age,
        "Occupation":occupation,
        "Sleep Duration":sleep_duration,
        "Quality of Sleep":qos,
        "Physical Activity Level":pal,
        "Stress Level":stress_level,
        "BMI Category":bmi_cat,
        "Blood Pressure":blood_pressure,
        "Heart Rate":heart_rate,
        "Daily Steps":daily_steps
    }


    data_inf = pd.DataFrame([data_inf])
    st.dataframe(data_inf)

    if submitted:
        # Split between Numerical Columns and Categorical Columns
        data_inf_num = data_inf[list_num_cols]
        data_inf_cat = data_inf[list_cat_cols]

        # Feature Scaling and Feature Encoding
        data_inf_num_scaled = model_scaler.transform(data_inf_num)
        data_inf_cat_encoded = model_encoder.transform(data_inf_cat)
        data_inf_final = np.concatenate([data_inf_num_scaled, data_inf_cat_encoded], axis=1)

        # Predict using Linear Regression
        rf_y_pred_inf = rf_best_model.predict(data_inf_final)
        gb_y_pred_inf = logreg_best_model.predict(data_inf_final)

        st.write('# Prediksi Sleep Disorder Random Forest: ', str(rf_y_pred_inf))
        st.write('# Prediksi Logistic Regression: ', str(gb_y_pred_inf))

    
if __name__ == '__main__':
    run()