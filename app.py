import streamlit as st
import pandas as pd
import numpy as np
import joblib

df = pd.read_csv('laptop_prices.csv')

# Load model and feature columns
model = joblib.load('laptop_price_model.pkl')
features = joblib.load('model_features.pkl')

# Streamlit UI
st.title('Laptop Price Prediction')

# Collect user inputs
company = st.selectbox('Company', ['Apple', 'Dell', 'HP', 'Lenovo', 'Asus', 'Acer', 'MSI', 'Toshiba', 'Samsung', 'Huawei', 'LG', 'Chuwi', 'Vero', 'Microsoft', 'Google', 'Xiaomi', 'Fujitsu', 'Razer', 'Mediacom'])
type_name = st.selectbox('Type', df['TypeName'].drop_duplicates())
inches = st.number_input('Screen Size (Inches)', 10.0, 20.0, 15.6)
ram = st.selectbox('RAM (GB)', sorted(df['Ram'].unique()))
weight = st.number_input('Weight (kg)', 0.5, 5.0, 2.0)
cpu_company = st.selectbox('CPU Brand', df['CPU_company'].drop_duplicates())
cpu_freq = st.number_input('CPU Frequency (GHz)', 1.0, 4.0, 2.5)
storage = st.selectbox('Primary Storage (GB)', sorted(df['PrimaryStorage'].unique()))
storage_type = st.selectbox('Storage Type', df['PrimaryStorageType'].drop_duplicates())
gpu_company = st.selectbox('GPU Brand', df['GPU_company'].drop_duplicates())

# Prepare user input for prediction
input_dict = {
    'Company_' + company: 1,
    'TypeName_' + type_name: 1,
    'Inches': inches,
    'Ram': ram,
    'Weight': weight,
    'CPU_company_' + cpu_company: 1,
    'CPU_freq': cpu_freq,
    'PrimaryStorage': storage,
    'PrimaryStorageType_' + storage_type: 1,
    'GPU_company_' + gpu_company: 1
}
input_df = pd.DataFrame([0] * len(features), index=features).T
for key in input_dict:
    if key in input_df.columns:
        input_df[key] = input_dict[key]

# Predict and display price
if st.button('Predict Price'):
    pred = model.predict(input_df)[0]
    st.success(f'Estimated Price: €{pred:,.2f}')
