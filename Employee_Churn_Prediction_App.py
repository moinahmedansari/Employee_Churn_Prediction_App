import streamlit as st
import pandas as pd
from joblib import load
import dill
import os
import pickle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load the pretrained model
with open(os.path.join(BASE_DIR, 'pipeline.pkl'), 'rb') as file:
    model = dill.load(file)

# Load feature dictionary
my_feature_dict = load(os.path.join(BASE_DIR, 'my_feature_dict.pkl'))

# Function to predict churn
def predict_churn(data):
    prediction = model.predict(data)
    return prediction

st.title('Employee Churn Prediction App')
st.subheader('Based on Employee Dataset')

# Display categorical features
st.subheader('Categorical Features')
categorical_input = my_feature_dict.get('CATEGORICAL')
categorical_input_vals = {}
for i, col in enumerate(categorical_input.get('Column Name').values()):
    categorical_input_vals[col] = st.selectbox(col, categorical_input.get('Members')[i], key=col)

# Load numerical features
numerical_input = my_feature_dict.get('NUMERICAL')

# Display numerical features
st.subheader('Numerical Features')
numerical_input_vals = {}
for col in numerical_input.get('Column Name'):
    numerical_input_vals[col] = st.number_input(col, key=col)

# Combine numerical and categorical input dicts
input_data = dict(list(categorical_input_vals.items()) + list(numerical_input_vals.items()))
input_data = pd.DataFrame.from_dict(input_data, orient='index').T

# Churn Prediction
if st.button('Predict'):
    prediction = predict_churn(input_data)[0]
    
    # Normalize the prediction result (capitalizing first letter to match dictionary keys)
    prediction = prediction.strip().capitalize()  # Strip any extra spaces and capitalize
    
    # Define translation dictionary
    translation_dict = {"Yes": "Expected", "No": "Not Expected"}
    
    # Get the translated prediction
    prediction_translate = translation_dict.get(prediction, "Unknown")  # Default to "Unknown" if not found
    
    # Display the result
    st.write(f"Model's Prediction is **{prediction}**, Hence employee is **{prediction_translate}** to Leave/Churn.")
