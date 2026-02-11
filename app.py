import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

# 1. Model aur Encoders load karo
model = tf.keras.models.load_model('model.h5')

with open('onehot_encoder_geo.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('label_encode_gender.pkl', 'rb') as file:
    label_encode_gender = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# --- Streamlit UI ---
st.title("Customer Churn Prediction")

# User Inputs
geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0])
gender = st.selectbox('Gender', label_encode_gender.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

# 2. Input Dataframe banaiye
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encode_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

# 3. Geography Encoding
geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

# 4. Merge
input_df = pd.concat([input_data, geo_encoded_df], axis=1)

# --- SABSE IMPORTANT FIX: Column Sequence ---
# Model ko train karte waqt jo sequence tha, wahi yahan set kar rahe hain
input_df = input_df[['CreditScore', 'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary', 'Geography_France', 'Geography_Germany', 'Geography_Spain']]

# 5. Scaling aur Prediction
input_scaled = scaler.transform(input_df)
prediction = model.predict(input_scaled)
prediction_proba = prediction[0][0]

# Results display
st.write(f"### Churn Probability: {prediction_proba:.2%}")

if prediction_proba > 0.5:
    st.error('The customer is likely to churn.')
else:
    st.success('The customer is not likely to churn.')