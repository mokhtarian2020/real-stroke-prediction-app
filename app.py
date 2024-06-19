import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

# Load the dataset (assuming 'selected_stroke_data.csv' is your dataset)
selected_stroke_data = pd.read_csv('selected_stroke_data.csv')

# Replace missing values (if any)
selected_stroke_data.fillna(selected_stroke_data.mean(), inplace=True)

# Separate features and target
X = selected_stroke_data.drop(columns=['Age'])
y = selected_stroke_data['Age']

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize numerical features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Streamlit interface
st.title("Brain Stroke Prediction")

st.write("""
### Enter the following details:
""")

# Collect user input
age = st.slider("Age", int(X['Age'].min()), int(X['Age'].max()))
female = st.selectbox("Female", [0, 1])
hypertension = st.selectbox("Hypertension", [0, 1])
diabetes = st.selectbox("Diabetes", [0, 1])
afib = st.selectbox("AFib", [0, 1])
pfo = st.selectbox("PFO", [0, 1])
dyslipid = st.selectbox("Dyslipid", [0, 1])
smoke = st.selectbox("Smoke", [0, 1])
live_alone = st.selectbox("Live Alone", [0, 1])
dissection = st.selectbox("Dissection", [0, 1])
previous_stroke = st.selectbox("Previous Stroke", [0, 1])
previous_tia = st.selectbox("Previous TIA", [0, 1])
cad = st.selectbox("CAD", [0, 1])
heart_failure = st.selectbox("Heart Failure", [0, 1])
carotid_stenosis = st.selectbox("Carotid Stenosis", [0, 1])

# Create user input dataframe
user_data = pd.DataFrame({
    "Age": [age],
    "female": [female],
    "Hypertension": [hypertension],
    "Diabetes": [diabetes],
    "AFib": [afib],
    "PFO": [pfo],
    "Dyslipid": [dyslipid],
    "Smoke": [smoke],
    "Live Alone": [live_alone],
    "Dissection": [dissection],
    "Previous Stroke": [previous_stroke],
    "Previous TIA": [previous_tia],
    "CAD": [cad],
    "Heart Failure": [heart_failure],
    "Carotid Stenosis": [carotid_stenosis]
})

# Handle missing values in user input (if any)
user_data.fillna(selected_stroke_data.mean(), inplace=True)

# Standardize user input
user_data_scaled = scaler.transform(user_data)

# Predict the probability of stroke
prediction = model.predict(user_data_scaled)

# Display prediction result with color and alarm
if prediction[0] == 1:
    st.error("## ⚠️ Caution! The model predicts that the patient is at risk of a brain stroke.")
else:
    st.success("## ✅ Good news! The model predicts that the patient is not at risk of a brain stroke.")
