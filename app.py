import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression
import numpy as np

# Perform KNN imputation
imputer = KNNImputer(n_neighbors=5)
selected_stroke_data_imputed = pd.DataFrame(imputer.fit_transform(selected_stroke_data), columns=selected_stroke_data.columns)

# Separate features and target
X = selected_stroke_data_imputed.drop(columns=['age'])
y = selected_stroke_data_imputed['age']

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
age = st.slider("Age", int(selected_stroke_data_imputed['age'].min()), int(selected_stroke_data_imputed['age'].max()))
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

# Create user input dataframe with normalized column names
user_data = pd.DataFrame({
    "age": [age],
    "female": [female],
    "hypertension": [hypertension],
    "diabetes": [diabetes],
    "afib": [afib],
    "pfo": [pfo],
    "dyslipid": [dyslipid],
    "smoke": [smoke],
    "live alone": [live_alone],
    "dissection": [dissection],
    "previous stroke": [previous_stroke],
    "previous tia": [previous_tia],
    "cad": [cad],
    "heart failure": [heart_failure],
    "carotid stenosis": [carotid_stenosis]
})

# Standardize user input
user_data_scaled = scaler.transform(user_data)

# Predict the probability of stroke
prediction = model.predict(user_data_scaled)

# Display prediction result with color and alarm
if prediction[0] == 1:
    st.markdown('<p style="color:red; font-size: 20px;">⚠️ Caution! The model predicts that the patient is at risk of a brain stroke.</p>', unsafe_allow_html=True)
else:
    st.markdown('<p style="color:green; font-size: 20px;">✅ Good news! The model predicts that the patient is not at risk of a brain stroke.</p>', unsafe_allow_html=True)
