import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE

# Load the new dataset
selected_stroke_data = pd.read_csv('selected_stroke_data.csv')

# Display the first few rows to understand the structure
st.write(selected_stroke_data.head())

# Separate features and target variable
X = selected_stroke_data.drop(columns=['Brain stroke'])
y = selected_stroke_data['Brain stroke']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the numerical features (assuming Age is the only continuous variable)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Apply SMOTE to balance the dataset (since it's implied by your context)
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train_scaled, y_train)

# Train the logistic regression model
model = LogisticRegression()
model.fit(X_resampled, y_resampled)

# Streamlit interface for prediction
st.title("Brain Stroke Prediction")

st.write("""
### Enter the following details:
""")

# Collect user input
age = st.slider("Age", 0, 100)
female = st.selectbox("Female", ["No", "Yes"])
hypertension = st.selectbox("Hypertension", ["No", "Yes"])
diabetes = st.selectbox("Diabetes", ["No", "Yes"])
afib = st.selectbox("AFib", ["No", "Yes"])
pfo = st.selectbox("PFO", ["No", "Yes"])
dyslipid = st.selectbox("Dyslipid", ["No", "Yes"])
smoke = st.selectbox("Smoke", ["No", "Yes"])
live_alone = st.selectbox("Live Alone", ["No", "Yes"])
dissection = st.selectbox("Dissection", ["No", "Yes"])
previous_stroke = st.selectbox("Previous Stroke", ["No", "Yes"])
previous_tia = st.selectbox("Previous TIA", ["No", "Yes"])
cad = st.selectbox("CAD", ["No", "Yes"])
heart_failure = st.selectbox("Heart Failure", ["No", "Yes"])
carotid_stenosis = st.selectbox("Carotid Stenosis", ["No", "Yes"])

# Preprocess user input
user_data = {
    "Age": [age],
    "Female": [1 if female == "Yes" else 0],
    "Hypertension": [1 if hypertension == "Yes" else 0],
    "Diabetes": [1 if diabetes == "Yes" else 0],
    "AFib": [1 if afib == "Yes" else 0],
    "PFO": [1 if pfo == "Yes" else 0],
    "Dyslipid": [1 if dyslipid == "Yes" else 0],
    "Smoke": [1 if smoke == "Yes" else 0],
    "Live Alone": [1 if live_alone == "Yes" else 0],
    "Dissection": [1 if dissection == "Yes" else 0],
    "Previous Stroke": [1 if previous_stroke == "Yes" else 0],
    "Previous TIA": [1 if previous_tia == "Yes" else 0],
    "CAD": [1 if cad == "Yes" else 0],
    "Heart Failure": [1 if heart_failure == "Yes" else 0],
    "Carotid Stenosis": [1 if carotid_stenosis == "Yes" else 0]
}

user_df = pd.DataFrame(user_data)

# Scale the user input using the same scaler
user_df_scaled = scaler.transform(user_df)

# Predict the probability of stroke
prediction = model.predict(user_df_scaled)

if prediction[0] == 1:
    st.write("## The model predicts that you are at risk of a brain stroke.")
else:
    st.write("## The model predicts that you are not at risk of a brain stroke.")
