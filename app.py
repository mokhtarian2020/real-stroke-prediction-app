import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import numpy as np

# Load the model and scaler from your training script
# Example setup to match your previous training setup
X_train = ...  # Your training feature dataset
y_train = ...  # Your training target dataset
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# Streamlit interface to predict based on user input
st.title("Brain Stroke Prediction")
st.write("""
### Enter the following details:
""")

# Collect user input
age = st.slider("Age", 0, 100, 50)  # Example age slider
# Assume other input fields are collected similarly
# Ensure the column names match the model's training feature names
user_data = pd.DataFrame({
    "age": [age],  # Ensure 'age' matches the feature name used in training
    # Add other features similarly
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
