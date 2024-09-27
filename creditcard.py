#importing necessay libraries/modules
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import streamlit as st

# Load data
data = pd.read_csv('creditcard.csv')

# Separate legitimate and fraudulent transactions
legit = data[data.Class == 0]
fraud = data[data.Class == 1]

# Undersample legitimate transactions to balance the classes
legit_sample = legit.sample(n=len(fraud), random_state=2)
data = pd.concat([legit_sample, fraud], axis=0)
 
# Split data into training and testing sets
X = data.drop(columns="Class", axis=1)
y = data["Class"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=2)

# Train logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluate model performance
train_acc = accuracy_score(model.predict(X_train), y_train)
test_acc = accuracy_score(model.predict(X_test), y_test)

# Create Streamlit app
st.title("Credit Card Fraud Detection Model")
st.write(f"Logistic Regression Model Training Accuracy: {train_acc:.2f}")
st.write(f"Logistic Regression Model Testing Accuracy: {test_acc:.2f}")
st.write("Enter the following features to check if the transaction is legitimate or fraudulent:")

# Create input fields for user to enter feature values
input_str = st.text_input('Enter all features separated by commas')

# Create a button to submit input and get prediction
if st.button("Submit"):
    try:
        # Parse input feature values
        features = np.array([float(x) for x in input_str.split(',')]).reshape(1, -1)
        if features.shape[1] != X.shape[1]:
            st.write("Error: The number of features entered does not match the expected number.")
        else:
            # Make prediction
            prediction = model.predict(features)
            # Display result
            if prediction[0] == 0:
                st.write("Legitimate transaction")
            else:
                st.write("Fraudulent transaction")
    except ValueError:
        st.write("Error: Please enter valid numeric values separated by commas.")
