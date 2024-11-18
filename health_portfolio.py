import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import roc_curve, auc, confusion_matrix
import shap

st.set_page_config(page_title="Healthcare Analytics Portfolio", layout="wide")

st.title("Hospital Readmission Risk Prediction")

# Generate sample data
@st.cache_data
def generate_sample_data(n_samples=10000):
    np.random.seed(42)
    data = {
        "age": np.random.normal(65, 15, n_samples),
        "length_of_stay": np.random.exponential(5, n_samples),
        "num_procedures": np.random.poisson(3, n_samples),
        "num_medications": np.random.poisson(5, n_samples),
        "num_diagnoses": np.random.poisson(4, n_samples),
        "num_prev_admissions": np.random.poisson(2, n_samples),
        "readmitted": np.random.binomial(1, 0.3, n_samples),
    }
    return pd.DataFrame(data)

df = generate_sample_data()

# Train model
X = df.drop("readmitted", axis=1)
y = df["readmitted"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestClassifier()
model.fit(X_train, y_train)

# Interactive risk calculator
st.subheader("Patient Risk Calculator")

col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age", 18, 100, 65)
    los = st.number_input("Length of Stay (days)", 1, 30, 5)
    procedures = st.number_input("Number of Procedures", 0, 20, 3)

with col2:
    medications = st.number_input("Number of Medications", 0, 30, 5)
    diagnoses = st.number_input("Number of Diagnoses", 0, 20, 4)
    prev_admissions = st.number_input("Previous Admissions", 0, 10, 2)

# Calculate risk
patient_data = pd.DataFrame(
    {
        "age": [age],
        "length_of_stay": [los],
        "num_procedures": [procedures],
        "num_medications": [medications],
        "num_diagnoses": [diagnoses],
        "num_prev_admissions": [prev_admissions],
    }
)

risk_score = model.predict_proba(patient_data)[0][1]

# Display risk gauge
fig = go.Figure(
    go.Indicator(
        mode="gauge+number",
        value=risk_score * 100,
        domain={"x": [0, 1], "y": [0, 1]},
        title={"text": "Readmission Risk Score"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "darkblue"},
            "steps": [
                {"range": [0, 30], "color": "lightgreen"},
                {"range": [30, 70], "color": "yellow"},
                {"range": [70, 100], "color": "red"},
            ],
        },
    )
)
st.plotly_chart(fig)
