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

# Navigation
project = st.sidebar.selectbox(
    "Select Project",
    [
        "Overview",
        "Readmission Risk",
        "ED Wait Times",
        "Clinical Trials",
        "Cost Prediction",
        "Population Health",
    ],
)

if project == "Overview":
    st.title("Healthcare Analytics Portfolio")
    st.write("""
    This portfolio showcases 5 key healthcare analytics projects demonstrating various analytical approaches and tools:
    
    1. **Readmission Risk Prediction**: Machine learning model to predict patient readmission risk
    2. **ED Wait Times Analysis**: Time series analysis of emergency department metrics
    3. **Clinical Trials Analysis**: Statistical analysis of trial outcomes
    4. **Healthcare Cost Prediction**: Predictive modeling for cost estimation
    5. **Population Health Dashboard**: Geographic and demographic health analysis
    """)

elif project == "Readmission Risk":
    st.title("Hospital Readmission Risk Prediction")

    # Generate sample data
    @st.cache_data
    def generate_sample_data(n_samples=1000):
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

elif project == "ED Wait Times":
    st.title("Emergency Department Wait Times Analysis")

    # Generate sample ED data
    @st.cache_data
    def generate_ed_data():
        dates = pd.date_range(start="2023-01-01", end="2024-12-31", freq="H")
        n_samples = len(dates)

        data = {
            "timestamp": dates,
            "wait_time": np.random.exponential(45, n_samples),  # minutes
            "patients_in_queue": np.random.poisson(10, n_samples),
            "staff_on_duty": np.random.randint(5, 15, n_samples),
        }
        return pd.DataFrame(data)

    ed_data = generate_ed_data()

    # Time series plot
    st.subheader("Wait Time Trends")
    fig = px.line(
        ed_data.resample("D", on="timestamp").mean(),
        y="wait_time",
        title="Average Daily Wait Times",
    )
    st.plotly_chart(fig)

    # Heatmap by hour and day
    pivot_data = ed_data.copy()
    pivot_data["hour"] = pivot_data["timestamp"].dt.hour
    pivot_data["day"] = pivot_data["timestamp"].dt.day_name()

    heatmap_data = pivot_data.pivot_table(
        values="wait_time", index="hour", columns="day", aggfunc="mean"
    )

    st.subheader("Wait Times by Hour and Day")
    fig = px.imshow(
        heatmap_data,
        labels=dict(x="Day of Week", y="Hour of Day", color="Wait Time (min)"),
        aspect="auto",
    )
    st.plotly_chart(fig)

elif project == "Clinical Trials":
    st.title("Clinical Trials Analysis")

    # Generate sample clinical trial data
    @st.cache_data
    def generate_trial_data():
        np.random.seed(42)
        n_patients = 500

        data = {
            "patient_id": range(n_patients),
            "age": np.random.normal(55, 12, n_patients),
            "treatment_group": np.random.choice(["Treatment", "Control"], n_patients),
            "baseline_score": np.random.normal(50, 10, n_patients),
            "final_score": None,
            "adverse_events": np.random.binomial(1, 0.15, n_patients),
        }

        df = pd.DataFrame(data)
        # Simulate treatment effect
        treatment_effect = 15
        df.loc[df["treatment_group"] == "Treatment", "final_score"] = df.loc[
            df["treatment_group"] == "Treatment", "baseline_score"
        ] + np.random.normal(
            treatment_effect, 5, len(df[df["treatment_group"] == "Treatment"])
        )

        df.loc[df["treatment_group"] == "Control", "final_score"] = df.loc[
            df["treatment_group"] == "Control", "baseline_score"
        ] + np.random.normal(5, 5, len(df[df["treatment_group"] == "Control"]))

        return df

    trial_data = generate_trial_data()

    # Treatment effect visualization
    st.subheader("Treatment Effect Analysis")
    fig = px.box(
        trial_data,
        x="treatment_group",
        y="final_score",
        title="Final Scores by Treatment Group",
    )
    st.plotly_chart(fig)

    # Adverse events analysis
    st.subheader("Adverse Events Analysis")
    adverse_events = trial_data.groupby("treatment_group")["adverse_events"].mean()
    fig = px.bar(adverse_events, title="Adverse Event Rates by Group")
    st.plotly_chart(fig)

elif project == "Cost Prediction":
    st.title("Healthcare Cost Prediction")

    # Generate sample cost data
    @st.cache_data
    def generate_cost_data():
        n_samples = 2000
        data = {
            "age": np.random.normal(50, 15, n_samples),
            "bmi": np.random.normal(28, 5, n_samples),
            "smoker": np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
            "chronic_conditions": np.random.poisson(1.5, n_samples),
            "insurance_type": np.random.choice(
                ["Private", "Medicare", "Medicaid"], n_samples
            ),
            "total_cost": None,
        }
        df = pd.DataFrame(data)

        # Simulate cost relationships
        base_cost = 5000
        df["total_cost"] = (
            base_cost
            + df["age"] * 100
            + df["bmi"] * 200
            + df["smoker"] * 2000
            + df["chronic_conditions"] * 3000
            + np.random.normal(0, 1000, n_samples)
        )

        return df

    cost_data = generate_cost_data()

    # Train cost prediction model
    X = pd.get_dummies(cost_data.drop("total_cost", axis=1))
    y = cost_data["total_cost"]

    model = xgb.XGBRegressor()
    model.fit(X, y)

    # Cost distribution
    st.subheader("Healthcare Cost Distribution")
    fig = px.histogram(
        cost_data, x="total_cost", nbins=50, title="Distribution of Healthcare Costs"
    )
    st.plotly_chart(fig)

    # Feature importance
    importance_df = pd.DataFrame(
        {"feature": X.columns, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)

    st.subheader("Cost Drivers")
    fig = px.bar(
        importance_df.head(10), x="feature", y="importance", title="Top Cost Drivers"
    )
    st.plotly_chart(fig)

elif project == "Population Health":
    st.title("Population Health Dashboard")

    # Generate sample population health data
    @st.cache_data
    def generate_population_data():
        regions = ["North", "South", "East", "West", "Central"]
        n_regions = len(regions)

        data = {
            "region": regions,
            "population": np.random.randint(100000, 1000000, n_regions),
            "obesity_rate": np.random.normal(30, 5, n_regions),
            "smoking_rate": np.random.normal(15, 3, n_regions),
            "diabetes_rate": np.random.normal(10, 2, n_regions),
            "vaccination_rate": np.random.normal(75, 8, n_regions),
        }
        return pd.DataFrame(data)

    pop_data = generate_population_data()

    # Health metrics comparison
    st.subheader("Regional Health Metrics")
    metrics = ["obesity_rate", "smoking_rate", "diabetes_rate", "vaccination_rate"]
    selected_metric = st.selectbox("Select Health Metric", metrics)

    fig = px.bar(
        pop_data,
        x="region",
        y=selected_metric,
        title=f'{selected_metric.replace("_", " ").title()} by Region',
    )
    st.plotly_chart(fig)

    # Correlation analysis
    st.subheader("Health Metrics Correlation")
    corr_matrix = pop_data[metrics].corr()
    fig = px.imshow(
        corr_matrix,
        labels=dict(color="Correlation Coefficient"),
        title="Correlation between Health Metrics",
    )
    st.plotly_chart(fig)

st.sidebar.markdown("""
---
Created by: [Your Name]  
Data: Synthetic healthcare data for demonstration

**Portfolio Highlights:**
- 5 comprehensive healthcare analytics projects
- Multiple analytical approaches demonstrated
- Industry-standard tools and packages
- Interactive visualizations
- Real-world healthcare applications
""")
