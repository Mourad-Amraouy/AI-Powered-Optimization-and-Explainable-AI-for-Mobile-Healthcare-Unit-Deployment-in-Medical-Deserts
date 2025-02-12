### **1. Load and Preprocess Data**

import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from ortools.linear_solver import pywraplp
from sklearn.metrics import mean_absolute_error, mean_squared_error
import streamlit as st

# Load and Preprocess Data
def load_data(file):
    df = pd.read_csv(file)
    df.fillna(df.mean(numeric_only=True), inplace=True)  # Handle missing values
    return df


### **2. Feature Engineering**

def feature_engineering(df):
    required_columns = {'elderly_population', 'total_population', 'num_doctors', 'avg_distance_to_hospital'}
    if not required_columns.issubset(df.columns):
        st.error(f"Missing columns: {required_columns - set(df.columns)}. Please check your dataset.")
        return None
    
    df['elderly_ratio'] = df['elderly_population'] / df['total_population']
    df['doctor_per_1000'] = df['num_doctors'] / (df['total_population'] / 1000)
    df['hospital_distance_score'] = 1 / (df['avg_distance_to_hospital'] + 1)
    return df


### **3. Train ML Model for Demand Prediction**

def train_demand_model(df):
    required_columns = {'elderly_ratio', 'doctor_per_1000', 'hospital_distance_score', 'healthcare_demand'}
    if not required_columns.issubset(df.columns):
        st.error(f"Missing columns: {required_columns - set(df.columns)}. Please check your dataset.")
        return None
    
    X = df[['elderly_ratio', 'doctor_per_1000', 'hospital_distance_score']]
    y = df['healthcare_demand']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    
    st.write(f"**Mean Absolute Error (MAE):** {mean_absolute_error(y_test, y_pred)}")
    st.write(f"**Root Mean Squared Error (RMSE):** {np.sqrt(mean_squared_error(y_test, y_pred))}")

    joblib.dump(model, 'demand_model.pkl')
    return model


### **4. Explainability (SHAP)**

def explain_model(model, X):
    explainer = shap.Explainer(model, X)
    shap_values = explainer(X)
    
    # Show SHAP Summary Plot in Streamlit
    fig, ax = plt.subplots()
    shap.summary_plot(shap_values, X, show=False)
    st.pyplot(fig)


### **5. Optimization Model for Mobile Unit Placement**

def optimize_mobile_units(df):
    required_columns = {'location', 'healthcare_demand', 'cost'}
    if not required_columns.issubset(df.columns):
        st.error(f"Missing columns: {required_columns - set(df.columns)}. Please check your dataset.")
        return None

    solver = pywraplp.Solver.CreateSolver('SCIP')
    num_locations = len(df)
    x = [solver.BoolVar(f'x_{i}') for i in range(num_locations)]
    
    solver.Add(solver.Sum([x[i] * df['cost'].iloc[i] for i in range(num_locations)]) <= 1000000)
    solver.Maximize(solver.Sum([x[i] * df['healthcare_demand'].iloc[i] for i in range(num_locations)]))
    
    status = solver.Solve()
    
    if status == pywraplp.Solver.OPTIMAL:
        return df.loc[[i for i in range(num_locations) if x[i].solution_value() == 1], 'location'].tolist()
    else:
        return []


### **6. Streamlit Dashboard**

def dashboard():
    st.title("XAI Mobile Health Unit Optimization")
    
    uploaded_file = st.file_uploader("Upload Health Data (CSV)", type=["csv"])
    
    if uploaded_file is not None:
        df = load_data(uploaded_file)
        
        if df is not None:
            df = feature_engineering(df)
            
            if df is not None:
                model = train_demand_model(df)
                
                if model:
                    X = df[['elderly_ratio', 'doctor_per_1000', 'hospital_distance_score']]
                    explain_model(model, X)
                    
                    optimized_locations = optimize_mobile_units(df)
                    if optimized_locations:
                        st.write("**Optimized Mobile Unit Locations:**", optimized_locations)
                    else:
                        st.error("Optimization failed. Check dataset constraints.")

if __name__ == '__main__':
    dashboard()