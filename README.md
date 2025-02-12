# AI-Powered Optimization and Explainable AI for Mobile Healthcare Unit Deployment in Medical Deserts 🚀

## Project Overview 🔍

This project focuses on developing an AI-driven decision support system for the optimal placement and resource allocation of mobile healthcare units in underserved regions (medical deserts). The system integrates demand prediction, facility location optimization, and staff allocation, ensuring model interpretability using Explainable AI (XAI) techniques.

## Objectives 🎯

- 📊 Predict future healthcare demand using Machine Learning.
- 📍 Optimize mobile healthcare unit location and capacity allocation using mathematical models.
- 🛠️ Ensure explainability (XAI) for transparent decision-making using SHAP.

## Features 🔧

- **📈 Healthcare Demand Prediction**: Uses Random Forest to predict demand for medical services.
- **📌 Optimization for Mobile Unit Placement**: Uses Mixed Integer Programming (MIP) with OR-Tools.
- **🧐 Explainable AI (XAI)**: Implements SHAP to make model decisions transparent.
- **📊 Interactive Dashboard**: Built with Streamlit for user-friendly visualization and insights.

## Technologies Used 🛠️

- **🐍 Python**: Pandas, NumPy, Scikit-Learn, OR-Tools, SHAP, Streamlit
- **🤖 Machine Learning**: RandomForestRegressor
- **📍 Optimization**: Google OR-Tools (pywraplp)
- **📊 Visualization**: Matplotlib, SHAP Summary Plots

## Installation 📂

### 1️⃣ Clone the repository:
```sh
git clone https://github.com/yourusername/ai-healthcare-optimization.git
cd ai-healthcare-optimization
```

### 2️⃣ Install dependencies:
```sh
pip install -r requirements.txt
```

### 3️⃣ Run the Streamlit dashboard:
```sh
streamlit run colab_kernel_launcherV0.py
```

## Usage 🧑‍💻

1. 📂 **Upload CSV Data**: The application accepts a structured CSV file containing demographic and healthcare demand data.
2. 🔍 **Predict Healthcare Demand**: The ML model estimates demand based on available features.
3. 🧐 **Explainability with SHAP**: Provides feature importance and explainable AI insights.
4. 📍 **Optimize Mobile Units**: Uses mathematical programming to determine optimal placement.

## Example Dataset Format 📊

| location | elderly_population | total_population | num_doctors | avg_distance_to_hospital | healthcare_demand | cost |
|----------|--------------------|------------------|------------|--------------------------|------------------|------|
| A        | 5000               | 50000            | 10         | 5                        | 3000             | 1000 |
| B        | 7000               | 70000            | 15         | 3                        | 5000             | 1500 |

## Future Enhancements 🚀

- **🧠 Deep Learning for Demand Prediction**: Implement LSTMs for better forecasting.
- **📡 Real-Time Optimization**: Incorporate real-time healthcare demand tracking.
- **🌍 Geospatial Analysis**: Use GIS for better location mapping and accessibility.
- **⚖️ Multi-Objective Optimization**: Balance cost, demand coverage, and equity.
- **🔗 Integrate with Public Health APIs**: Fetch real-time demographic and health data.

## License 📜

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments 🙌

- Inspired by research on AI-driven healthcare optimization.
- Thanks to the OR-Tools and SHAP communities for their amazing tools.
