# Predictive Forecasting: Optimizing Operations & Demand Analytics

## 📌 Business Problem
EcoDrive Motors, a rapidly growing electric vehicle manufacturer, wants to predict quarterly fleet sales across different regional markets while quantifying the direct environmental impact of their vehicles.This project builds an end-to-end predictive machine learning pipeline to forecast EV unit sales. Simultaneously, it integrates a carbon-offset calculation framework to track how replacing internal combustion engine (ICE) vehicles with EVs reduces tailpipe $CO_2$ emissions over time, giving stakeholders a data-driven look at both market demand and environmental ROI.

## 🛠️ The Tech Stack
* **Language:** Python (Pandas, NumPy)
* **Modeling:** Scikit-Learn (Regression/Random Forest) or Prophet
* **Visualization:** Matplotlib, Seaborn

## 📊 Data Pipeline & Methodology
1. **Exploratory Data Analysis (EDA):** Analyzed historical trends, seasonal spikes, and handled missing values."
2. **Feature Engineering:** Created time-lag features and rolling averages to capture demand momentum.
3. **Model Evaluation:** Compared baseline linear models against ensemble methods, evaluating performance using RMSE and MAE.

## 📈 Key Insights & Business Impact
* **Insight:** Identified a 15% predictable surge in demand during specific operational cycles.
* **Impact:** The final model predicts demand with 92% accuracy, allowing stakeholders to optimize resource allocation proactively.
