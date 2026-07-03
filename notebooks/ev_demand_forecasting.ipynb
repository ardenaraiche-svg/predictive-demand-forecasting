import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# -------------------------------------------------------------
# 1. DATA SIMULATION & ENVIRONMENTAL CALCULATIONS
# -------------------------------------------------------------
np.random.seed(42)
quarters = pd.date_range(start="2021-01-01", end="2026-01-01", freq="QE")
regions = ['Pacific Northwest', 'Midwest', 'Northeast', 'Southwest', 'Southeast']
models = ['EcoSedan', 'EcoSUV', 'EcoTruck']

data = []
for q in quarters:
    for r in regions:
        for m in models:
            # Simulate a steady quarterly growth trend over 5 years
            time_trend = (q.year - 2021) * 150 
            base_sales = np.random.randint(200, 800) + time_trend
            
            # Boost adoption simulation for certain regions
            if r in ['Pacific Northwest', 'Northeast']:
                base_sales = int(base_sales * 1.3)
                
            # Environmental Metrics (Standard ICE car emits ~400g of CO2 per mile)
            avg_miles_year = 11500
            co2_saved_per_mile_tons = 0.0004 
            
            # Aggregate total metrics
            total_miles_driven = base_sales * (avg_miles_year / 4) # Per quarter
            co2_offset = total_miles_driven * co2_saved_per_mile_tons
            
            data.append({
                'Quarter': q,
                'Region': r,
                'EV_Model': m,
                'Units_Sold': base_sales,
                'CO2_Offset_Tons': round(co2_offset, 2)
            })

df = pd.DataFrame(data)

# -------------------------------------------------------------
# 2. FEATURE ENGINEERING
# -------------------------------------------------------------
# Create a lag feature: What were the sales for this specific region/model last quarter?
df['Sales_Lag_1'] = df.groupby(['Region', 'EV_Model'])['Units_Sold'].shift(1)
df.dropna(inplace=True) # Remove the first quarter rows since they won't have a lag

# Convert categorical variables to numeric using One-Hot Encoding
df_encoded = pd.get_dummies(df, columns=['Region', 'EV_Model'], drop_first=True)

# -------------------------------------------------------------
# 3. MODEL TRAINING & EVALUATION
# -------------------------------------------------------------
# Define Features (X) and Target (y)
X = df_encoded.drop(columns=['Quarter', 'Units_Sold', 'CO2_Offset_Tons'])
y = df_encoded['Units_Sold']

# Split into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict and Evaluate
predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("--- MODEL PERFORMANCE METRICS ---")
print(f"Mean Absolute Error (MAE): {mae:.2f} units")
print(f"R² Score (Variance Explained): {r2 * 100:.2f}%\n")

# -------------------------------------------------------------
# 4. ENVIRONMENTAL IMPACT VISUALIZATION
# -------------------------------------------------------------
# Aggregate total environmental impact over time
environmental_trend = df.groupby('Quarter')['CO2_Offset_Tons'].sum().reset_index()

plt.figure(figsize=(10, 5))
sns.lineplot(data=environmental_trend, x='Quarter', y='CO2_Offset_Tons', marker='o', color='#10b981', linewidth=2.5)
plt.title('Cumulative Fleet CO₂ Emission Reductions (Metric Tons Over Time)', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Timeline (Quarters)', fontsize=11)
plt.ylabel('CO₂ Offset (Metric Tons)', fontsize=11)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
