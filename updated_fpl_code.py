import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Load the dataset
player_hist_stats = pd.read_csv('player_hist_stats.csv')

# Select relevant features from the dataset for modeling
features = ['minutes', 'goals_scored', 'assists', 'clean_sheets', 'influence', 
            'creativity', 'threat', 'ict_index', 'total_points']
player_data = player_hist_stats[features].dropna()

# Define the target (total points) and the features for prediction
X = player_data.drop('total_points', axis=1)  # Features (independent variables)
y = player_data['total_points']  # Target (dependent variable)

# Split the data into training and test sets for model evaluation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 1: Random Forest Regressor
rf_regressor = RandomForestRegressor()
rf_regressor.fit(X_train, y_train)
rf_y_pred = rf_regressor.predict(X_test)

# Step 2: Gradient Boosting Regressor
gb_regressor = GradientBoostingRegressor()
gb_regressor.fit(X_train, y_train)
gb_y_pred = gb_regressor.predict(X_test)

# Visualize the actual vs predicted values for the two models
plt.figure(figsize=(10, 6))
plt.scatter(range(len(y_test)), y_test, label="Actual Player Points", color='blue', alpha=0.6)
plt.scatter(range(len(y_test)), rf_y_pred, label="Random Forest Prediction", color='red', alpha=0.6)
plt.scatter(range(len(y_test)), gb_y_pred, label="Gradient Boosting Prediction", color='green', alpha=0.6)
plt.title("Player Points: Actual vs Predicted (Model Comparison)")
plt.xlabel("Sample Index")
plt.ylabel("Player Points")
plt.legend()
plt.show()
