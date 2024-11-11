import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, PowerTransformer
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset
player_hist_stats = pd.read_csv("C:\\Users\\osama\\Downloads\\cleaned_merged_seasons.csv")

# Select relevant features for modeling
features = ['minutes', 'goals_scored', 'assists', 'clean_sheets', 'influence',
            'creativity', 'threat', 'ict_index', 'total_points', 'value']
player_data = player_hist_stats[features].dropna()

# Apply log transformation to skewed features
skewed_features = ['minutes', 'goals_scored', 'assists', 'clean_sheets', 'influence',
                   'creativity', 'threat', 'ict_index']
transformer = PowerTransformer(method='yeo-johnson')
player_data[skewed_features] = transformer.fit_transform(player_data[skewed_features])

# Define the target (total points) and the features for prediction
X = player_data.drop(['total_points'], axis=1)  # Features (independent variables)
y = player_data['total_points']  # Target (dependent variable)

# Split the data into training and test sets for model evaluation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize and train the XGBoost regressor
xgb_regressor = XGBRegressor(n_estimators=100, random_state=42)
xgb_regressor.fit(X_train, y_train)

# Make predictions
y_pred = xgb_regressor.predict(X_test)

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")

# Calculate ROI (points per million cost)
player_data['predicted_points'] = xgb_regressor.predict(scaler.transform(X))
player_data['ROI'] = player_data['predicted_points'] / player_data['value']

# Draft a team based on the highest predicted ROI within budget constraints
budget = 84  # Budget constraint in millions
selected_team = player_data.sort_values(by='ROI', ascending=False).head(11)
team_cost = selected_team['value'].sum()
team_points = selected_team['predicted_points'].sum()

# Ensure that 'name' and 'position' columns exist before printing them
available_columns = selected_team.columns
if 'name' in available_columns and 'position' in available_columns:
    print("\nSelected Team:")
    print(selected_team[['name', 'position', 'predicted_points', 'ROI', 'value']])
else:
    print("\nSelected Team (without name and position columns):")
    print(selected_team[['predicted_points', 'ROI', 'value']])

print(f"\nTotal Team Cost: {team_cost:.2f} million")
print(f"Total Predicted Points for Team: {team_points:.2f}")

# Plot actual vs predicted points for sample players
plt.figure(figsize=(10, 6))
plt.scatter(range(len(y_test)), y_test, color='blue', label='Actual Points')
plt.scatter(range(len(y_pred)), y_pred, color='red', label='Predicted Points')
plt.title('Actual vs Predicted Points')
plt.xlabel('Player Index')
plt.ylabel('Points')
plt.legend()
plt.show()
