import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, PowerTransformer
import numpy as np

# Load the dataset
player_hist_stats = pd.read_csv("C:\\Users\\osama\\Downloads\\cleaned_players.csv")


# Select relevant features for modeling
features = ['minutes', 'goals_scored', 'assists', 'clean_sheets', 'influence',
            'creativity', 'threat', 'ict_index', 'total_points']
player_data = player_hist_stats[features].dropna()

# Apply log transformation to skewed features
skewed_features = ['minutes', 'goals_scored', 'assists', 'clean_sheets', 'influence',
                   'creativity', 'threat', 'ict_index']
transformer = PowerTransformer(method='yeo-johnson')
player_data[skewed_features] = transformer.fit_transform(player_data[skewed_features])

# Define the target (total points) and the features for prediction
X = player_data.drop('total_points', axis=1)  # Features (independent variables)
y = player_data['total_points']  # Target (dependent variable)

# Split the data into training and test sets for model evaluation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize XGBoost regressor
xgb_regressor = XGBRegressor()

# Hyperparameter grid for randomized search
param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7],
    'min_child_weight': [1, 3, 5],
    'gamma': [0, 0.1, 0.2],
    'colsample_bytree': [0.3, 0.5, 0.7]
}

# Randomized search with cross-validation
random_search = RandomizedSearchCV(xgb_regressor, param_distributions=param_grid,
                                   n_iter=50, scoring='neg_root_mean_squared_error',
                                   cv=3, verbose=1, random_state=42, n_jobs=-1)

# Train the model with hyperparameter tuning
random_search.fit(X_train, y_train)

# Make predictions
y_pred = random_search.best_estimator_.predict(X_test)

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")

# Display best parameters
print("Best Parameters:", random_search.best_params_)
