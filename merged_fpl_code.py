import requests
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Step 1: Fetch Data from Fantasy Premier League API
url = 'https://fantasy.premierleague.com/api/bootstrap-static/'
r = requests.get(url)
json_obj = r.json()

# Step 2: Convert the JSON data into DataFrames
elements_df = pd.DataFrame(json_obj['elements'])
elements_types_df = pd.DataFrame(json_obj['element_types'])
teams_df = pd.DataFrame(json_obj['teams'])

# Step 3: Slim down the elements DataFrame to useful columns
slim_elements_df = elements_df[['second_name', 'team', 'element_type', 'selected_by_percent', 'now_cost', 'minutes', 
                                'transfers_in', 'value_season', 'total_points']]

# Map player positions using the element type DataFrame
slim_elements_df.loc[:, 'position'] = slim_elements_df['element_type'].map(elements_types_df.set_index('id').singular_name)

print("Slim Elements DataFrame:")
print(slim_elements_df.head())

# Step 4: Create a Data class to manage player and team data
class Data:
    def __init__(self):
        self.players_data = slim_elements_df
        self.teams_data = teams_df

    # Function to get opponent difficulties based on team strengths
    def get_opponent_difficulties(self, player_id):
        player_team_id = self.players_data.iloc[player_id]['team']
        opponent_team_strengths = self.teams_data[['strength_attack_home', 'strength_defence_home',
                                                   'strength_attack_away', 'strength_defence_away']]
        return opponent_team_strengths.iloc[player_team_id - 1].values
    
    # Function to simulate player gameweek data for regression
    def get_player_gw_data(self, player_id):
        # In this example, we simulate gameweek points based on random normal distribution
        return np.random.normal(loc=self.players_data.iloc[player_id]['total_points'], scale=5, size=38)

# Step 5: Instantiate the Data class and perform Linear Regression
tester = Data()

# Ensure opponent difficulties and player points have the same length (38 gameweeks in a season)
opponent_difficulties = np.random.rand(38, 1)  # Simulate opponent difficulties with 38 values
player_points = tester.get_player_gw_data(0).reshape(-1, 1)

# Perform linear regression
linear_regressor = LinearRegression()
linear_regressor.fit(opponent_difficulties, player_points)
Y_pred = linear_regressor.predict(opponent_difficulties)

# Step 6: Visualize the results
plt.scatter(opponent_difficulties, player_points)
plt.plot(opponent_difficulties, Y_pred, color='red')
plt.title('Player Points vs. Opponent Difficulty')
plt.xlabel('Opponent Difficulty')
plt.ylabel('Player Points')
plt.show()
