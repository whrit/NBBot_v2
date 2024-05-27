import pandas as pd

# Load the datasets
odds_df = pd.read_csv('odds_2022-23.csv')
game_stats_df = pd.read_csv('combined_gd_22_23_v1.csv')

# Convert dates to datetime objects
odds_df['Date'] = pd.to_datetime(odds_df['Date'], format='%m/%d/%y')
game_stats_df['date'] = pd.to_datetime(game_stats_df['date'], format='%m/%d/%y')

# Assuming team IDs are already correct in the odds_df (based on your last message)

# This function will check both home and away IDs against the game stats to find matching game_ids
def find_game_ids(row):
    # Find matches where both home and away teams match on the same game_id and date
    matches = game_stats_df[
        (game_stats_df['date'] == row['Date']) &
        (game_stats_df['team_id'].isin([row['Home'], row['Away']]))
    ]
    # Filter further to get only those game_ids that have both teams involved
    valid_game_ids = matches['game_id'].value_counts()
    return valid_game_ids[valid_game_ids == 2].index.tolist()  # Only return game_ids where exactly two matches are found

# Apply the function to find corresponding game IDs
odds_df['Game_IDs'] = odds_df.apply(find_game_ids, axis=1)

# Explode the Game_IDs list to match on individual rows
combined_df = odds_df.explode('Game_IDs')

# Merge the exploded dataframe with the game stats
combined_df = pd.merge(combined_df, game_stats_df, left_on=['Game_IDs', 'Date'], right_on=['game_id', 'date'], how='left')

# Save the combined dataframe
combined_df.to_csv('combined_gd_odds_22_23_v1.csv', index=False)