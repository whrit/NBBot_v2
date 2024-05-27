import pandas as pd

# Function to read and combine player data
def combine_player_data(last_season_path, current_season_path):
    last_season_data = pd.read_csv(last_season_path)
    current_season_data = pd.read_csv(current_season_path)
    
    # Combine player data from both seasons
    combined_player_data = pd.concat([last_season_data, current_season_data], ignore_index=True)
    
    return combined_player_data

# Function to read and combine game data
def combine_game_data(last_season_path, current_season_path):
    last_season_data = pd.read_csv(last_season_path)
    current_season_data = pd.read_csv(current_season_path)
    
    # Combine game data from both seasons
    combined_game_data = pd.concat([last_season_data, current_season_data], ignore_index=True)
    
    return combined_game_data

# File paths for player data
last_season_player_path = "all_nba_player_stats_2022_23.csv"
current_season_player_path = "all_nba_player_stats_2023_24.csv"

# File paths for game data
last_season_game_path = "nba_games_2022_23_data.csv"
current_season_game_path = "nba_games_2023_24_data.csv"

# Combine player data
combined_player_data = combine_player_data(last_season_player_path, current_season_player_path)

# Combine game data
combined_game_data = combine_game_data(last_season_game_path, current_season_game_path)

# Save the combined data to new CSV files
combined_player_data.to_csv("combined_pd_v1.csv", index=False)
combined_game_data.to_csv("combined_gd_v1.csv", index=False)