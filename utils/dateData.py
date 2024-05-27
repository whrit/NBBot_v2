import requests
import pandas as pd
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from time import sleep

MAX_THREADS = 10  # Adjust as necessary
RATE_LIMIT_PER_MINUTE = 300  # Adjust based on API rate limit
SECONDS_PER_MINUTE = 60
INITIAL_SLEEP_INTERVAL = SECONDS_PER_MINUTE / RATE_LIMIT_PER_MINUTE

def get_game_statistics(game_id):
    url = "https://api-nba-v1.p.rapidapi.com/games"
    headers = {
        "X-RapidAPI-Key": "04bbc1c2d6mshbeed6a20bcbb5edp11712bjsne4f585ea9eb4",
        "X-RapidAPI-Host": "api-nba-v1.p.rapidapi.com"
    }
    
    querystring = {"id": game_id}
    response = requests.get(url, headers=headers, params=querystring)
    response.raise_for_status()
    
    if 'response' in response.json():
        return response.json()['response']
    else:
        return []

def fetch_stats_and_process(game_id, sleep_interval):
    try:
        game_data = get_game_statistics(game_id)
        sleep(sleep_interval)  # Respect the API rate limit dynamically
        
        if not game_data:
            print(f"No data found for game ID {game_id}")
            return pd.DataFrame()
        
        rows = []
        for game in game_data:
            date_start = game['date']['start']
            row = {
                'game_id': game_id,
                'date_start': date_start,
            }
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    except requests.exceptions.HTTPError as err:
        if err.response.status_code == 429:  # Too Many Requests
            print(f"Rate limit exceeded fetching stats for game ID {game_id}. Retrying...")
            sleep(sleep_interval + 5)  # Increase sleep time and retry
            return fetch_stats_and_process(game_id, sleep_interval + 5)
        print(f"Error fetching stats for game ID {game_id}: {err}")
        return pd.DataFrame()

def process_and_save_game_data(game_ids, output_file, sleep_interval):
    all_games_df = pd.DataFrame()
    
    with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
        futures = [executor.submit(fetch_stats_and_process, game_id, sleep_interval) for game_id in game_ids]
        
        for future in as_completed(futures):
            game_df = future.result()
            if not game_df.empty:
                all_games_df = pd.concat([all_games_df, game_df], ignore_index=True)
    
    # Save the consolidated data to CSV
    all_games_df.to_csv(output_file, index=False)

if __name__ == '__main__':
    game_ids_file = 'date_fetch_input1.csv'  # CSV file where game IDs are stored
    output_csv_file = 'date_fetch_input1_Results.csv'  # Output CSV file for saving game stats
    
    # Load game IDs from the CSV
    game_ids = pd.read_csv(game_ids_file)['game_id'].tolist()
    
    # Process game data and save it
    process_and_save_game_data(game_ids, output_csv_file, INITIAL_SLEEP_INTERVAL)