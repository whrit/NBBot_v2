import requests
import pandas as pd
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from time import sleep

MAX_THREADS = 5
RATE_LIMIT_PER_MINUTE = 300
SECONDS_PER_MINUTE = 60
INITIAL_SLEEP_INTERVAL = SECONDS_PER_MINUTE / RATE_LIMIT_PER_MINUTE
rate_limit_lock = threading.Lock()

def get_player_stats(player_id, season="2022"):
    url = "https://api-nba-v1.p.rapidapi.com/players/statistics"
    querystring = {"id": player_id, "season": season}
    headers = {
        "X-RapidAPI-Key": "04bbc1c2d6mshbeed6a20bcbb5edp11712bjsne4f585ea9eb4",
        "X-RapidAPI-Host": "api-nba-v1.p.rapidapi.com"
    }
    response = requests.get(url, headers=headers, params=querystring)
    response.raise_for_status()
    return response.json()['response']

def fetch_stats_and_process(row, sleep_interval):
    player_id = row['id']
    player_name = row['name']
    try:
        stats = get_player_stats(player_id)
        sleep(sleep_interval)  # Respect the API rate limit dynamically
        player_stats_df = pd.DataFrame({
            'game_id': [game['game']['id'] for game in stats],
            'points': [game['points'] for game in stats],
            'assists': [game['assists'] for game in stats],
            'rebounds': [game['totReb'] for game in stats],
            'minutes': [game['min'] for game in stats],
            'fgm': [game['fgm'] for game in stats],
            'fga': [game['fga'] for game in stats],
            'fgp': [game['fgp'] for game in stats],
            'ftm': [game['ftm'] for game in stats],
            'fta': [game['fta'] for game in stats],
            'ftp': [game['ftp'] for game in stats],
            'tpm': [game['tpm'] for game in stats],
            'tpa': [game['tpa'] for game in stats],
            'tpp': [game['tpp'] for game in stats],
            'offReb': [game['offReb'] for game in stats],
            'defReb': [game['defReb'] for game in stats],
            'steals': [game['steals'] for game in stats],
            'turnovers': [game['turnovers'] for game in stats],
            'blocks': [game['blocks'] for game in stats],
            'plusMinus': [game['plusMinus'] for game in stats],
            'player_id': [player_id] * len(stats)  # Ensure player_id is replicated for each game
        })
        return player_stats_df
    except requests.exceptions.HTTPError as err:
        if err.response.status_code == 429:  # Too Many Requests
            print(f"Rate limit exceeded fetching stats for {player_name}. Retrying...")
            sleep(sleep_interval + 5)  # Increase sleep time and retry
            return fetch_stats_and_process(row, sleep_interval + 5)
        print(f"Error fetching stats for {player_name}: {err}")
        return pd.DataFrame()

def download_and_save_stats(player_data, sleep_interval):
    all_players_df = pd.DataFrame()
    with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
        futures = [executor.submit(fetch_stats_and_process, row, sleep_interval) for index, row in player_data.iterrows()]
        for future in as_completed(futures):
            player_stats_df = future.result()
            if not player_stats_df.empty:
                all_players_df = pd.concat([all_players_df, player_stats_df], ignore_index=True)

    # Save the consolidated data to CSV
    all_players_df.to_csv("all_nba_player_stats.csv", index=False)

# Load player data from your CSV file
player_data = pd.read_csv("nba_players_and_teams.csv")

download_and_save_stats(player_data, INITIAL_SLEEP_INTERVAL)
