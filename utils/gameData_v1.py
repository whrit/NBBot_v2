import requests
import pandas as pd
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from time import sleep

MAX_THREADS = 10  # Adjust as necessary
RATE_LIMIT_PER_MINUTE = 300  # Adjust based on API rate limit
SECONDS_PER_MINUTE = 60
INITIAL_SLEEP_INTERVAL = SECONDS_PER_MINUTE / RATE_LIMIT_PER_MINUTE

def get_game_statistics(game_id, season="2023-24"):
    url = "https://api-nba-v1.p.rapidapi.com/games/statistics"
    querystring = {"id": game_id, "season": season}
    headers = {
        "X-RapidAPI-Key": "04bbc1c2d6mshbeed6a20bcbb5edp11712bjsne4f585ea9eb4",
        "X-RapidAPI-Host": "api-nba-v1.p.rapidapi.com"
    }
    response = requests.get(url, headers=headers, params=querystring)
    response.raise_for_status()
    return response.json()['response']

def fetch_stats_and_process(game_id, sleep_interval):
    try:
        game_data = get_game_statistics(game_id)
        sleep(sleep_interval)  # Respect the API rate limit dynamically
        rows = []

        for team_data in game_data:
            team_stats = team_data['statistics'][0]
            team_info = team_data['team']
            date_start = game_data['date']['start']

            row = {
                'game_id': game_id,
                'team_id': team_info['id'],
                'fastBreakPoints': team_stats['fastBreakPoints'],
                'pointsInPaint': team_stats['pointsInPaint'],
                'biggestLead': team_stats['biggestLead'],
                'secondChancePoints': team_stats['secondChancePoints'],
                'pointsOffTurnovers': team_stats['pointsOffTurnovers'],
                'longestRun': team_stats['longestRun'],
                'points': team_stats['points'],
                'fgm': team_stats['fgm'],
                'fga': team_stats['fga'],
                'fgp': team_stats['fgp'],
                'ftm': team_stats['ftm'],
                'fta': team_stats['fta'],
                'ftp': team_stats['ftp'],
                'tpm': team_stats['tpm'],
                'tpa': team_stats['tpa'],
                'tpp': team_stats['tpp'],
                'offReb': team_stats['offReb'],
                'defReb': team_stats['defReb'],
                'totReb': team_stats['totReb'],
                'assists': team_stats['assists'],
                'pFouls': team_stats['pFouls'],
                'steals': team_stats['steals'],
                'turnovers': team_stats['turnovers'],
                'blocks': team_stats['blocks'],
                'plusMinus': team_stats['plusMinus'],
                'min': team_stats['min'],
                'date_start': date_start
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
    game_ids_file = 'GD_Pull_Input1.csv'  # CSV file where game IDs are stored
    output_csv_file = 'GD_Pull_Input1_Results.csv'  # Output CSV file for saving game stats

    # Load game IDs from the CSV
    game_ids = pd.read_csv(game_ids_file)['game_id'].tolist()

    # Process game data and save it
    process_and_save_game_data(game_ids, output_csv_file, INITIAL_SLEEP_INTERVAL)
