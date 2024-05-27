import requests
import pandas as pd
import threading
import time

MAX_THREADS = 5  
RATE_LIMIT_PER_MINUTE = 300
SECONDS_PER_MINUTE = 60

def get_games_by_season(season):
    url = "https://api-nba-v1.p.rapidapi.com/games"
    querystring = {"season": season}
    headers = {
        "X-RapidAPI-Key": "04bbc1c2d6mshbeed6a20bcbb5edp11712bjsne4f585ea9eb4",
        "X-RapidAPI-Host": "api-nba-v1.p.rapidapi.com"
    }
    response = requests.get(url, headers=headers, params=querystring)
    response.raise_for_status()
    return response.json()['response']

def fetch_all_games(season="2022"):
    games = get_games_by_season(season)
    games_data = {}

    for game in games:
        game_id = game['id']
        games_data[game_id] = {
            'game_id': game_id,
        }
    return games_data

if __name__ == '__main__':
    game_data = fetch_all_games()
    print(game_data)  # Inspect the game data

    # Store in a CSV
    df = pd.DataFrame.from_dict(game_data, orient='index')
    df.to_csv('nba_games_2022_23_id.csv', index=False)