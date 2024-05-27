import requests
import pandas as pd
import threading
import time

MAX_THREADS = 5  
RATE_LIMIT_PER_MINUTE = 300
SECONDS_PER_MINUTE = 60

def get_players_by_team(team_id, season="2023"):
    url = "https://api-nba-v1.p.rapidapi.com/players"
    querystring = {"team": team_id, "season": season}
    headers = {
        "X-RapidAPI-Key": "04bbc1c2d6mshbeed6a20bcbb5edp11712bjsne4f585ea9eb4",
        "X-RapidAPI-Host": "api-nba-v1.p.rapidapi.com"
    }
    response = requests.get(url, headers=headers, params=querystring)
    response.raise_for_status()
    return response.json()['response']  

def get_all_players():
    team_ids = list(range(1, 31)) 
    all_players = {}

    def fetch_team_data(team_id):  
        try:
            players = get_players_by_team(team_id)
            print(players)
            for player in players:
                player_id = player['id']
                all_players[player_id] = {
                    'player_id': player_id,
                    'name': player['firstname'] + ' ' + player['lastname'],  # Construct full name
                    'team_id': team_id 
                }
        except requests.exceptions.HTTPError as err:
            print(f"HTTP Error for team {team_id}: {err}") 

    threads = []
    for i in range(0, len(team_ids), MAX_THREADS):
        batch = team_ids[i:i+MAX_THREADS]
        for team_id in batch:
            t = threading.Thread(target=fetch_team_data, args=(team_id,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()  
        threads = []  # Clear threads for the next batch 

    return all_players

if __name__ == '__main__':
    player_data = get_all_players()
    print(player_data)  # Inspect the player data 
    # Option 1: Store in a CSV
    df = pd.DataFrame.from_dict(player_data, orient='index')
    df.to_csv('nba_players_and_teams.csv', index=True)