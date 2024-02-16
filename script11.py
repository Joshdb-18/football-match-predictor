import requests
import json
import pandas as pd

# Define the base URL and the list of numbers
base_url = "https://vgls.betradar.com/vfl/feeds/?/bet9javirtuals/en/Asia:Dhaka/gismo/stats_season_fixtures2/"
numbers = [
    "2803026", "2803056", "2803079", "2803107", "2803131",
    "2803156", "2803180", "2803211", "2803233", "2803260",
    "2803283", "2803313", "2803335", "2803361", "2803388",
    "2803414", "2803437", "2803464", "2803488", "2803515",
    "2803543", "2803569", "2803592", "2803618", "2803643",
    "2803667", "2803697", "2803726", "2803751", "2803776",
    "2803801", "2803826", "2803849", "2803878", "2803902"
]

matches_dataset = []

# Iterate over the numbers, make API calls, and save the responses
for number in numbers:
    url = base_url + number
    print(url)
    response = requests.get(url)
    if response.status_code == 200:
        json_data = response.json()

        # Extracting the 'matches' array from each response
        matches_data = json_data['doc'][0]['data']['matches']
        
        for match in matches_data:
            match_info = {
                'season_id': number,  # Adding the season ID
                '_id': match['_id'],
                '_sid': match['_sid'],
                '_rcid': match['_rcid'],
                '_tid': match['_tid'],
                '_utid': match['_utid'],
                'time': match['time']['time'],
                'date': match['time']['date'],
                'round': match['round'],
                'week': match['week'],
                'home_team': match['teams']['home']['name'],
                'away_team': match['teams']['away']['name'],
                'home_score': match['result']['home'],
                'away_score': match['result']['away'],
                'periods_p1_home': match['periods']['p1']['home'],
                'periods_p1_away': match['periods']['p1']['away'],
                'periods_ft_home': match['periods']['ft']['home'],
                'periods_ft_away': match['periods']['ft']['away'],
                'neutralground': match['neutralground'],
                'inlivescore': match['inlivescore'],
                'winner': match['result']['winner'],
                'comment': match['comment'],
                'status': match['status'],
                'tobeannounced': match['tobeannounced'],
                'postponed': match['postponed'],
                'canceled': match['canceled'],
                'stadiumid': match['stadiumid'],
                'bestof': match['bestof'],
                'walkover': match['walkover'],
                'retired': match['retired'],
                'disqualified': match['disqualified'],
            }
            matches_dataset.append(match_info)

# Create DataFrame from the extracted data
df_matches = pd.DataFrame(matches_dataset)

# Save the dataset to a CSV file
df_matches.to_csv('append to clay matches.csv', index=False)
