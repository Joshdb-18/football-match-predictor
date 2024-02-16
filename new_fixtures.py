import requests
import pandas as pd
from datetime import datetime

# Sample base URL and season ID
base_url = "https://vgls.betradar.com/vfl/feeds/?/bet9javirtuals/en/Asia:Dhaka/gismo/stats_season_fixtures2/"
season_id = "2803056"

# Fetch the fixture data
fixture_url = f"{base_url}{season_id}"
response = requests.get(fixture_url)
if response.status_code == 200:
    fixture_data = response.json()
else:
    print(f"Failed to fetch fixture data. Status code: {response.status_code}")
    exit()

# Create a list to store the match data
matches_dataset = []

# Iterate over the matches in the fixture data
for match in fixture_data['doc'][0]['data']['matches']:
    match_info = {
        'season_id': season_id,
        'round': match['round'],
        'week': match['week'],
        'home_team_abbr': match['teams']['home']['abbr'],
        'away_team_abbr': match['teams']['away']['abbr'],
    }

    # Additional features can be added based on your specific requirements

    matches_dataset.append(match_info)

# Create a DataFrame from the extracted data
df_matches = pd.DataFrame(matches_dataset)

# Save the dataset to a CSV file
df_matches.to_csv('current fixtures.csv', index=False)
