import requests

url = "https://github.com/sportsdataverse/sportsdataverse-data/releases/download/espn_wnba_player_boxscores/player_box_2025.csv"

filename = "playerboxes/player_box_2025.csv"

response = requests.get(url)
if response.status_code == 200:
    with open(filename, "wb") as f:
        f.write(response.content)
    print(f"Updated")
else:
    print("Failed")
