import requests
from datetime import datetime

year = datetime.now().year
url = f"https://github.com/sportsdataverse/sportsdataverse-data/releases/download/espn_wnba_player_boxscores/player_box_{year}.csv"
filename = f"playerboxes/player_box_{year}.csv"

response = requests.get(url)
if response.status_code == 200:
    with open(filename, "wb") as f:
        f.write(response.content)
    print(f"Updated")
else:
    print("Failed")
