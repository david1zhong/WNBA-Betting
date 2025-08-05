import requests
import json
from datetime import datetime
from zoneinfo import ZoneInfo
from bs4 import BeautifulSoup

url = "https://www.scoresandodds.com/wnba/props"
resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
soup = BeautifulSoup(resp.text, "html.parser")

players = soup.select("div.table-list-row.col-5")
data = []

eastern = ZoneInfo("America/New_York")
now_et = datetime.now(eastern)
timestamp = now_et.strftime("%Y-%m-%d %H:%M %Z")

for player in players:
    name_tag = player.select_one("div.props-name span")
    if not name_tag:
        continue
    name = name_tag.get_text(strip=True)

    odds_containers = player.select("div.best-odds.row")
    if len(odds_containers) < 2:
        continue

    over_line = odds_containers[0].select_one("span.data-moneyline").get_text(strip=True)
    over_line = over_line[1:]
    over_odds = odds_containers[0].select_one("small.data-odds").get_text(strip=True)

    under_line = odds_containers[1].select_one("span.data-moneyline").get_text(strip=True)
    under_line = under_line[1:]
    under_odds = odds_containers[1].select_one("small.data-odds").get_text(strip=True)

    if over_odds == "even":
        over_odds = under_odds

    if under_odds == "even":
        under_odds = over_odds

    data.append({
        "name": name,
        "over_line": over_line,
        "over_odds": over_odds,
        "under_line": under_line,
        "under_odds": under_odds
    })

instances = len(data)

output = {
    "_comment": f"{instances} player props as of {timestamp}",
    "players": data
}

with open("props.json", "w") as f:
    json.dump(output, f, indent=2)
