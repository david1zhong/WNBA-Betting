import requests
import json
from datetime import datetime
from zoneinfo import ZoneInfo
from bs4 import BeautifulSoup

url = "https://www.scoresandodds.com/wnba/props"
resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
soup = BeautifulSoup(resp.text, "html.parser")

players = soup.select("div.table-list-row.col-5")

for player in players:
    name_tag = player.select_one("div.props-name span")
    if not name_tag:
        continue
    name = name_tag.get_text(strip=True)

    odds_containers = player.select("div.best-odds.row")

    if len(odds_containers) < 2:
        continue

    over_line = odds_containers[0].select_one("span.data-moneyline").get_text(strip=True)
    over_odds = odds_containers[0].select_one("small.data-odds").get_text(strip=True)

    under_line = odds_containers[1].select_one("span.data-moneyline").get_text(strip=True)
    under_odds = odds_containers[1].select_one("small.data-odds").get_text(strip=True)

    eastern = ZoneInfo("America/New_York")
    now_et = datetime.now(eastern)
    timestamp = now_et.strftime("%Y-%m-%d %H:%M %Z")

    output = {
        "_comment": f"Player props as of {timestamp}",
        "name": name,
        "over_line": over_line,
        "over_odds": over_odds,
        "under_line": under_line,
        "under_odds": under_odds
    }

    with open("props.json", "w") as f:
        json.dump(output, f, indent=2)
