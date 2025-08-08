import requests
import json
import pytz
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
current_date_est = datetime.now(eastern).strftime('%Y-%m-%d')

for player in players:
    name_tag = player.select_one("div.props-name span")
    if not name_tag:
        continue
    name = name_tag.get_text(strip=True)

    odds_containers = player.select("div.best-odds.row")
    if len(odds_containers) < 2:
        continue

    over_line_raw = odds_containers[0].select_one("span.data-moneyline")
    under_line_raw = odds_containers[1].select_one("span.data-moneyline")

    over_line = over_line_raw.get_text(strip=True) if over_line_raw else None
    under_line = under_line_raw.get_text(strip=True) if under_line_raw else None

    def is_probably_an_odd(val):
        return val is not None and val.lstrip("+-").isdigit()


    if is_probably_an_odd(over_line):
        over_line = None
    if is_probably_an_odd(under_line):
        under_line = None

    if over_line is None and under_line is None:
        continue

    if over_line is None:
        over_line = "o" + under_line[1:]
    if under_line is None:
        under_line = "u" + over_line[1:]

    over_odds_raw = odds_containers[0].select_one("small.data-odds.best")
    under_odds_raw = odds_containers[1].select_one("small.data-odds.best")

    over_odds = over_odds_raw.get_text(strip=True) if over_odds_raw else None
    under_odds = under_odds_raw.get_text(strip=True) if under_odds_raw else None

    if over_odds is None and under_odds is None:
        over_odds = under_odds = "-120"
    elif over_odds is None:
        over_odds = under_odds
    elif under_odds is None:
        under_odds = over_odds

    if over_odds == "even":
        over_odds = under_odds

    if under_odds == "even":
        under_odds = over_odds

    data.append({
        "name": name,
        "date": current_date_est,
        "over_line": over_line[1:],
        "over_odds": over_odds,
        "under_line": under_line[1:],
        "under_odds": under_odds
    })

instances = len(data)

output = {
    "_comment": f"{instances} player props as of {timestamp}",
    "players": data
}

with open("props.json", "w") as f:
    json.dump(output, f, indent=2)
