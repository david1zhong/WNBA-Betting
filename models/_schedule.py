"""Pre-game opponent/home-away lookup from ESPN's public WNBA scoreboard.

The box-score CSVs only contain games already played, so on the morning of a
game the models can't learn today's matchup from them. This asks ESPN's
scoreboard API instead. Any failure — network down, API shape change, date
not on the scoreboard — degrades to (None, None), which the models treat as
"no opponent context": factors stay neutral, exactly the behavior before
this source existed.

Not a model: master_file skips files starting with "_".
"""
import json
import urllib.request

_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/wnba/scoreboard?dates={date}"
_TIMEOUT_S = 10

# "YYYYMMDD" -> {team_name_lower: (opponent_team_name, "home"|"away")}
_CACHE = {}


def _fetch_day(date_key):
    req = urllib.request.Request(
        _URL.format(date=date_key), headers={"User-Agent": "Mozilla/5.0"}
    )
    with urllib.request.urlopen(req, timeout=_TIMEOUT_S) as resp:
        payload = json.load(resp)

    mapping = {}
    for event in payload.get("events", []):
        for comp in event.get("competitions", []):
            sides = comp.get("competitors", [])
            if len(sides) != 2:
                continue
            for me, other in ((sides[0], sides[1]), (sides[1], sides[0])):
                my_team = (me.get("team") or {}).get("name")
                opp_team = (other.get("team") or {}).get("name")
                ha = str(me.get("homeAway", "")).lower()
                if not my_team or not opp_team or ha not in ("home", "away"):
                    continue
                mapping[my_team.lower()] = (opp_team, ha)
    return mapping


def opponent_for_team(team_name, target_date):
    """(opponent_team_name, "home"|"away") for team_name's game on
    target_date, or (None, None) if unknown or the source is unavailable.

    team_name and the returned opponent use the box CSVs' team_name /
    opponent_team_name vocabulary ("Sparks", "Aces", ...) — ESPN's short
    name, since both datasets originate from ESPN.
    """
    if not team_name:
        return (None, None)
    try:
        date_key = str(target_date)[:10].replace("-", "")
        if len(date_key) != 8 or not date_key.isdigit():
            return (None, None)
        if date_key not in _CACHE:
            try:
                _CACHE[date_key] = _fetch_day(date_key)
            except Exception:
                # Cache the failure so a dead source is hit once per run,
                # not once per player.
                _CACHE[date_key] = {}
        return _CACHE[date_key].get(str(team_name).lower(), (None, None))
    except Exception:
        return (None, None)
