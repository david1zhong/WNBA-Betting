"""Grade ungraded predictions against the local box-score CSVs.

Reads every (player, date) in the predictions table that is still ungraded
(result NULL) or previously marked DNP, for game dates before today (ET),
and grades whatever the box scores now cover. Keyed to the DB rather than
props.json, so a missed morning run or a late box-score release is retried
automatically on every subsequent run, and a manual run can never touch
tonight's unplayed slate.

Grading:
  WON  — actual on the winning side of the bet's line
  VOID — push: actual exactly on the bet's line (stake returned, profit 0)
  LOST — otherwise
  DNP  — no usable box row for that ET date (re-checked on every run)

Commits per player-date, so one bad row can't discard a whole day's grading.
"""
import os
from datetime import datetime, timedelta

import pandas as pd
import psycopg2
import pytz

conn = psycopg2.connect(
    dbname=os.getenv("DB_NAME"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"),
    host=os.getenv("DB_HOST"),
    port=5432
)
cur = conn.cursor()

eastern = pytz.timezone("US/Eastern")
today_et = datetime.now(eastern).date()

cur.execute(
    """
    SELECT DISTINCT player_name, date
    FROM predictions
    WHERE (result IS NULL OR result = 'DNP')
      AND date < %s
    ORDER BY date, player_name;
    """,
    (today_et,),
)
targets = cur.fetchall()
print(f"{len(targets)} ungraded player-dates (dates before {today_et}).")

_box_cache = {}


def _box_for_year(year):
    """Box scores for one season, loaded once. None if the file is absent."""
    if year not in _box_cache:
        try:
            d = pd.read_csv(f"playerboxes/player_box_{year}.csv")
            d["game_date_time"] = pd.to_datetime(d["game_date_time"])
            d["_name_lower"] = d["athlete_display_name"].str.lower()
        except FileNotFoundError:
            d = None
        _box_cache[year] = d
    return _box_cache[year]


graded = dnp = errors = 0
for player_name, game_date in targets:
    try:
        box = _box_for_year(game_date.year)

        candidate_games = None
        if box is not None:
            day_start = eastern.localize(
                datetime(game_date.year, game_date.month, game_date.day)
            ).astimezone(pytz.UTC)
            day_end = day_start + timedelta(days=1)
            candidate_games = box[
                (box["_name_lower"] == player_name.lower())
                & (box["game_date_time"] >= day_start)
                & (box["game_date_time"] <= day_end)
            ]
            candidate_games = candidate_games[candidate_games["points"].notna()]

        if candidate_games is None or candidate_games.empty:
            print(f"{player_name} — no usable box row for {game_date}, marking DNP")
            cur.execute(
                """
                UPDATE predictions
                SET actual_pts = NULL, result = 'DNP', profit = NULL
                WHERE player_name = %s AND date = %s
                  AND result IS DISTINCT FROM 'VOID';
                """,
                (player_name, game_date),
            )
            conn.commit()
            dnp += 1
            continue

        actual_points = int(candidate_games["points"].values[0])
        player_team = candidate_games["team_name"].values[0]

        cur.execute(
            """
            UPDATE predictions
            SET actual_pts = %s,
                pts_differential = %s - predicted_pts,
                team = %s,
                result = CASE
                    WHEN bet = 'OVER'  AND %s > over_line  THEN 'WON'
                    WHEN bet = 'OVER'  AND %s = over_line  THEN 'VOID'
                    WHEN bet = 'UNDER' AND %s < under_line THEN 'WON'
                    WHEN bet = 'UNDER' AND %s = under_line THEN 'VOID'
                    ELSE 'LOST'
                END
            WHERE player_name = %s AND date = %s
              AND result IS DISTINCT FROM 'VOID';
            """,
            (
                actual_points, actual_points, player_team,
                actual_points, actual_points, actual_points, actual_points,
                player_name, game_date,
            ),
        )

        cur.execute(
            """
            UPDATE predictions
            SET profit = CASE
                WHEN amount IS NULL THEN NULL

                WHEN result = 'VOID' THEN 0

                WHEN result = 'WON' AND bet = 'OVER' THEN
                    CASE
                        WHEN over_odds > 0 THEN ROUND((amount * (over_odds / 100.0))::numeric, 2)
                        ELSE ROUND((amount * (100.0 / ABS(over_odds)))::numeric, 2)
                    END

                WHEN result = 'WON' AND bet = 'UNDER' THEN
                    CASE
                        WHEN under_odds > 0 THEN ROUND((amount * (under_odds / 100.0))::numeric, 2)
                        ELSE ROUND((amount * (100.0 / ABS(under_odds)))::numeric, 2)
                    END

                WHEN result = 'LOST' THEN -amount

                ELSE 0
            END
            WHERE player_name = %s AND date = %s
              AND result IS DISTINCT FROM 'VOID';
            """,
            (player_name, game_date),
        )

        conn.commit()
        graded += 1
        print(f"Graded {player_name} {game_date}: {actual_points} points, team {player_team}")

    except Exception as e:
        conn.rollback()
        errors += 1
        print(f"ERROR grading {player_name} {game_date}: {type(e).__name__}: {e}")

print(f"Done. {graded} graded, {dnp} marked DNP, {errors} errors.")
cur.close()
conn.close()
