import pandas as pd
import os
from datetime import datetime, timedelta
import pytz
import psycopg2
import json

conn = psycopg2.connect(
    dbname=os.getenv("DB_NAME"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"),
    host=os.getenv("DB_HOST"),
    port=5432
)
cur = conn.cursor()

year = datetime.now().year
file_path = f"playerboxes/player_box_{year}.csv"
df = pd.read_csv(file_path)
df["game_date_time"] = pd.to_datetime(df["game_date_time"])

eastern = pytz.timezone('US/Eastern')
current_time_est = datetime.now(eastern)

yesterday = current_time_est - timedelta(days=1)
yesterday_date = yesterday.strftime('%Y-%m-%d')

with open("props.json", "r") as f:
    data = json.load(f)

for player in data["players"]:
    player_name = player["name"]

    props_date = datetime.strptime(player["date"], "%Y-%m-%d")
    props_date_est = eastern.localize(props_date)
    props_date_utc = props_date_est.astimezone(pytz.UTC)

    window_end = props_date + timedelta(days=1)
    window_end_est = eastern.localize(window_end)
    window_end_utc = window_end_est.astimezone(pytz.UTC)

    candidate_games = df[(df["athlete_display_name"] == player_name) &
                         (df["game_date_time"] >= props_date_utc) &
                         (df["game_date_time"] <= window_end_utc)]

    if candidate_games.empty:
        print(f"{player_name} did not play on {props_date.date()}")
        cur.execute("""
            UPDATE predictions
            SET actual_pts = NULL, result = 'DNP', profit = NULL
            WHERE player_name = %s AND date = %s;
        """, (player_name, props_date.date()))
        continue

    actual_points = int(candidate_games['points'].values[0])
    player_team = candidate_games['team_name'].values[0]

    
    cur.execute("""
        UPDATE predictions
        SET actual_pts = %s,
            pts_differential = %s - predicted_pts,
            team = %s,
            result = CASE
                WHEN bet = 'OVER' AND %s > over_line THEN 'WON'
                WHEN bet = 'UNDER' AND %s < under_line THEN 'WON'
                ELSE 'LOST'
            END
        WHERE player_name = %s AND date = %s;
    """, (
        actual_points,
        actual_points,
        player_team,
        actual_points,
        actual_points,
        player_name,
        props_date.date()
    ))

    cur.execute("""
        UPDATE predictions
        SET profit = CASE
            WHEN amount IS NULL THEN NULL
    
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
        WHERE player_name = %s AND date = %s;
    """, (player_name, props_date.date()))


    print(f"Updated {player_name}: {actual_points} points, team {player_team}")

conn.commit()
cur.close()
conn.close()
