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
file_path = f"wnba/player_box/player_box_{year}.csv"
df = pd.read_csv(file_path)

est = pytz.timezone('US/Eastern')
current_time_est = datetime.now(est)
yesterday = current_time_est - timedelta(days=2)
formatted_date = yesterday.strftime('%Y-%m-%d')

with open("props.json", "r") as f:
    data = json.load(f)

for player in data["players"]:
    player_name = player["name"]

    filtered_data = df[
        (df['athlete_display_name'] == player_name) &
        (df['game_date'] == formatted_date)
    ]

    if len(filtered_data) == 0:
        print(f"No data for {player_name} on {formatted_date}")
        continue

    actual_points = int(filtered_data['points'].values[0])
    player_team = filtered_data['team_name'].values[0]

    cur.execute("""
        UPDATE predictions
        SET actual_pts = %s
        WHERE player_name = %s AND date = %s;
    """, (actual_points, player_name, formatted_date))

    cur.execute("""
        UPDATE predictions
        SET pts_differential = actual_pts - predicted_pts
        WHERE player_name = %s AND date = %s;
    """, (player_name, formatted_date))

    cur.execute("""
        UPDATE predictions
        SET team = %s
        WHERE player_name = %s AND date = %s;
    """, (player_team, player_name, formatted_date))

    cur.execute("""
        UPDATE predictions
        SET result = CASE
            WHEN bet = 'OVER' AND actual_pts > over_line THEN 'WON'
            WHEN bet = 'UNDER' AND actual_pts < under_line THEN 'WON'
            ELSE 'LOST'
        END
        WHERE player_name = %s AND date = %s;
    """, (player_name, formatted_date))

    print(f"Updated {player_name}: {actual_points} points, team {player_team}")


conn.commit()

cur.close()
conn.close()
