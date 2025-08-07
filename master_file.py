import importlib
import os
import json
from sqlalchemy import create_engine, text


MODEL_DIR = "models"
DATABASE_URL = os.getenv("DATABASE_URL")

engine = create_engine(DATABASE_URL)

model_files = [f[:-3] for f in os.listdir(MODEL_DIR) if f.endswith(".py") and f != "__init__.py"]

models = {}
for mf in model_files:
    module = importlib.import_module(f"{MODEL_DIR}.{mf}")
    models[mf] = module


with open("props.json") as f:
    data = json.load(f)
    players = data["players"]


with engine.begin() as conn:
    for player in players:
        for model_name, model in models.items():
            result = model.predict(player)

            if result is None:
                continue

            predicted_pts = result["predicted_points"]
            bet = result["bet"]

            conn.execute(
                text("""
                    INSERT INTO predictions
                        (player_name, model_name, date,
                         predicted_pts, over_line, under_line,
                         over_odds, under_odds, bet)
                    VALUES
                        (:player_name, :model_name, :date,
                         :predicted_pts, :over_line, :under_line,
                         :over_odds, :under_odds, :bet)
                    ON CONFLICT (player_name, model_name, date) DO NOTHING
                """),
                {
                    'player_name': player_name,
                    'model_name': model_name,
                    'date': player['date'],
                    'predicted_pts': predicted_pts,
                    'over_line': over_line,
                    'under_line': under_line,
                    'over_odds': over_odds,
                    'under_odds': under_odds,
                    'bet': bet
                }
            )


print("Predictions added to database.")

