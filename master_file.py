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
    players = json.load(f)


with engine.begin() as conn:
    for player in players:
        for model_name, model in models.items():
            predicted_pts = model.predict(player)

            if predicted_pts is None:
                continue

            bet = "OVER" if predicted_pts > player["over_line"] else "UNDER"

            conn.execute(
                text("""
                    INSERT INTO predictions
                        (player_name, model_name,
                         predicted_pts, over_line, under_line,
                         over_odds, under_odds, bet)
                    VALUES
                        (:player_name, :model_name,
                         :predicted_pts, :over_line, :under_line,
                         :over_odds, :under_odds, :bet)
                    ON CONFLICT (player_name, model_name) DO NOTHING
                """),
                {
                    "player_name": player["name"],
                    "model_name": model_name,
                    "predicted_pts": predicted_pts,
                    "over_line": player["over_line"],
                    "under_line": player["under_line"],
                    "over_odds": player["over_odds"],
                    "under_odds": player["under_odds"],
                    "bet": bet,
                }
            )

print("Predictions added to database.")
