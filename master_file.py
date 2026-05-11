import importlib
import os
import json
from sqlalchemy import create_engine, text


MODEL_DIR = "models"
DATABASE_URL = os.getenv("DATABASE_URL")

engine = create_engine(DATABASE_URL)

model_files = [f[:-3] for f in os.listdir(MODEL_DIR) if f.endswith(".py") and f != "__init__.py"]
# Run order: base models → LEARN models → SELECTIVE meta-models.
# Each later tier reads earlier-tier predictions from the DB, so ordering matters.
def _model_sort_key(n):
    if "SELECTIVE" in n:
        return (2, n)
    if "LEARN" in n:
        return (1, n)
    return (0, n)
model_files.sort(key=_model_sort_key)

models = {}
for mf in model_files:
    module = importlib.import_module(f"{MODEL_DIR}.{mf}")
    models[mf] = module


with open("props.json") as f:
    data = json.load(f)
    players = data["players"]


# NOTE: each INSERT runs in its own short transaction so that base-model
# predictions for today are committed and visible to LEARN models that read
# them back from the predictions table (e.g. CLC3_LEARN's _get_source_predictions
# opens a separate psycopg2 connection — uncommitted writes on this SQLAlchemy
# connection wouldn't be visible to it under Postgres READ COMMITTED isolation).
with engine.connect() as conn:
    for player in players:
        print(f"\n{'#' * 60}")
        print(f"# {player['name']} — {player['date']}")
        print(f"{'#' * 60}")

        for model_name, model in models.items():
            tag = getattr(model, "TAG", f"[{model_name}]")
            print(f"\n=== {tag} {player['name']} on {player['date']} ===")
            result = model.predict(player)
            print()  # blank-line separator after each model finishes

            if result is None:
                continue

            predicted_pts = result.get("predicted_points")
            bet = result.get("bet")
            note = result.get("note")
            amount = result.get("amount", None)

            with conn.begin():
                conn.execute(
                    text("""
                        INSERT INTO predictions
                            (player_name, model_name, date,
                             predicted_pts, over_line, under_line,
                             over_odds, under_odds, bet, note, amount)
                        VALUES
                            (:player_name, :model_name, :date,
                             :predicted_pts, :over_line, :under_line,
                             :over_odds, :under_odds, :bet, :note, :amount)
                        ON CONFLICT (player_name, model_name, date) DO NOTHING
                    """),
                    {
                        'player_name': player['name'],
                        'model_name': model_name,
                        'date': player['date'],
                        'predicted_pts': predicted_pts,
                        'over_line': player['over_line'],
                        'under_line': player['under_line'],
                        'over_odds': player['over_odds'],
                        'under_odds': player['under_odds'],
                        'bet': bet,
                        'note': note,
                        'amount': amount
                    }
                )


print("Predictions added to database.")

