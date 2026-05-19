import importlib
import os
import sys
import json
import traceback
from sqlalchemy import create_engine, text


MODEL_DIR = "models"
DATABASE_URL = os.getenv("DATABASE_URL")

engine = create_engine(DATABASE_URL)


def _normalize_odds(v):
    """Coerce sportsbook odds to a number the DB's double precision column
    will accept. Handles "even" / "EV" (American +100), numeric strings with
    optional + sign, and ints/floats. Returns None for empty or unparseable."""
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip()
    if not s:
        return None
    if s.lower() in ("even", "ev", "evens", "pk", "pick"):
        return 100.0
    try:
        return float(s.lstrip("+"))
    except (TypeError, ValueError):
        return None

model_files = [f[:-3] for f in os.listdir(MODEL_DIR) if f.endswith(".py") and f != "__init__.py"]

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


with engine.connect() as conn:
    for player in players:
        print(f"\n{'#' * 60}")
        print(f"# {player['name']} — {player['date']}")
        print(f"{'#' * 60}")

        for model_name, model in models.items():
            tag = getattr(model, "TAG", f"[{model_name}]")
            print(f"\n=== {tag} {player['name']} on {player['date']} ===")
            result = model.predict(player)
            print()

            if result is None:
                continue

            predicted_pts = result.get("predicted_points")
            bet = result.get("bet")
            note = result.get("note")
            amount = result.get("amount", None)

            params = {
                'player_name': player['name'],
                'model_name': model_name,
                'date': player['date'],
                'predicted_pts': predicted_pts,
                'over_line': player['over_line'],
                'under_line': player['under_line'],
                'over_odds': _normalize_odds(player.get('over_odds')),
                'under_odds': _normalize_odds(player.get('under_odds')),
                'bet': bet,
                'note': note,
                'amount': amount,
            }

            try:
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
                        params,
                    )
            except Exception as e:
                print(
                    f"!!! INSERT FAILED for {player['name']} / {model_name} on "
                    f"{player['date']}: {type(e).__name__}: {e}\n"
                    f"    params: {params}",
                    file=sys.stderr,
                )
                traceback.print_exc(file=sys.stderr)
                continue


print("Predictions added to database.")

