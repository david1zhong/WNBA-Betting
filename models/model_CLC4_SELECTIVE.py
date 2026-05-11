"""
CLC4_SELECTIVE — a selective ensemble meta-model.

Reads today's predictions from the six existing source models, weights each
by its rolling historical win rate (last 100 graded bets), and bets only when
the ensemble probability of being right clears the sportsbook break-even
plus a margin of safety. Uses quarter-Kelly sizing clamped to 1-5.

The point of this model is selectivity: it abstains far more often than it
bets. Skipping when there is no edge is the cheapest profit-protection move
available — every coin-flip you don't take is the vig you don't pay.

The decision logic lives in `decide()` — a pure function with no I/O — so the
backtester can simulate it walk-forward against historical data.
"""
import pandas as pd
import numpy as np
import os
import json
import warnings
import psycopg2
from datetime import datetime
from dotenv import load_dotenv

warnings.filterwarnings('ignore')

np.random.seed(42)

load_dotenv()


TAG = "[CLC4_SELECTIVE]"

YEARS_FILES = {
    2026: "playerboxes/player_box_2026.csv",
    2025: "playerboxes/player_box_2025.csv",
    2024: "playerboxes/player_box_2024.csv",
    2023: "playerboxes/player_box_2023.csv",
    2022: "playerboxes/player_box_2022.csv",
    2021: "playerboxes/player_box_2021.csv",
    2020: "playerboxes/player_box_2020.csv",
    2019: "playerboxes/player_box_2019.csv",
    2018: "playerboxes/player_box_2018.csv",
    2017: "playerboxes/player_box_2017.csv",
    2016: "playerboxes/player_box_2016.csv",
    2015: "playerboxes/player_box_2015.csv",
    2014: "playerboxes/player_box_2014.csv",
    2013: "playerboxes/player_box_2013.csv",
    2012: "playerboxes/player_box_2012.csv",
    2011: "playerboxes/player_box_2011.csv",
    2010: "playerboxes/player_box_2010.csv",
    2009: "playerboxes/player_box_2009.csv",
}

SOURCE_MODELS = (
    "model_CL1", "model_CL2", "model_CL3_LEARN", "model_CH1_LEARN",
    "model_CLC1", "model_CLC2", "model_CLC3_LEARN",
)
MIN_SOURCE_PREDS = 3
MIN_CAREER_GAMES = 40
ROLLING_HISTORY_N = 100   # last N graded bets per source model
BREAK_EVEN = 0.524        # -110 / -110 sportsbook break-even
EV_MARGIN = 0.04          # require ensemble_prob >= break_even + this margin
LINE_GAP_THRESHOLD = 1.0  # under_line - over_line >= this triggers the split-line boost
LINE_GAP_BOOST = 0.03     # additive boost when the gap is favorable
QUARTER_KELLY = 0.25
KELLY_TO_STAKE_MULTIPLIER = 20  # f * 20 → integer 1-5 after rounding+clamp

_BOX_CACHE = {}
_DB_CONN = {"conn": None, "tried": False}
_WIN_RATES_CACHE = {"data": None}
_SOURCE_PRED_CACHE = {}


# ----------------------------------------------------------------------- odds

def _american_to_break_even(odds):
    if odds is None:
        odds = -110
    odds = float(odds)
    if odds > 0:
        return 100.0 / (odds + 100.0)
    return abs(odds) / (abs(odds) + 100.0)


def _american_to_decimal(odds):
    if odds is None:
        odds = -110
    odds = float(odds)
    if odds > 0:
        return (odds / 100.0) + 1.0
    return (100.0 / abs(odds)) + 1.0


# ----------------------------------------------------------------- pure logic

def decide(source_preds, source_win_rates, over_line, under_line, over_odds, under_odds):
    """
    Pure CLC4 decision logic. No I/O — used by both production predict() and
    the walk-forward backtester.

    Args:
        source_preds: {model_name: {"pred": float, "bet": "OVER"|"UNDER"}}
        source_win_rates: {model_name: float|None}  — rolling win rate;
                          None means "no history yet"; <0.50 gets weight 0.
        over_line, under_line: floats
        over_odds, under_odds: American odds (numeric or numeric-string)

    Returns:
        dict on bet: {"bet", "predicted_points", "note", "amount", "_diag"}
        dict on skip: {"_skip_reason": str}
    """
    if len(source_preds) < MIN_SOURCE_PREDS:
        return {"_skip_reason": f"only {len(source_preds)} source preds (<{MIN_SOURCE_PREDS})"}

    over_models = []
    under_models = []
    for model, p in source_preds.items():
        rate = source_win_rates.get(model)
        # Models with no track record or below 50% get weight zero — do NOT
        # anti-follow a losing model (it's likely just noise, not signal-flipped).
        if rate is None or rate < 0.50:
            continue
        weight = max(0.0, rate - BREAK_EVEN)
        if weight <= 0:
            continue
        if p["bet"] == "OVER":
            over_models.append((model, rate, weight))
        elif p["bet"] == "UNDER":
            under_models.append((model, rate, weight))

    over_w = sum(w for _, _, w in over_models)
    under_w = sum(w for _, _, w in under_models)

    if over_w == 0 and under_w == 0:
        return {"_skip_reason": "no source model with positive edge weighed in"}

    if over_w > under_w:
        direction = "OVER"
        voting = over_models
        odds_for_bet = over_odds
    elif under_w > over_w:
        direction = "UNDER"
        voting = under_models
        odds_for_bet = under_odds
    else:
        return {"_skip_reason": "weighted vote tied"}

    # Ensemble probability = weight-averaged win rate of models voting that side
    ensemble_prob = sum(r * w for _, r, w in voting) / sum(w for _, _, w in voting)

    # Split-line boost: when under_line - over_line >= 1.0 (e.g. O14.5 / U15.5)
    # the bookmaker has "split" the line. Both bets win on the middle integer
    # (a 15-point game wins both OVER 14.5 and UNDER 15.5). This is a structural
    # edge from line shopping across books — boost the probability estimate by
    # +0.03 to reflect it. Equal-line cases (over_line == under_line, no gap)
    # get no boost.
    if (under_line - over_line) >= LINE_GAP_THRESHOLD:
        ensemble_prob = min(0.99, ensemble_prob + LINE_GAP_BOOST)

    be = _american_to_break_even(odds_for_bet)
    margin = ensemble_prob - be
    if margin < EV_MARGIN:
        return {
            "_skip_reason": (
                f"edge insufficient: prob={ensemble_prob:.3f}, "
                f"break_even={be:.3f}, margin={margin:+.3f} < +{EV_MARGIN:.2f}"
            )
        }

    # Quarter-Kelly sizing. Full Kelly fraction = edge / (b-1) where b = decimal odds.
    decimal_odds = _american_to_decimal(odds_for_bet)
    edge = ensemble_prob * decimal_odds - 1.0
    if edge <= 0 or decimal_odds <= 1:
        return {"_skip_reason": "kelly edge non-positive"}
    f_kelly = QUARTER_KELLY * edge / (decimal_odds - 1.0)
    if f_kelly <= 0:
        return {"_skip_reason": "kelly fraction non-positive"}
    amount = round(f_kelly * KELLY_TO_STAKE_MULTIPLIER)
    if amount <= 0:
        return {"_skip_reason": "kelly stake rounds to 0"}
    amount = max(1, min(5, int(amount)))

    # Note tier based on margin above break-even
    if margin >= 0.10:
        note = "High Edge"
    elif margin >= 0.07:
        note = "Medium Edge"
    else:
        note = "Small Edge"

    # predicted_points = median of source point predictions (point estimate
    # for the schema only — the bet decision uses ensemble_prob, not this).
    preds_list = [p["pred"] for p in source_preds.values()]
    predicted_points = int(round(float(np.median(preds_list))))

    return {
        "bet": direction,
        "predicted_points": predicted_points,
        "note": note,
        "amount": amount,
        "_diag": {
            "ensemble_prob": ensemble_prob,
            "break_even": be,
            "margin": margin,
            "kelly_f": f_kelly,
            "n_voting": len(voting),
            "voting_models": [m for m, _, _ in voting],
        },
    }


# ---------------------------------------------------------------- I/O helpers

def _get_db():
    if _DB_CONN["tried"]:
        return _DB_CONN["conn"]
    _DB_CONN["tried"] = True
    try:
        conn = psycopg2.connect(
            dbname=os.getenv("DB_NAME"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            host=os.getenv("DB_HOST"),
            port=int(os.getenv("DB_PORT", 5432)),
        )
        _DB_CONN["conn"] = conn
        return conn
    except Exception:
        _DB_CONN["conn"] = None
        return None


def _load_playerboxes():
    if "df" in _BOX_CACHE:
        return _BOX_CACHE["df"]
    frames = []
    for year, path in YEARS_FILES.items():
        try:
            d = pd.read_csv(path)
            d["Year"] = year
            frames.append(d)
        except Exception:
            continue
    if not frames:
        empty = pd.DataFrame()
        _BOX_CACHE["df"] = empty
        return empty
    df = pd.concat(frames, ignore_index=True)
    df["did_not_play"] = df["did_not_play"].astype(str).str.upper().eq("TRUE")
    df = df[~df["did_not_play"]].copy()
    df = df.dropna(subset=["athlete_display_name", "game_date", "points"])
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
    df = df.dropna(subset=["game_date"])
    df["points"] = pd.to_numeric(df["points"], errors="coerce")
    df = df.dropna(subset=["points"])
    df = df.sort_values(["athlete_display_name", "game_date"]).reset_index(drop=True)
    _BOX_CACHE["df"] = df
    return df


def _source_win_rates(conn):
    """Rolling per-model win rate from the last ROLLING_HISTORY_N graded bets.
    Cached so we only query once per run."""
    if _WIN_RATES_CACHE["data"] is not None:
        return _WIN_RATES_CACHE["data"]
    rates = {}
    cur = conn.cursor()
    try:
        for model in SOURCE_MODELS:
            cur.execute(
                """
                SELECT result FROM predictions
                WHERE model_name = %s
                  AND result IN ('WON', 'LOST')
                ORDER BY date DESC
                LIMIT %s
                """,
                (model, ROLLING_HISTORY_N),
            )
            rows = cur.fetchall()
            if not rows:
                rates[model] = None
                continue
            wins = sum(1 for (r,) in rows if r == "WON")
            rates[model] = wins / len(rows)
    except Exception:
        pass
    finally:
        try:
            cur.close()
        except Exception:
            pass
    _WIN_RATES_CACHE["data"] = rates
    return rates


def _source_preds_today(conn, player_name, date_str):
    """Each source model's prediction for THIS player on THIS date.
    Reads predictions table. Per-INSERT commits in master_file.py ensure these
    rows are visible by the time CLC4 runs (CLC4 sorts last via SELECTIVE)."""
    key = (player_name.lower().strip(), str(date_str))
    if key in _SOURCE_PRED_CACHE:
        return _SOURCE_PRED_CACHE[key]
    out = {}
    cur = conn.cursor()
    try:
        cur.execute(
            """
            SELECT model_name, predicted_pts, bet
            FROM predictions
            WHERE LOWER(TRIM(player_name)) = LOWER(TRIM(%s))
              AND date = %s
              AND model_name = ANY(%s)
              AND predicted_pts IS NOT NULL
              AND bet IN ('OVER', 'UNDER')
            """,
            (player_name, date_str, list(SOURCE_MODELS)),
        )
        for model_name, pred_pts, bet in cur.fetchall():
            try:
                out[model_name] = {"pred": float(pred_pts), "bet": bet}
            except (TypeError, ValueError):
                continue
    except Exception:
        pass
    finally:
        try:
            cur.close()
        except Exception:
            pass
    _SOURCE_PRED_CACHE[key] = out
    return out


# ----------------------------------------------------------------- production

def predict(player):
    name = player["name"]
    date_str = player["date"]
    over_line = float(player["over_line"])
    under_line = float(player["under_line"])
    over_odds = player.get("over_odds")
    under_odds = player.get("under_odds")
    if over_odds in (None, ""):
        over_odds = -110
    if under_odds in (None, ""):
        under_odds = -110

    box_df = _load_playerboxes()
    if box_df.empty:
        print(TAG, f"{name} skipped [reason: no playerbox data]")
        return None

    target = pd.to_datetime(date_str)
    pdata = box_df[
        (box_df["athlete_display_name"] == name) & (box_df["game_date"] < target)
    ]
    career_n = len(pdata)
    if career_n < MIN_CAREER_GAMES:
        print(TAG, f"{name} skipped [reason: career_games={career_n} < {MIN_CAREER_GAMES}]")
        return None

    conn = _get_db()
    if conn is None:
        print(TAG, f"{name} skipped [reason: no DB connection]")
        return None

    preds = _source_preds_today(conn, name, date_str)
    if not preds:
        print(TAG, f"{name} skipped [reason: no source models predicted today]")
        return None
    if len(preds) < MIN_SOURCE_PREDS:
        print(TAG, f"{name} skipped [reason: only {len(preds)} source models predicted (<{MIN_SOURCE_PREDS})]")
        return None

    rates = _source_win_rates(conn)
    result = decide(preds, rates, over_line, under_line, over_odds, under_odds)

    if "_skip_reason" in result:
        print(TAG, f"{name} skipped [reason: {result['_skip_reason']}]")
        return None

    diag = result["_diag"]
    print(
        TAG,
        f"{name}: bet={result['bet']} amount=${result['amount']} "
        f"prob={diag['ensemble_prob']:.3f} margin={diag['margin']:+.3f} "
        f"voters={diag['n_voting']}/{len(preds)}"
    )

    return {
        "predicted_points": result["predicted_points"],
        "bet": result["bet"],
        "over_line": over_line,
        "under_line": under_line,
        "note": result["note"],
        "amount": result["amount"],
    }


if __name__ == "__main__":
    with open("props.json") as f:
        data = json.load(f)

    seen = set()
    for player in data["players"]:
        key = (player["name"], player["date"])
        if key in seen:
            continue
        seen.add(key)

        print(f"\n--- Running prediction for {player['name']} on {player['date']} ---")
        result = predict(player)
        if result is None:
            print(f"Prediction not generated for {player['name']}")
            continue
        print(f"Prediction successful for {player['name']}: {result['predicted_points']} pts")
        print(f"{player['name']} predicted points: {result['predicted_points']}")
        print(f"Bet: {result['bet']}, Over line: {result['over_line']}, Under line: {result['under_line']}")
        print(f"Amount: {result['amount']}, Note: {result['note']}")
