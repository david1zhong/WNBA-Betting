import pandas as pd
import numpy as np
import math
import os
import json
import warnings
import psycopg2
from datetime import datetime
from dotenv import load_dotenv


TAG = "[CLC3_LEARN]"
warnings.filterwarnings('ignore')

np.random.seed(42)

load_dotenv()


YEARS_FILES = {
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

ALL_SOURCE_MODELS = (
    "model_CL1", "model_CL2", "model_CL3_LEARN", "model_CH1_LEARN",
    "model_CLC1", "model_CLC2",
)
ESTABLISHED_MODELS = ("model_CL1", "model_CL2", "model_CL3_LEARN", "model_CH1_LEARN")
NEW_MODELS = ("model_CLC1", "model_CLC2")
PHASE_IN_MIN_PREDS = 20

MIN_CAREER_GAMES = 20
MIN_CAL_SAMPLES = 5
LOW_OUTPUT_SD = 1.5
LOW_OUTPUT_CONF = 0.85

_BOX_CACHE = {}
_DB_CONN = {"conn": None, "tried": False}
_GLOBAL_COUNTS = {"data": None}
_SOURCE_PRED_CACHE = {}


def _normal_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


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


def _global_model_counts(conn, models):
    """Total graded predictions per model across ALL players (cached)."""
    if _GLOBAL_COUNTS["data"] is not None:
        return _GLOBAL_COUNTS["data"]
    cur = conn.cursor()
    try:
        cur.execute(
            "SELECT model_name, COUNT(*) FROM predictions "
            "WHERE model_name = ANY(%s) AND actual_pts IS NOT NULL "
            "GROUP BY model_name",
            (list(models),),
        )
        result = dict(cur.fetchall())
    except Exception:
        result = {}
    finally:
        try:
            cur.close()
        except Exception:
            pass
    _GLOBAL_COUNTS["data"] = result
    return result


def _get_source_predictions(player):
    """Read today's per-model predictions from Supabase.

    master_file.py runs base models before LEARN models, so by the time CLC3_LEARN
    executes, today's predictions from CL1/CL2/CLC1/CLC2/CH1_LEARN/CL3_LEARN are
    already in the predictions table for this date.
    """
    cache_key = (player.get("name"), player.get("date"))
    if cache_key in _SOURCE_PRED_CACHE:
        return _SOURCE_PRED_CACHE[cache_key]

    out = {}
    conn = _get_db()
    if conn is None:
        _SOURCE_PRED_CACHE[cache_key] = out
        return out

    cur = conn.cursor()
    try:
        cur.execute(
            """
            SELECT model_name, predicted_pts
            FROM predictions
            WHERE LOWER(TRIM(player_name)) = LOWER(TRIM(%s))
              AND date = %s
              AND model_name = ANY(%s)
              AND predicted_pts IS NOT NULL
            """,
            (player["name"], player["date"], list(ALL_SOURCE_MODELS)),
        )
        for model_name, pp in cur.fetchall():
            try:
                out[model_name] = float(pp)
            except (TypeError, ValueError):
                continue
    except Exception:
        pass
    finally:
        try:
            cur.close()
        except Exception:
            pass

    _SOURCE_PRED_CACHE[cache_key] = out
    return out


def _fetch_player_predictions(conn, player_name):
    cur = conn.cursor()
    try:
        sql = """
            SELECT model_name, date, predicted_pts, actual_pts,
                   over_line, under_line, bet, result, profit
            FROM predictions
            WHERE LOWER(TRIM(player_name)) = LOWER(TRIM(%s))
              AND model_name = ANY(%s)
            ORDER BY date ASC
        """
        cur.execute(sql, (player_name, list(ALL_SOURCE_MODELS)))
        rows = cur.fetchall()
        cols = [d[0] for d in cur.description]
        return pd.DataFrame(rows, columns=cols)
    except Exception:
        return pd.DataFrame()
    finally:
        try:
            cur.close()
        except Exception:
            pass


def _player_features_at(box_df, player_name, target_date):
    """Features observable at the moment a prediction was made (no leakage)."""
    target = pd.to_datetime(target_date)
    pdf = box_df[(box_df["athlete_display_name"] == player_name) &
                 (box_df["game_date"] < target)]
    if pdf.empty:
        return None
    pts = pdf["points"].values.astype(float)
    overall = float(np.mean(pts))
    rec5 = float(np.mean(pts[-5:])) if len(pts) >= 1 else overall
    recent_form = (rec5 - overall) / max(1.0, overall)

    today_row = box_df[(box_df["athlete_display_name"] == player_name) &
                       (box_df["game_date"] == target)]
    if not today_row.empty:
        ha = str(today_row.iloc[0].get("home_away", "")).lower()
        home = 1.0 if ha == "home" else (0.0 if ha == "away" else 0.5)
    else:
        home_split = (pdf["home_away"].astype(str).str.lower() == "home").mean()
        home = float(home_split) if pd.notna(home_split) else 0.5

    return {
        "recent_form": float(recent_form),
        "home": float(home),
        "rec5": float(rec5),
        "career_mean": overall,
        "career_std": float(pdf["points"].std(ddof=0)) if len(pdf) > 1 else 5.0,
        "n_games": len(pdf),
    }


def _ridge_solve(X, y, w, alpha):
    """Weighted ridge regression. Don't penalize the intercept (col 0)."""
    Wsq = np.diag(w)
    A = X.T @ Wsq @ X
    reg = alpha * np.eye(X.shape[1])
    reg[0, 0] = 0.0
    try:
        return np.linalg.solve(A + reg, X.T @ Wsq @ y)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(X * np.sqrt(w)[:, None], y * np.sqrt(w), rcond=None)[0]


def _calibrate_source(source_df, box_df, player_name):
    """Conditional, profit-weighted calibration for a single source model."""
    if source_df.empty:
        return None

    df = source_df.dropna(subset=["actual_pts", "predicted_pts"]).copy()
    if df.empty:
        return None

    df["actual_pts"] = pd.to_numeric(df["actual_pts"], errors="coerce")
    df["predicted_pts"] = pd.to_numeric(df["predicted_pts"], errors="coerce")
    df = df.dropna(subset=["actual_pts", "predicted_pts"])
    if len(df) < MIN_CAL_SAMPLES:
        return None

    rows = []
    for _, r in df.iterrows():
        feats = _player_features_at(box_df, player_name, r["date"])
        if feats is None:
            continue
        profit = r["profit"]
        try:
            profit = float(profit) if pd.notna(profit) else 0.0
        except Exception:
            profit = 0.0
        rows.append({
            "pred": float(r["predicted_pts"]),
            "recent_form": feats["recent_form"],
            "home": feats["home"],
            "actual": float(r["actual_pts"]),
            "profit": profit,
        })

    if len(rows) < MIN_CAL_SAMPLES:
        return None

    fdf = pd.DataFrame(rows)

    X = np.column_stack([
        np.ones(len(fdf)),
        fdf["pred"].values,
        fdf["recent_form"].values,
        fdf["home"].values,
        fdf["pred"].values * fdf["recent_form"].values,
    ])
    y = fdf["actual"].values
    w = np.maximum(1.0 + fdf["profit"].values, 0.1)

    n_train = len(fdf)
    alpha = 1.5 + max(0.0, (40 - n_train) / 40.0) * 4.0
    beta = _ridge_solve(X, y, w, alpha)

    pred_cal = X @ beta
    residuals = y - pred_cal
    weighted_rmse = float(np.sqrt(np.average(residuals ** 2, weights=w)))
    resid_std = float(np.std(residuals, ddof=0))

    profit_per_bet = float(fdf["profit"].mean()) if (fdf["profit"] != 0).any() else 0.0

    return {
        "beta": beta,
        "weighted_rmse": weighted_rmse,
        "residual_std": resid_std,
        "n_train": n_train,
        "profit_per_bet": profit_per_bet,
    }


def _calibrated_predict(cal, baseline_pred, recent_form, home):
    """Apply learned calibration. baseline_pred is the source model's today prediction."""
    if cal is None:
        return None
    x = np.array([
        1.0,
        baseline_pred,
        recent_form,
        home,
        baseline_pred * recent_form,
    ], dtype=float)
    return float(x @ cal["beta"])


def _confidence_to_amount(confidence):
    if confidence >= 0.70:
        return 5
    if confidence >= 0.62:
        return 3
    if confidence >= 0.55:
        return 1
    return None


def _categorize(predicted, career_mean):
    if career_mean <= 0:
        return "Average Game"
    dev = (predicted - career_mean) / career_mean
    if dev <= -0.20:
        return "Very Bad Game"
    if dev <= -0.08:
        return "Below Average"
    if dev < 0.05:
        return "Average Game"
    if dev < 0.15:
        return "Good Game"
    return "Very Good Game"


def _detect_monthly_dip(pdata, target_date, career_mean):
    if career_mean <= 0:
        return False
    target = pd.to_datetime(target_date)
    history = pdata[pdata["game_date"] < target].copy()
    if history.empty:
        return False
    history["ym"] = history["game_date"].dt.to_period("M")
    last_months = sorted(history["ym"].unique())[-3:]
    if len(last_months) < 2:
        return False
    threshold = career_mean * 0.80
    months_with_dip = 0
    for ym in last_months:
        m = history[history["ym"] == ym]
        nearby = m[(m["game_date"].dt.day - target.day).abs() <= 3]
        if not nearby.empty and (nearby["points"] < threshold).any():
            months_with_dip += 1
    return months_with_dip >= 2


def predict(player):
    name = player["name"]
    date_str = player["date"]
    over_line = float(player["over_line"])
    under_line = float(player["under_line"])

    box_df = _load_playerboxes()
    if box_df.empty:
        print(TAG, f"{name} not in dip results [exit: no playerbox CSV data]")
        return None

    pdata = box_df[box_df["athlete_display_name"] == name].copy()
    target = pd.to_datetime(date_str)
    pdata = pdata[pdata["game_date"] < target].copy()
    if len(pdata) < MIN_CAREER_GAMES:
        print(TAG, f"{name} not in dip results [exit: career_games={len(pdata)} < {MIN_CAREER_GAMES}]")
        return None

    feats = _player_features_at(box_df, name, date_str)
    if feats is None:
        print(TAG, f"{name} not in dip results [exit: features unavailable]")
        return None

    conn = _get_db()
    if conn is None:
        print(TAG, f"{name} not in dip results [exit: no DB connection]")
        return None

    pred_df = _fetch_player_predictions(conn, name)
    if pred_df.empty:
        print(TAG, f"{name} not in dip results [exit: no historical predictions for player]")
        return None

    # Phase-in: check GLOBAL graded prediction counts (across all players),
    # not per-player. CLC1/CLC2 are mature once they have >=20 graded predictions
    # in the league as a whole, even if this specific player has fewer.
    global_counts = _global_model_counts(conn, NEW_MODELS)
    use_models = list(ESTABLISHED_MODELS)
    for m in NEW_MODELS:
        if global_counts.get(m, 0) >= PHASE_IN_MIN_PREDS:
            use_models.append(m)

    # Each source model's today prediction (read from Supabase). The calibration
    # was trained on each model's predicted_pts, so feeding a shared surrogate
    # would collapse the meta-learner.
    today_source_preds = _get_source_predictions(player)

    contributions = []
    for model in use_models:
        if model not in today_source_preds:
            continue  # source model didn't generate a prediction for today
        sub = pred_df[pred_df["model_name"] == model]
        cal = _calibrate_source(sub, box_df, name)
        if cal is None:
            continue
        source_pred_today = today_source_preds[model]
        adj = _calibrated_predict(cal, source_pred_today, feats["recent_form"], feats["home"])
        if adj is None or not np.isfinite(adj):
            continue

        # Weight each source by inverse RMSE × profit factor.
        # Profit-weighting is the optimization target: a -$2/bet model gets ~0.40 weight;
        # a +$1/bet model gets 1.30. Floor at 0.10 keeps even chronic losers' calibration
        # signal in the mix without letting them dominate.
        rmse_score = 1.0 / max(1.0, cal["weighted_rmse"])
        profit_factor = max(0.10, 1.0 + cal["profit_per_bet"] * 0.30)
        weight = rmse_score * profit_factor

        contributions.append({
            "model": model,
            "pred": float(adj),
            "weight": float(weight),
            "residual_std": float(cal["residual_std"]),
            "n_train": cal["n_train"],
            "profit_per_bet": cal["profit_per_bet"],
        })

    if not contributions:
        n_today_preds = len(today_source_preds)
        print(
            TAG,
            f"{name} not in dip results "
            f"[exit: no source-model contributions today; "
            f"today_source_preds={n_today_preds} models, "
            f"use_models={len(use_models)}, "
            f"global_counts={global_counts}]",
        )
        return None

    weights = np.array([c["weight"] for c in contributions], dtype=float)
    preds = np.array([c["pred"] for c in contributions], dtype=float)
    weights = weights / weights.sum()

    predicted_raw = float(np.sum(weights * preds))
    predicted_points = int(round(predicted_raw))

    # Confidence: combine source-model agreement with distance from line under
    # the predicted distribution (P(pick correct)). Profit-weighted residual std
    # supplies the spread; agreement std of source contributions adds disagreement risk.
    weighted_resid_std = float(np.sqrt(np.sum(
        weights * np.array([c["residual_std"] ** 2 for c in contributions])
    )))
    agreement_std = float(np.sqrt(np.sum(weights * (preds - predicted_raw) ** 2)))
    sigma = max(2.0, weighted_resid_std + 0.5 * agreement_std)

    z = (over_line - predicted_raw) / sigma
    p_under = _normal_cdf(z)
    p_over = 1.0 - p_under
    confidence = float(min(0.95, max(0.0, max(p_over, p_under))))

    # Self-skip: drop if calibrated meta-confidence is too low
    if confidence < 0.55:
        print(
            TAG,
            f"{name} not in dip results "
            f"[exit: confidence={confidence:.3f} < 0.55; "
            f"predicted={predicted_raw:.2f}, line={over_line}, "
            f"sigma={sigma:.2f}, n_contrib={len(contributions)}]",
        )
        return None

    bet = "OVER" if predicted_points > over_line else "UNDER"

    note = _categorize(predicted_raw, feats["career_mean"])

    rec10 = pdata.tail(10)["points"]
    rec_mean = float(rec10.mean()) if len(rec10) >= 3 else feats["career_mean"]
    rec_std = float(rec10.std(ddof=0)) if len(rec10) >= 3 else feats["career_std"]
    if rec_std and rec_std > 0:
        sd_below = (rec_mean - predicted_raw) / rec_std
        if (sd_below >= LOW_OUTPUT_SD
                and confidence >= LOW_OUTPUT_CONF
                and _detect_monthly_dip(pdata, date_str, feats["career_mean"])):
            note = "Low Output"

    amount = _confidence_to_amount(confidence)

    return {
        "predicted_points": predicted_points,
        "bet": bet,
        "over_line": over_line,
        "under_line": under_line,
        "note": note,
        "amount": amount,
    }

"""
if __name__ == "__main__":
    with open("props.json") as f:
        data = json.load(f)

    seen = set()
    for player in data["players"]:
        key = (player["name"], player["date"])
        if key in seen:
            continue
        seen.add(key)

        print(TAG, f"\n--- Running prediction for {player['name']} on {player['date']} ---")
        result = predict(player)
        if result is None:
            print(TAG, f"Prediction not generated for {player['name']}")
            continue
        print(TAG, f"Prediction successful for {player['name']}: {result['predicted_points']} pts")
        print(TAG, f"{player['name']} predicted points: {result['predicted_points']}")
        print(TAG, f"Bet: {result['bet']}, Over line: {result['over_line']}, Under line: {result['under_line']}")
"""
