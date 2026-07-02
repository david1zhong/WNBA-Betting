"""
CLCF1 — a continuously-retrained logistic-regression market model.

Every morning it refits a small regularized logistic regression on ALL graded
history in the predictions table (deduplicated to one market row per
player-date, so it learns from outcomes, not from other models' opinions),
then prices today's slate and bets only when the modeled probability clears
the sportsbook break-even by a calibrated margin.

Features (chosen by walk-forward ablation on ~1,800 graded player-dates;
each survived a margin x regularization robustness grid):
  gap         under_line - over_line. Split lines (gap >= 1) are a structural
              middle: UNDER at the higher number won 56.7% historically.
  juice_asym  break-even(under_odds) - break-even(over_odds). Where the book
              loads the vig is where the sharp side is.
  player_post the player's historical under-rate, shrunk hard toward the
              global base rate (player persistence is weak: split-half
              correlation ~0.09, so BETA_K keeps this feature honest).
  mid         line level (high lines behave differently from low lines).
  min_trend   last-5 vs last-15 average minutes — role expanding/shrinking.
  form_edge   recency-weighted last-15 scoring average minus the line.

Decision rule: EV-gate BOTH sides against their actual odds, require
prob - break_even >= EV_MARGIN, size with quarter-Kelly clamped to 1-5.
Abstaining is a feature: the walk-forward backtest made +9% ROI betting
~570 of ~1,550 opportunities and was profitable in every month tested.

Numpy-only training (deterministic gradient descent, no sklearn, no seed
sensitivity). Walk-forward by construction in production: each day's fit
only ever sees already-graded games.
"""
import os
import json
import warnings
from urllib.parse import urlparse, unquote

import numpy as np
import pandas as pd
import psycopg2
from dotenv import load_dotenv

warnings.filterwarnings("ignore")

load_dotenv()

TAG = "[CLCF1]"

YEARS_FILES = {
    year: f"playerboxes/player_box_{year}.csv" for year in range(2009, 2027)
}

FEATURES = ("gap", "juice_asym", "player_post", "mid", "min_trend", "form_edge")

EV_MARGIN = 0.03          # required prob edge over break-even (calibrated)
L2 = 2.0                  # ridge strength for the logistic fit
GD_ITERS = 300
GD_LR = 0.5
BETA_K = 12.0             # shrinkage strength for player_post
QUARTER_KELLY = 0.25
KELLY_TO_STAKE_MULTIPLIER = 20   # f * 20 -> integer stake, clamped 1-5
MIN_TRAIN_ROWS = 250      # refuse to bet on a thin training set
MIN_CAREER_GAMES = 10     # need real box history for form/minutes features
FORM_GAMES = 15           # lookback for form_edge
FORM_DECAY = 0.85         # recency weight per game (newest weighted most)
MIN_SHORT = 5             # minutes-trend short window
MIN_LONG = 15             # minutes-trend long window

_BOX_CACHE = {}
_DB_CONN = {"conn": None, "tried": False}
_MODEL_CACHE = {"fit": None, "tried": False}


# ------------------------------------------------------------------ odds math

def _to_odds(v, default=-110.0):
    """American odds from numbers or strings; '' / None / 'even' handled."""
    if v is None:
        return default
    if isinstance(v, (int, float)):
        return float(v) if not pd.isna(v) else default
    s = str(v).strip()
    if not s:
        return default
    if s.lower() in ("even", "ev", "evens", "pk", "pick"):
        return 100.0
    try:
        return float(s.lstrip("+"))
    except (TypeError, ValueError):
        return default


def _break_even(odds):
    odds = float(odds)
    if odds > 0:
        return 100.0 / (odds + 100.0)
    return abs(odds) / (abs(odds) + 100.0)


def _decimal(odds):
    odds = float(odds)
    if odds > 0:
        return (odds / 100.0) + 1.0
    return (100.0 / abs(odds)) + 1.0


# ------------------------------------------------------------------- data I/O

def _get_db():
    if _DB_CONN["tried"]:
        return _DB_CONN["conn"]
    _DB_CONN["tried"] = True
    try:
        if os.getenv("DB_NAME"):
            conn = psycopg2.connect(
                dbname=os.getenv("DB_NAME"),
                user=os.getenv("DB_USER"),
                password=os.getenv("DB_PASSWORD"),
                host=os.getenv("DB_HOST"),
                port=int(os.getenv("DB_PORT", 5432)),
            )
        elif os.getenv("DATABASE_URL"):
            u = urlparse(os.getenv("DATABASE_URL"))
            conn = psycopg2.connect(
                dbname=u.path.lstrip("/"),
                user=unquote(u.username or ""),
                password=unquote(u.password or ""),
                host=u.hostname,
                port=u.port or 5432,
            )
        else:
            conn = None
        _DB_CONN["conn"] = conn
    except Exception:
        _DB_CONN["conn"] = None
    return _DB_CONN["conn"]


def _load_playerboxes():
    if "df" in _BOX_CACHE:
        return _BOX_CACHE["df"]
    frames = []
    for year, path in YEARS_FILES.items():
        try:
            d = pd.read_csv(path, low_memory=False)
            frames.append(d)
        except Exception:
            continue
    if not frames:
        _BOX_CACHE["df"] = pd.DataFrame()
        return _BOX_CACHE["df"]
    df = pd.concat(frames, ignore_index=True)
    df = df[df["did_not_play"].astype(str).str.upper() != "TRUE"].copy()
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
    df["points"] = pd.to_numeric(df["points"], errors="coerce")
    df["minutes"] = pd.to_numeric(df["minutes"], errors="coerce")
    df = df.dropna(subset=["athlete_display_name", "game_date", "points"])
    df["name_key"] = df["athlete_display_name"].str.lower().str.strip()
    df = df.sort_values("game_date").reset_index(drop=True)
    _BOX_CACHE["df"] = df
    return df


def _box_groups():
    if "groups" not in _BOX_CACHE:
        box = _load_playerboxes()
        _BOX_CACHE["groups"] = (
            {k: g.reset_index(drop=True) for k, g in box.groupby("name_key")}
            if not box.empty else {}
        )
    return _BOX_CACHE["groups"]


def _load_history(conn):
    """One graded market row per (player, date): the outcome vs the line.
    Any model's row works — lines/odds/actuals are identical across models
    for the same player-date because they come from the same props scrape."""
    df = pd.read_sql(
        """
        SELECT DISTINCT ON (player_name, date)
               player_name, date, actual_pts, over_line, under_line,
               over_odds, under_odds
        FROM predictions
        WHERE result IN ('WON', 'LOST')
          AND actual_pts IS NOT NULL
          AND over_line IS NOT NULL AND under_line IS NOT NULL
        ORDER BY player_name, date
        """,
        conn,
    )
    df["date_dt"] = pd.to_datetime(df["date"])
    df = df.sort_values("date_dt").reset_index(drop=True)
    return df


# ------------------------------------------------------------------- features

def _box_features(name_key, target_dt):
    """(min_trend, form, career_games) from box games strictly before
    target_dt. form is the recency-weighted scoring average; the caller
    subtracts the line to get form_edge."""
    grp = _box_groups().get(name_key)
    if grp is None:
        return 0.0, None, 0
    hist = grp[grp["game_date"] < target_dt]
    n = len(hist)
    if n == 0:
        return 0.0, None, 0
    mins = hist["minutes"].dropna()
    m_short = mins.tail(MIN_SHORT).mean()
    m_long = mins.tail(MIN_LONG).mean()
    min_trend = float(m_short - m_long) if pd.notna(m_short) and pd.notna(m_long) else 0.0
    pts = hist["points"].tail(FORM_GAMES).to_numpy(dtype=float)
    wts = FORM_DECAY ** np.arange(len(pts) - 1, -1, -1)
    form = float((pts * wts).sum() / wts.sum())
    return min_trend, form, n


def _build_training_matrix(hist):
    """Feature matrix + labels from the graded market history. player_post is
    computed cumulatively (each row only sees results from earlier rows), so
    the fit stays walk-forward-honest even within the training set."""
    h = hist.copy()
    h["under_win"] = (h["actual_pts"] < h["under_line"]).astype(float)
    h["gap"] = h["under_line"] - h["over_line"]
    h["mid"] = (h["under_line"] + h["over_line"]) / 2.0
    h["u_be"] = h["under_odds"].apply(lambda v: _break_even(_to_odds(v)))
    h["o_be"] = h["over_odds"].apply(lambda v: _break_even(_to_odds(v)))
    h["juice_asym"] = h["u_be"] - h["o_be"]

    global_under = float(h["under_win"].mean())
    h["name_key"] = h["player_name"].str.lower().str.strip()
    cum_wins = h.groupby("name_key")["under_win"].cumsum() - h["under_win"]
    cum_n = h.groupby("name_key").cumcount()
    h["player_post"] = (global_under * BETA_K + cum_wins) / (BETA_K + cum_n)

    min_trends, form_edges = [], []
    for _, r in h.iterrows():
        mt, form, _n = _box_features(r["name_key"], r["date_dt"])
        min_trends.append(mt)
        form_edges.append((form - r["mid"]) if form is not None else 0.0)
    h["min_trend"] = min_trends
    h["form_edge"] = form_edges

    X = h[list(FEATURES)].to_numpy(dtype=float)
    y = h["under_win"].to_numpy(dtype=float)
    return X, y, global_under, h


def _fit_logistic(X, y):
    """Deterministic ridge-regularized logistic regression."""
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd < 1e-9] = 1.0
    Xs = (X - mu) / sd
    n, _k = Xs.shape
    w = np.zeros(Xs.shape[1])
    b = 0.0
    for _ in range(GD_ITERS):
        z = np.clip(Xs @ w + b, -30, 30)
        p = 1.0 / (1.0 + np.exp(-z))
        w -= GD_LR * (Xs.T @ (p - y) / n + L2 * w / n)
        b -= GD_LR * float((p - y).mean())
    return {"w": w, "b": b, "mu": mu, "sd": sd}


def _get_fitted():
    """Train once per process on all graded history. Returns dict with the
    fit, the global under rate, and per-player (wins, n) records — or None."""
    if _MODEL_CACHE["tried"]:
        return _MODEL_CACHE["fit"]
    _MODEL_CACHE["tried"] = True

    conn = _get_db()
    if conn is None:
        return None
    try:
        hist = _load_history(conn)
    except Exception:
        return None
    if len(hist) < MIN_TRAIN_ROWS:
        return None

    X, y, global_under, h = _build_training_matrix(hist)
    fit = _fit_logistic(X, y)
    player_rec = h.groupby("name_key")["under_win"].agg(["sum", "size"])
    _MODEL_CACHE["fit"] = {
        "fit": fit,
        "global_under": global_under,
        "player_rec": {k: (float(r["sum"]), int(r["size"]))
                       for k, r in player_rec.iterrows()},
        "n_train": len(h),
    }
    return _MODEL_CACHE["fit"]


def _prob_under(model, x_row):
    f = model["fit"]
    xs = (np.asarray(x_row, dtype=float) - f["mu"]) / f["sd"]
    z = float(np.clip(xs @ f["w"] + f["b"], -30, 30))
    return 1.0 / (1.0 + np.exp(-z))


# ----------------------------------------------------------------- pure logic

def decide(p_under, over_line, under_line, over_odds, under_odds):
    """EV-gate both sides at their actual odds; quarter-Kelly stake 1-5.
    Pure function — no I/O. Returns a bet dict or {'_skip_reason': ...}."""
    over_odds = _to_odds(over_odds)
    under_odds = _to_odds(under_odds)

    candidates = (
        ("UNDER", p_under, under_odds),
        ("OVER", 1.0 - p_under, over_odds),
    )
    best = None
    for side, prob, odds in candidates:
        margin = prob - _break_even(odds)
        if margin < EV_MARGIN:
            continue
        if best is None or margin > best[3]:
            best = (side, prob, odds, margin)

    if best is None:
        u_m = p_under - _break_even(under_odds)
        o_m = (1.0 - p_under) - _break_even(over_odds)
        return {"_skip_reason": (
            f"no edge: p_under={p_under:.3f}, "
            f"under margin={u_m:+.3f}, over margin={o_m:+.3f} (< +{EV_MARGIN:.2f})")}

    side, prob, odds, margin = best
    dec_odds = _decimal(odds)
    edge = prob * dec_odds - 1.0
    if edge <= 0:
        return {"_skip_reason": "kelly edge non-positive"}
    f_kelly = QUARTER_KELLY * edge / (dec_odds - 1.0)
    amount = int(max(1, min(5, round(f_kelly * KELLY_TO_STAKE_MULTIPLIER))))

    if margin >= 0.10:
        note = "High Edge"
    elif margin >= 0.06:
        note = "Medium Edge"
    else:
        note = "Small Edge"

    return {"bet": side, "prob": prob, "margin": margin,
            "amount": amount, "note": note}


# ----------------------------------------------------------------- production

def predict(player):
    name = player["name"]
    date_str = player["date"]
    try:
        over_line = float(player["over_line"])
        under_line = float(player["under_line"])
    except (TypeError, ValueError, KeyError):
        print(TAG, f"{name} skipped [reason: bad line values]")
        return None
    over_odds = _to_odds(player.get("over_odds"))
    under_odds = _to_odds(player.get("under_odds"))

    name_key = name.lower().strip()
    target_dt = pd.to_datetime(date_str)

    min_trend, form, career_n = _box_features(name_key, target_dt)
    if career_n < MIN_CAREER_GAMES or form is None:
        print(TAG, f"{name} skipped [reason: career_games={career_n} < {MIN_CAREER_GAMES}]")
        return None

    model = _get_fitted()
    if model is None:
        print(TAG, f"{name} skipped [reason: no DB connection or training history]")
        return None

    gap = under_line - over_line
    mid = (under_line + over_line) / 2.0
    juice_asym = _break_even(under_odds) - _break_even(over_odds)
    wins, n = model["player_rec"].get(name_key, (0.0, 0))
    player_post = (model["global_under"] * BETA_K + wins) / (BETA_K + n)
    form_edge = form - mid

    x = [gap, juice_asym, player_post, mid, min_trend, form_edge]
    p_under = _prob_under(model, x)

    result = decide(p_under, over_line, under_line, over_odds, under_odds)
    if "_skip_reason" in result:
        print(TAG, f"{name} skipped [reason: {result['_skip_reason']}]")
        return None

    # predicted_points: recency-weighted form, clamped to the bet's side of
    # the line so the stored prediction is never inconsistent with the bet.
    predicted = int(round(form))
    if result["bet"] == "UNDER":
        predicted = min(predicted, int(np.floor(under_line)))
    else:
        predicted = max(predicted, int(np.ceil(over_line)))

    print(
        TAG,
        f"{name}: bet={result['bet']} amount=${result['amount']} "
        f"p_under={p_under:.3f} margin={result['margin']:+.3f} "
        f"form={form:.1f} line={mid} (trained on {model['n_train']} rows)"
    )

    return {
        "predicted_points": predicted,
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
            print(f"No bet for {player['name']}")
            continue
        print(f"{player['name']}: {result['predicted_points']} pts, "
              f"bet {result['bet']} (O{result['over_line']}/U{result['under_line']}), "
              f"amount {result['amount']}, note {result['note']}")
