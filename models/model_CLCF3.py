"""
CLCF3 — the project's core premise (cyclical low-output games), made rigorous.

CL2 detects roughly-monthly performance dips in a player's box history and
bets UNDER inside the predicted dip window. It was profitable in 2025 but on
a small, unpriced sample: it never looked at the odds, never calibrated its
confidence against outcomes, and its dip math wasn't walk-forward-safe
(z-scores were computed over the whole career, future games included).

CLCF3 keeps the premise and rebuilds the machinery in the CLCF mold:

Signal (per player, box games strictly before the target date only):
  - performance score = mean of z-scored points and FG%, dips = bottom 25%
  - intervals between consecutive dips in the 20-40 day band define the
    player's personal cycle (need >= 3 such intervals)
  - the next dip is projected k full cycles after the last observed dip
    (k >= 1, so a long dip drought wraps forward instead of pointing at a
    date in the past — a bug-fix over CL2, which only looked one cycle out)
  - dip_score = gaussian proximity of the target date to that projection
  - dip_gap  = recency-weighted form minus mean points in past dip games
    (how much this player actually drops when she dips)

Calibration: a ridge logistic regression (numpy, deterministic) trained on
all graded market history in the predictions DB — one row per player-date,
walk-forward honest — maps the dip signal plus market structure (line gap,
juice asymmetry, line level, player under-rate, form edge) to P(under).

Decision rule — UNDER ONLY, dip-gated. This model exists to call low-output
games, so it never takes the OVER:
  - a personal cycle must exist (>= 3 monthly intervals)
  - dip_score >= DIP_GATE and dip_gap > MIN_DIP_GAP (the projected dip is
    near AND this player's dips are real scoring drops)
  - calibrated P(under) must clear the under's break-even by EV_MARGIN
  - quarter-Kelly stake clamped to 1-5

Walk-forward backtest on the graded history through 2026-07-07 (refit each
morning, features from prior games only): 165 bets, 96-69 (58.2%), +6.8% ROI
at the actual under odds, positive in 3 of the 5 months covered. Small
sample, like CL2's — the premise bets are ~10% of the slate by design.
"""
import os
import json
import warnings
from datetime import datetime
from urllib.parse import urlparse, unquote

import numpy as np
import pandas as pd
import psycopg2
from dotenv import load_dotenv

warnings.filterwarnings("ignore")

load_dotenv()

TAG = "[CLCF3]"

YEARS_FILES = {
    year: f"playerboxes/player_box_{year}.csv"
    for year in range(2009, datetime.now().year + 1)
}

FEATURES = ("gap", "juice_asym", "player_post", "mid",
            "form_edge", "dip_signal", "dip_drop")

EV_MARGIN = 0.03          # required P(under) edge over the under break-even
DIP_GATE = 0.2            # minimum dip_score to consider a bet
MIN_DIP_GAP = 2.0         # past dips must average >2 pts below form
L2 = 2.0                  # ridge strength for the logistic fit
GD_ITERS = 300
GD_LR = 0.5
BETA_K = 12.0             # shrinkage strength for player_post
QUARTER_KELLY = 0.25
KELLY_TO_STAKE_MULTIPLIER = 20   # f * 20 -> integer stake, clamped 1-5
MIN_TRAIN_ROWS = 250      # refuse to bet on a thin training set
MIN_CAREER_GAMES = 15     # dip stats need a real career sample
FORM_GAMES = 15           # lookback for form
FORM_DECAY = 0.85         # recency weight per game (newest weighted most)
DIP_QUANTILE = 0.25       # bottom 25% of performance scores are dips
CYCLE_LO_DAYS = 20        # monthly-cycle interval band
CYCLE_HI_DAYS = 40
MIN_CYCLES = 3            # intervals in band needed to trust the cycle
MIN_CYCLE_SIGMA = 2.5     # floor on the dip-window width (days)

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
    for c in ("points", "field_goals_made", "field_goals_attempted"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
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

def _cycle_features(name_key, target_dt):
    """Dip-cycle stats from box games strictly before target_dt.

    Returns (dip_score, dip_gap, cycle_quality, form, career_n), or None if
    the career is too thin to say anything. dip_score/cycle_quality are 0.0
    when no usable monthly cycle exists."""
    grp = _box_groups().get(name_key)
    if grp is None:
        return None
    hist = grp[grp["game_date"] < target_dt]
    n = len(hist)
    if n < MIN_CAREER_GAMES:
        return None

    pts_all = hist["points"].to_numpy(dtype=float)
    pts = pts_all[-FORM_GAMES:]
    wts = FORM_DECAY ** np.arange(len(pts) - 1, -1, -1)
    form = float((pts * wts).sum() / wts.sum())

    fga = hist["field_goals_attempted"].to_numpy(dtype=float)
    fgm = hist["field_goals_made"].to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        fg_pct = np.where(fga > 0, fgm / fga, 0.0)
    fg_pct = np.nan_to_num(fg_pct)

    def _z(a):
        s = a.std()
        return (a - a.mean()) / s if s > 1e-9 else np.zeros_like(a)

    perf = (_z(pts_all) + _z(fg_pct)) / 2.0
    thr = np.quantile(perf, DIP_QUANTILE)
    dip_mask = perf < thr
    dip_dates = hist["game_date"].to_numpy()[dip_mask]
    if len(dip_dates) < MIN_CYCLES:
        return 0.0, 0.0, 0.0, form, n

    dip_gap = float(form - pts_all[dip_mask].mean())

    intervals = np.diff(dip_dates).astype("timedelta64[D]").astype(float)
    monthly = intervals[(intervals >= CYCLE_LO_DAYS) & (intervals <= CYCLE_HI_DAYS)]
    if len(monthly) < MIN_CYCLES:
        return 0.0, dip_gap, 0.0, form, n

    avg_cycle = float(monthly.mean())
    std_cycle = float(monthly.std()) if len(monthly) > 1 else 5.0
    sigma = max(std_cycle, MIN_CYCLE_SIGMA)
    last_dip = pd.Timestamp(dip_dates[-1])
    days_since = (target_dt - last_dip).days
    k = max(1, round(days_since / avg_cycle))
    dist = days_since - k * avg_cycle
    dip_score = float(np.exp(-0.5 * (dist / sigma) ** 2))
    cv = std_cycle / avg_cycle
    cycle_quality = min(1.0, len(monthly) / 10.0) / (1.0 + cv)
    return dip_score, dip_gap, cycle_quality, form, n


def _feature_row(cf, gap, juice_asym, player_post, mid):
    dip_score, dip_gap, cyc_q, form, _n = cf
    return [gap, juice_asym, player_post, mid,
            form - mid, dip_score * cyc_q, dip_score * dip_gap]


def _build_training_matrix(hist):
    """Feature matrix + labels from the graded market history. player_post is
    computed cumulatively (each row only sees results from earlier rows), and
    every cycle feature uses box games strictly before its row's date, so the
    fit stays walk-forward-honest even within the training set."""
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

    rows, labels, kept = [], [], []
    for idx, r in h.iterrows():
        cf = _cycle_features(r["name_key"], r["date_dt"])
        if cf is None:
            continue
        rows.append(_feature_row(cf, r["gap"], r["juice_asym"],
                                 r["player_post"], r["mid"]))
        labels.append(r["under_win"])
        kept.append(idx)

    X = np.asarray(rows, dtype=float)
    y = np.asarray(labels, dtype=float)
    return X, y, global_under, h.loc[kept]


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
    if len(y) < MIN_TRAIN_ROWS:
        return None
    fit = _fit_logistic(X, y)
    player_rec = h.groupby("name_key")["under_win"].agg(["sum", "size"])
    _MODEL_CACHE["fit"] = {
        "fit": fit,
        "global_under": global_under,
        "player_rec": {k: (float(r["sum"]), int(r["size"]))
                       for k, r in player_rec.iterrows()},
        "n_train": len(y),
    }
    return _MODEL_CACHE["fit"]


def _prob_under(model, x_row):
    f = model["fit"]
    xs = (np.asarray(x_row, dtype=float) - f["mu"]) / f["sd"]
    z = float(np.clip(xs @ f["w"] + f["b"], -30, 30))
    return 1.0 / (1.0 + np.exp(-z))


# ----------------------------------------------------------------- pure logic

def decide(p_under, dip_score, dip_gap, cycle_quality, under_odds):
    """UNDER-only, dip-gated decision. Pure function — no I/O.
    Returns a bet dict or {'_skip_reason': ...}."""
    if cycle_quality <= 0.0:
        return {"_skip_reason": "no usable monthly dip cycle"}
    if dip_score < DIP_GATE:
        return {"_skip_reason": f"outside dip window (dip_score={dip_score:.2f} < {DIP_GATE})"}
    if dip_gap <= MIN_DIP_GAP:
        return {"_skip_reason": f"dips too shallow (dip_gap={dip_gap:.1f} <= {MIN_DIP_GAP})"}

    under_odds = _to_odds(under_odds)
    margin = p_under - _break_even(under_odds)
    if margin < EV_MARGIN:
        return {"_skip_reason": (
            f"no edge: p_under={p_under:.3f}, "
            f"under margin={margin:+.3f} (< +{EV_MARGIN:.2f})")}

    dec_odds = _decimal(under_odds)
    edge = p_under * dec_odds - 1.0
    if edge <= 0:
        return {"_skip_reason": "kelly edge non-positive"}
    f_kelly = QUARTER_KELLY * edge / (dec_odds - 1.0)
    amount = int(max(1, min(5, round(f_kelly * KELLY_TO_STAKE_MULTIPLIER))))

    note = "Low Output" if dip_score >= 0.5 else "Dip Window"
    return {"bet": "UNDER", "prob": p_under, "margin": margin,
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

    cf = _cycle_features(name_key, target_dt)
    if cf is None:
        print(TAG, f"{name} skipped [reason: career shorter than {MIN_CAREER_GAMES} games]")
        return None
    dip_score, dip_gap, cycle_quality, form, career_n = cf

    model = _get_fitted()
    if model is None:
        print(TAG, f"{name} skipped [reason: no DB connection or training history]")
        return None

    gap = under_line - over_line
    mid = (under_line + over_line) / 2.0
    juice_asym = _break_even(under_odds) - _break_even(over_odds)
    wins, n = model["player_rec"].get(name_key, (0.0, 0))
    player_post = (model["global_under"] * BETA_K + wins) / (BETA_K + n)

    x = _feature_row(cf, gap, juice_asym, player_post, mid)
    p_under = _prob_under(model, x)

    result = decide(p_under, dip_score, dip_gap, cycle_quality, under_odds)
    if "_skip_reason" in result:
        print(TAG, f"{name} skipped [reason: {result['_skip_reason']}]")
        return None

    # predicted_points: form minus the dip-weighted expected drop, clamped
    # under the line so the stored prediction always matches the UNDER bet.
    predicted = int(round(form - dip_score * dip_gap))
    predicted = max(0, min(predicted, int(np.floor(under_line))))

    print(
        TAG,
        f"{name}: bet=UNDER amount=${result['amount']} "
        f"p_under={p_under:.3f} margin={result['margin']:+.3f} "
        f"dip_score={dip_score:.2f} dip_gap={dip_gap:.1f} "
        f"form={form:.1f} line={mid} (trained on {model['n_train']} rows)"
    )

    return {
        "predicted_points": predicted,
        "bet": "UNDER",
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
