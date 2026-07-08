"""
CLCF4 — an honest points-prediction model, eligible for the model-vs-
sportsbook error comparison.

Unlike CLCF1/2/3 (which decide a side and store a placeholder on that side
of the line), CLCF4 predicts the player's actual point total first and
derives everything else from that number. predicted_pts is the raw model
output, never clamped to the bet side.

Prediction: a deterministic ridge regression (numpy closed-form) trained on
the full playerbox history (2009-present), one row per player-game with at
least MIN_CAREER_GAMES prior games. Every feature is computed from games
strictly before that row's date, so the table is walk-forward-honest by
construction:

  - form:       recency-weighted points over the last 15 games (decay 0.85)
  - momentum:   short form (last 5) minus long form
  - season_avg: points per game this season to date (career avg if none)
  - career_avg: points per game, career to date
  - min_form:   recency-weighted minutes over the last 10 games
  - fg_form:    recency-weighted FG% over the last 15 games
  - rest:       days since the previous game, capped at 10
  - home:       1/0 from the box rows; 0.5 in production when the ESPN
                scoreboard lookup (models/_schedule.py) can't resolve it
  - opp_def:    opponent's points allowed per game (last 20 team-games)
                minus the league average to date; 0 when unknown

Uncertainty: the training residuals' absolute error is regressed on the
prediction (bigger scorers are noisier), giving sigma(pred). A normal CDF
turns prediction-vs-line into raw P(under)/P(over).

Staking is calibrated, not raw. Walk-forward, the raw regression loses to
the book on accuracy (MAE 4.996 vs 4.893 on 1,902 graded player-dates), so
betting its raw disagreement with the line is -4% ROI at every gate — the
vig. Instead, a ridge logistic (z = standardized line-vs-prediction gap,
plus juice asymmetry) is trained on the graded market history in the
predictions DB, with each history row's z computed from coefficients refit
on box games strictly before that row's date. The calibrated P(under)
scores Brier 0.2485 vs the book's no-vig 0.2492, and staking only when it
clears the break-even by EV_MARGIN backtests +8.1% ROI (52 bets, 32-20,
61.5%, positive 3 of 4 months — small sample by design).

Output: every eligible player gets a row — the honest prediction plus the
side the model favors — so the error comparison accumulates a full slate
every day. amount is set (quarter-Kelly, 1-5) only on calibrated edges;
otherwise the row is a paper pick, same convention as CL1's 2025 season.
With no DB (or thin graded history) the model still emits paper picks and
never stakes.
"""
import os
import json
import math
import warnings
from datetime import datetime
from urllib.parse import urlparse, unquote

import numpy as np
import pandas as pd
import psycopg2
from dotenv import load_dotenv

try:
    from . import _schedule
except ImportError:
    import _schedule

warnings.filterwarnings("ignore")

load_dotenv()

TAG = "[CLCF4]"

YEARS_FILES = {
    year: f"playerboxes/player_box_{year}.csv"
    for year in range(2009, datetime.now().year + 1)
}

FEATURES = ("form", "momentum", "season_avg", "career_avg",
            "min_form", "fg_form", "rest", "home", "opp_def")
CAL_FEATURES = ("z", "juice_asym")

EV_MARGIN = 0.06          # calibrated edge over the break-even to stake
L2 = 1.0                  # ridge strength (points regression)
CAL_L2 = 2.0              # ridge strength (calibration logistic)
GD_ITERS = 300
GD_LR = 0.5
QUARTER_KELLY = 0.25
KELLY_TO_STAKE_MULTIPLIER = 20   # f * 20 -> integer stake, clamped 1-5
MIN_TRAIN_ROWS = 10000    # box player-games needed to fit the regression
MIN_CAL_ROWS = 250        # graded market rows needed to allow staking
MIN_CAREER_GAMES = 15     # prior games needed before a player is predictable
FORM_GAMES = 15
SHORT_FORM_GAMES = 5
MIN_FORM_GAMES = 10
FORM_DECAY = 0.85
REST_CAP = 10.0
DEF_GAMES = 20            # opponent defense lookback (team-games)
SIGMA_FLOOR = 3.0         # minimum points std for the probability model
HOME_NEUTRAL = 0.5

_BOX_CACHE = {}
_DB_CONN = {"conn": None, "tried": False}
_MODEL_CACHE = {"tried": False, "ridge": None, "cal": None}


# ------------------------------------------------------------------ odds math

def _to_odds(v, default=-110.0):
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


def _norm_cdf(z):
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


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
    for c in ("points", "minutes", "field_goals_made", "field_goals_attempted",
              "opponent_team_score", "season"):
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
    """One graded market row per (player, date). Any model's row works —
    lines/odds/actuals are identical across models for the same player-date
    because they come from the same props scrape."""
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


def _defense_tables():
    """Per-team points-allowed history plus a league-wide cumulative mean,
    each as sorted arrays for walk-forward lookups by date."""
    if "defense" in _BOX_CACHE:
        return _BOX_CACHE["defense"]
    box = _load_playerboxes()
    if box.empty:
        _BOX_CACHE["defense"] = ({}, np.array([]), np.array([]))
        return _BOX_CACHE["defense"]

    games = (
        box.dropna(subset=["team_name", "opponent_team_score"])
        .drop_duplicates(subset=["game_id", "team_name"])
        [["team_name", "game_date", "opponent_team_score"]]
        .sort_values("game_date")
    )
    team_tbl = {}
    for team, g in games.groupby(games["team_name"].str.lower()):
        team_tbl[team] = (g["game_date"].to_numpy(),
                          g["opponent_team_score"].to_numpy(dtype=float))

    lg_dates = games["game_date"].to_numpy()
    lg_cum = np.cumsum(games["opponent_team_score"].to_numpy(dtype=float))
    _BOX_CACHE["defense"] = (team_tbl, lg_dates, lg_cum)
    return _BOX_CACHE["defense"]


def _opp_def(opponent_team, target_dt):
    """Opponent's points allowed per game (last DEF_GAMES before target_dt)
    minus the league average to date. 0.0 when unknown."""
    if not opponent_team:
        return 0.0
    team_tbl, lg_dates, lg_cum = _defense_tables()
    rec = team_tbl.get(str(opponent_team).lower())
    if rec is None or len(lg_dates) == 0:
        return 0.0
    dates, allowed = rec
    i = np.searchsorted(dates, np.datetime64(target_dt))
    if i < 5:
        return 0.0
    recent = allowed[max(0, i - DEF_GAMES):i]
    j = np.searchsorted(lg_dates, np.datetime64(target_dt))
    if j < 50:
        return 0.0
    league_avg = lg_cum[j - 1] / j
    return float(recent.mean() - league_avg)


# ------------------------------------------------------------------- features

def _decay_mean(a, k):
    a = a[-k:]
    w = FORM_DECAY ** np.arange(len(a) - 1, -1, -1)
    return float((a * w).sum() / w.sum())


def _decay_ratio(num, den, k):
    num, den = num[-k:], den[-k:]
    w = FORM_DECAY ** np.arange(len(num) - 1, -1, -1)
    d = float((den * w).sum())
    return float((num * w).sum() / d) if d > 0 else 0.0


def _history_features(hist, target_dt, target_season, home, opp_def):
    """Feature vector from a player's box games strictly before target_dt."""
    pts = hist["points"].to_numpy(dtype=float)
    mins = np.nan_to_num(hist["minutes"].to_numpy(dtype=float))
    fgm = np.nan_to_num(hist["field_goals_made"].to_numpy(dtype=float))
    fga = np.nan_to_num(hist["field_goals_attempted"].to_numpy(dtype=float))

    form = _decay_mean(pts, FORM_GAMES)
    momentum = _decay_mean(pts, SHORT_FORM_GAMES) - form
    career_avg = float(pts.mean())
    season_pts = pts[hist["season"].to_numpy() == target_season]
    season_avg = float(season_pts.mean()) if len(season_pts) else career_avg
    min_form = _decay_mean(mins, MIN_FORM_GAMES)
    fg_form = _decay_ratio(fgm, fga, FORM_GAMES)
    rest = min((target_dt - hist["game_date"].iloc[-1]).days, REST_CAP)

    return [form, momentum, season_avg, career_avg,
            min_form, fg_form, float(rest), float(home), float(opp_def)]


def _build_training_table():
    """One row per player-game with MIN_CAREER_GAMES prior games: the
    feature vector, the target points, and the game date — sorted by date."""
    rows, targets, dates = [], [], []
    for _key, g in _box_groups().items():
        n = len(g)
        if n <= MIN_CAREER_GAMES:
            continue
        game_dates = g["game_date"]
        seasons = g["season"].to_numpy()
        homes = (g["home_away"].astype(str).str.lower() == "home").to_numpy()
        opps = g["opponent_team_name"].to_numpy()
        for i in range(MIN_CAREER_GAMES, n):
            dt = game_dates.iloc[i]
            rows.append(_history_features(
                g.iloc[:i], dt, seasons[i], 1.0 if homes[i] else 0.0,
                _opp_def(opps[i], dt),
            ))
            targets.append(float(g["points"].iloc[i]))
            dates.append(dt)
    X = np.asarray(rows, dtype=float)
    y = np.asarray(targets, dtype=float)
    d = np.asarray(dates, dtype="datetime64[ns]")
    order = np.argsort(d, kind="stable")
    return X[order], y[order], d[order]


# ---------------------------------------------------------------------- fits

def _fit_ridge(X, y):
    """Deterministic closed-form ridge regression on standardized features,
    plus a linear model of the absolute residual for sigma(pred)."""
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd < 1e-9] = 1.0
    Xs = (X - mu) / sd
    n, k = Xs.shape
    A = Xs.T @ Xs / n + L2 * np.eye(k) / n
    b = Xs.T @ (y - y.mean()) / n
    w = np.linalg.solve(A, b)
    intercept = float(y.mean())

    pred = Xs @ w + intercept
    abs_resid = np.abs(y - pred)
    P = np.column_stack([np.ones(n), pred])
    coef = np.linalg.lstsq(P, abs_resid, rcond=None)[0]

    return {"w": w, "b": intercept, "mu": mu, "sd": sd,
            "sig_a": float(coef[0]), "sig_b": float(coef[1])}


def _predict_points(fit, x_row):
    xs = (np.asarray(x_row, dtype=float) - fit["mu"]) / fit["sd"]
    return float(xs @ fit["w"] + fit["b"])


def _sigma(fit, pred):
    return max(SIGMA_FLOOR, math.sqrt(math.pi / 2.0)
               * (fit["sig_a"] + fit["sig_b"] * pred))


def _fit_logistic(X, y):
    """Deterministic ridge-regularized logistic regression."""
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd < 1e-9] = 1.0
    Xs = (X - mu) / sd
    n = len(y)
    w = np.zeros(Xs.shape[1])
    b = 0.0
    for _ in range(GD_ITERS):
        z = np.clip(Xs @ w + b, -30, 30)
        p = 1.0 / (1.0 + np.exp(-z))
        w -= GD_LR * (Xs.T @ (p - y) / n + CAL_L2 * w / n)
        b -= GD_LR * float((p - y).mean())
    return {"w": w, "b": b, "mu": mu, "sd": sd}


def _cal_prob(cal, z, juice_asym):
    xs = (np.array([z, juice_asym]) - cal["mu"]) / cal["sd"]
    zz = float(np.clip(xs @ cal["w"] + cal["b"], -30, 30))
    return 1.0 / (1.0 + np.exp(-zz))


def _fit_calibration(hist, X_all, y_all, d_all):
    """Logistic mapping (z, juice_asym) -> P(under) on the graded market
    history. Each row's z uses ridge coefficients refit on box player-games
    strictly before that row's date, so the calibration stays walk-forward-
    honest even within its own training set."""
    groups = _box_groups()
    rows, labels = [], []
    h = hist.copy()
    h["name_key"] = h["player_name"].str.lower().str.strip()

    for date, day in h.groupby("date_dt"):
        cut = np.searchsorted(d_all, np.datetime64(date))
        if cut < MIN_TRAIN_ROWS:
            continue
        fit = _fit_ridge(X_all[:cut], y_all[:cut])
        for _, r in day.iterrows():
            grp = groups.get(r["name_key"])
            if grp is None:
                continue
            player_hist = grp[grp["game_date"] < date]
            if len(player_hist) < MIN_CAREER_GAMES:
                continue
            game = grp[grp["game_date"] == date]
            if len(game):
                home = 1.0 if str(game["home_away"].iloc[0]).lower() == "home" else 0.0
                opp = game["opponent_team_name"].iloc[0]
            else:
                home, opp = HOME_NEUTRAL, None
            x = _history_features(player_hist, date, date.year, home,
                                  _opp_def(opp, date))
            pred = _predict_points(fit, x)
            sigma = _sigma(fit, pred)
            under_line = float(r["under_line"])
            u_be = _break_even(_to_odds(r["under_odds"]))
            o_be = _break_even(_to_odds(r["over_odds"]))
            rows.append([(under_line - pred) / sigma, u_be - o_be])
            labels.append(float(float(r["actual_pts"]) < under_line))

    if len(labels) < MIN_CAL_ROWS:
        return None
    cal = _fit_logistic(np.asarray(rows, dtype=float),
                        np.asarray(labels, dtype=float))
    cal["n_cal"] = len(labels)
    return cal


def _get_fitted():
    """Fit once per process: the points regression always, the staking
    calibration only when the DB's graded history is available and thick
    enough. Returns the ridge fit or None."""
    if _MODEL_CACHE["tried"]:
        return _MODEL_CACHE["ridge"]
    _MODEL_CACHE["tried"] = True

    X, y, d = _build_training_table()
    if len(y) < MIN_TRAIN_ROWS:
        return None
    ridge = _fit_ridge(X, y)
    ridge["n_train"] = len(y)
    _MODEL_CACHE["ridge"] = ridge

    conn = _get_db()
    if conn is not None:
        try:
            hist = _load_history(conn)
        except Exception:
            hist = None
        if hist is not None and len(hist) >= MIN_CAL_ROWS:
            try:
                _MODEL_CACHE["cal"] = _fit_calibration(hist, X, y, d)
            except Exception:
                _MODEL_CACHE["cal"] = None
    return ridge


# ----------------------------------------------------------------- pure logic

def decide(p_under, p_over, over_odds, under_odds, can_stake):
    """Pick the side with the better probability margin; stake it only when
    calibrated (can_stake) and the margin clears EV_MARGIN. Pure function.
    Always returns a pick dict: amount is None for paper picks."""
    u_odds, o_odds = _to_odds(under_odds), _to_odds(over_odds)
    m_under = p_under - _break_even(u_odds)
    m_over = p_over - _break_even(o_odds)

    if m_under >= m_over:
        side, prob, margin, odds = "UNDER", p_under, m_under, u_odds
    else:
        side, prob, margin, odds = "OVER", p_over, m_over, o_odds

    amount = None
    note = "Lean"
    if can_stake and margin >= EV_MARGIN:
        dec_odds = _decimal(odds)
        edge = prob * dec_odds - 1.0
        if edge > 0:
            f_kelly = QUARTER_KELLY * edge / (dec_odds - 1.0)
            amount = int(max(1, min(5, round(f_kelly * KELLY_TO_STAKE_MULTIPLIER))))
            note = "Value"

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

    grp = _box_groups().get(name_key)
    if grp is None:
        print(TAG, f"{name} skipped [reason: no box history]")
        return None
    hist = grp[grp["game_date"] < target_dt]
    if len(hist) < MIN_CAREER_GAMES:
        print(TAG, f"{name} skipped [reason: career shorter than {MIN_CAREER_GAMES} games]")
        return None

    fit = _get_fitted()
    if fit is None:
        print(TAG, f"{name} skipped [reason: box history too thin to train]")
        return None

    team = hist["team_name"].iloc[-1]
    opponent, home_away = _schedule.opponent_for_team(team, date_str)
    home = 1.0 if home_away == "home" else 0.0 if home_away == "away" else HOME_NEUTRAL
    opp_def = _opp_def(opponent, target_dt)

    x = _history_features(hist, target_dt, target_dt.year, home, opp_def)
    pred = _predict_points(fit, x)
    sigma = _sigma(fit, pred)

    cal = _MODEL_CACHE["cal"]
    if cal is not None:
        juice_asym = _break_even(under_odds) - _break_even(over_odds)
        p_under = _cal_prob(cal, (under_line - pred) / sigma, juice_asym)
        p_over = 1.0 - p_under
    else:
        p_under = _norm_cdf((under_line - pred) / sigma)
        p_over = 1.0 - _norm_cdf((over_line - pred) / sigma)

    result = decide(p_under, p_over, over_odds, under_odds, cal is not None)
    predicted = max(0, int(round(pred)))
    staked = f"${result['amount']}" if result["amount"] else "paper"
    cal_note = f"cal n={cal['n_cal']}" if cal is not None else "uncalibrated"

    print(
        TAG,
        f"{name}: pred={pred:.1f} sigma={sigma:.1f} bet={result['bet']} "
        f"({staked}) prob={result['prob']:.3f} margin={result['margin']:+.3f} "
        f"line O{over_line}/U{under_line} opp={opponent or '?'} "
        f"({cal_note}, trained on {fit['n_train']} rows)"
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
        amt = f"${result['amount']}" if result["amount"] else "paper"
        print(f"{player['name']}: {result['predicted_points']} pts, "
              f"bet {result['bet']} ({amt}) "
              f"(O{result['over_line']}/U{result['under_line']}), note {result['note']}")
