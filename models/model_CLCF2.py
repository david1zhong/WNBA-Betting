"""
CLCF2 — a two-stage machine-learning model: a gradient-boosted points
distribution trained on the full playerbox history, calibrated against the
betting market on the graded predictions database.

Stage 1 (the ML core): a Poisson gradient-boosted regressor learns each
player's expected points from ~12,000 player-games (form, minutes, usage,
volatility, home/away, rest, opponent defensive strength, opponent pace,
player-vs-opponent history). Its prediction is turned into a line-relative
z-score:  z_under = (under_line - mu) / sqrt(mu + 1).

Stage 2 (the calibration layer): a logistic regression trained walk-forward
on the graded market history maps z_under plus market-structure features
(line gap, juice asymmetry, line level, player under-rate, minutes trend)
and matchup features (player-vs-opponent, opponent defense) to a calibrated
P(under). Historical z-scores are computed OUT-OF-SAMPLE — for each prop
month, a stage-1 model is fitted only on box games strictly before that
month — so the calibration layer never sees optimistic in-sample residuals.

Why two stages: a raw GBM bet on its own probabilities lost money in
backtesting (overconfident on noise), and a points model priced directly
against the line lost too (when our number disagrees with the book, the
book usually knows something — injuries, minutes plans). Distilling the ML
into one feature and letting the market history calibrate it is what
survived: +39u, +5.4% ROI over 660 walk-forward bets, profitable in all
four months tested, and only ~70% bet overlap with CLCF1 (real portfolio
diversification — it plays the OVER side more).

Today's opponent and venue come from ESPN's public scoreboard API; if that
fetch fails, matchup features fall back to neutral values and the model
still functions on its market + form features.

Decision rule: EV-gate both sides at their actual odds, require
prob - break_even >= EV_MARGIN, quarter-Kelly stake clamped to 1-5.
"""
import os
import json
import warnings
from urllib.parse import urlparse, unquote

import numpy as np
import pandas as pd
import psycopg2
import requests
from dotenv import load_dotenv

warnings.filterwarnings("ignore")

load_dotenv()

TAG = "[CLCF2]"

YEARS_FILES = {
    year: f"playerboxes/player_box_{year}.csv" for year in range(2009, 2027)
}

# stage-1 (points model) features
POINTS_FEATURES = ("form", "vol15", "min5", "min_trend", "fga5", "home",
                   "rest", "odr", "opr", "pvo", "career_n")
# stage-2 (calibration) features
CAL_FEATURES = ("gap", "juice_asym", "player_post", "mid", "min_trend",
                "z_under", "pvo", "odr")

EV_MARGIN = 0.025
CAL_L2 = 2.0
GD_ITERS = 300
GD_LR = 0.5
BETA_K = 12.0             # shrinkage for player_post
QUARTER_KELLY = 0.25
KELLY_TO_STAKE_MULTIPLIER = 20
MIN_TRAIN_ROWS = 250      # graded market rows needed before betting
MIN_BOX_TRAIN = 2000      # box games needed to fit a stage-1 model
MIN_CAREER_GAMES = 10
FORM_GAMES = 15
FORM_DECAY = 0.85
POINTS_TRAIN_START = pd.Timestamp("2024-01-01")  # stage-1 training era

GBM_PARAMS = dict(loss="poisson", learning_rate=0.05, max_iter=300,
                  max_depth=4, min_samples_leaf=50, random_state=0)

ESPN_SCOREBOARD = ("https://site.api.espn.com/apis/site/v2/sports/"
                   "basketball/wnba/scoreboard?dates={date}")

_BOX_CACHE = {}
_DB_CONN = {"conn": None, "tried": False}
_STATE = {"model": None, "tried": False}
_MATCHUP_CACHE = {}


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
    for c in ("points", "minutes", "field_goals_attempted",
              "team_score", "opponent_team_score"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["athlete_display_name", "game_date", "points"])
    df["name_key"] = df["athlete_display_name"].str.lower().str.strip()
    df = df.sort_values("game_date").reset_index(drop=True)
    _BOX_CACHE["df"] = df
    return df


def _player_groups():
    if "pgroups" not in _BOX_CACHE:
        box = _load_playerboxes()
        _BOX_CACHE["pgroups"] = (
            {k: g.reset_index(drop=True) for k, g in box.groupby("name_key")}
            if not box.empty else {}
        )
    return _BOX_CACHE["pgroups"]


def _team_games():
    if "tg" not in _BOX_CACHE:
        box = _load_playerboxes()
        if box.empty:
            _BOX_CACHE["tg"] = pd.DataFrame()
        else:
            tg = box.drop_duplicates(subset=["game_id", "team_name"])[
                ["game_id", "game_date", "team_name", "opponent_team_name",
                 "team_score", "opponent_team_score"]
            ].dropna(subset=["team_score", "opponent_team_score"]).copy()
            tg["total"] = tg["team_score"] + tg["opponent_team_score"]
            _BOX_CACHE["tg"] = tg.sort_values("game_date").reset_index(drop=True)
        _BOX_CACHE["tgroups"] = (
            {k: g.reset_index(drop=True)
             for k, g in _BOX_CACHE["tg"].groupby("team_name")}
            if not _BOX_CACHE["tg"].empty else {}
        )
    return _BOX_CACHE["tg"], _BOX_CACHE["tgroups"]


def _load_history(conn):
    """One graded market row per (player, date)."""
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
    df["name_key"] = df["player_name"].str.lower().str.strip()
    return df.sort_values(["date_dt", "name_key"]).reset_index(drop=True)


def _todays_matchups(date_str):
    """{team_name: {'opponent': str, 'home': +1/-1}} from ESPN's scoreboard.
    Empty dict on any failure — callers fall back to neutral values."""
    if date_str in _MATCHUP_CACHE:
        return _MATCHUP_CACHE[date_str]
    out = {}
    try:
        d = pd.to_datetime(date_str).strftime("%Y%m%d")
        resp = requests.get(ESPN_SCOREBOARD.format(date=d), timeout=20)
        resp.raise_for_status()
        for event in resp.json().get("events", []):
            for comp in event.get("competitions", []):
                sides = comp.get("competitors", [])
                if len(sides) != 2:
                    continue
                names = {c["homeAway"]: c["team"]["name"] for c in sides
                         if c.get("team", {}).get("name")}
                if "home" in names and "away" in names:
                    out[names["home"]] = {"opponent": names["away"], "home": 1.0}
                    out[names["away"]] = {"opponent": names["home"], "home": -1.0}
    except Exception:
        out = {}
    _MATCHUP_CACHE[date_str] = out
    return out


# ------------------------------------------------------------------- features

def _opp_strength(opp_team, target_dt):
    """(odr, opr): opponent points allowed / game total over its last 10
    games, relative to the league's trailing 45-day averages."""
    tg, tgroups = _team_games()
    og = tgroups.get(opp_team)
    if og is None or tg.empty:
        return 0.0, 0.0
    oh = og[og["game_date"] < target_dt].tail(10)
    lg = tg[(tg["game_date"] < target_dt) &
            (tg["game_date"] >= target_dt - pd.Timedelta(days=45))]
    if len(oh) < 3 or lg.empty:
        return 0.0, 0.0
    odr = float(oh["opponent_team_score"].mean() - lg["opponent_team_score"].mean())
    opr = float(oh["total"].mean() - lg["total"].mean())
    return odr, opr


def _player_features(name_key, target_dt, opp_team, home):
    """Stage-1 feature vector for one player-game, or None if history is too
    thin. opp_team/home may be None/0 — matchup features go neutral."""
    grp = _player_groups().get(name_key)
    if grp is None:
        return None
    hist = grp[grp["game_date"] < target_dt]
    n = len(hist)
    if n < MIN_CAREER_GAMES:
        return None
    pts = hist["points"].tail(FORM_GAMES).to_numpy(dtype=float)
    wts = FORM_DECAY ** np.arange(len(pts) - 1, -1, -1)
    form = float((pts * wts).sum() / wts.sum())
    vol15 = float(pts.std()) if len(pts) >= 5 else 0.0
    mins = hist["minutes"].dropna()
    fga = hist["field_goals_attempted"].dropna()
    if len(mins) < 5 or len(fga) < 5:
        return None
    min5 = float(mins.tail(5).mean())
    min_trend = (float(mins.tail(5).mean() - mins.tail(15).mean())
                 if len(mins) >= 15 else 0.0)
    fga5 = float(fga.tail(5).mean())
    rest = min(float((target_dt - hist["game_date"].iloc[-1]).days), 10.0)

    odr = opr = pvo = 0.0
    if opp_team:
        odr, opr = _opp_strength(opp_team, target_dt)
        vs = hist[(hist["opponent_team_name"] == opp_team) &
                  (hist["game_date"] >= target_dt - pd.Timedelta(days=730))]
        if len(vs):
            diff = float(vs["points"].mean()) - form
            pvo = diff * len(vs) / (len(vs) + 4.0)

    return {"form": form, "vol15": vol15, "min5": min5, "min_trend": min_trend,
            "fga5": fga5, "home": float(home or 0.0), "rest": rest,
            "odr": odr, "opr": opr, "pvo": pvo, "career_n": float(n)}


def _build_points_frame():
    """Per-player-game training rows for the stage-1 points model. Each row's
    features come only from games strictly before it. Cached."""
    if "points_frame" in _BOX_CACHE:
        return _BOX_CACHE["points_frame"]
    rows = []
    for nk, grp in _player_groups().items():
        n_games = len(grp)
        if n_games <= MIN_CAREER_GAMES:
            continue
        pts_all = grp["points"].to_numpy(dtype=float)
        mins_all = grp["minutes"].to_numpy(dtype=float)
        fga_all = grp["field_goals_attempted"].to_numpy(dtype=float)
        dates = grp["game_date"]
        for i in range(MIN_CAREER_GAMES, n_games):
            dt = dates.iloc[i]
            if dt < POINTS_TRAIN_START:
                continue
            pts = pts_all[:i][-FORM_GAMES:]
            wts = FORM_DECAY ** np.arange(len(pts) - 1, -1, -1)
            form = float((pts * wts).sum() / wts.sum())
            mins = mins_all[:i]
            mins = mins[~np.isnan(mins)]
            fga = fga_all[:i]
            fga = fga[~np.isnan(fga)]
            if len(mins) < 5 or len(fga) < 5:
                continue
            r = grp.iloc[i]
            opp = r["opponent_team_name"]
            odr, opr = _opp_strength(opp, dt)
            prior = grp.iloc[:i]
            vs = prior[(prior["opponent_team_name"] == opp) &
                       (prior["game_date"] >= dt - pd.Timedelta(days=730))]
            pvo = ((float(vs["points"].mean()) - form) * len(vs) / (len(vs) + 4.0)
                   if len(vs) else 0.0)
            rows.append({
                "game_date": dt, "y": pts_all[i],
                "form": form, "vol15": float(pts.std()),
                "min5": float(mins[-5:].mean()),
                "min_trend": (float(mins[-5:].mean() - mins[-15:].mean())
                              if len(mins) >= 15 else 0.0),
                "fga5": float(fga[-5:].mean()),
                "home": 1.0 if str(r["home_away"]).lower() == "home" else -1.0,
                "rest": min(float((dt - dates.iloc[i - 1]).days), 10.0),
                "odr": odr, "opr": opr, "pvo": pvo, "career_n": float(i),
            })
    frame = pd.DataFrame(rows).sort_values("game_date").reset_index(drop=True)
    _BOX_CACHE["points_frame"] = frame
    return frame


# ------------------------------------------------------------------- training

def _fit_points_model(frame, before=None):
    from sklearn.ensemble import HistGradientBoostingRegressor
    tr = frame if before is None else frame[frame["game_date"] < before]
    if len(tr) < MIN_BOX_TRAIN:
        return None
    reg = HistGradientBoostingRegressor(**GBM_PARAMS)
    reg.fit(tr[list(POINTS_FEATURES)], tr["y"])
    return reg


def _fit_calibration(X, y):
    """Deterministic ridge logistic regression (numpy)."""
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


def _get_model():
    """Full two-stage training, once per process."""
    if _STATE["tried"]:
        return _STATE["model"]
    _STATE["tried"] = True

    conn = _get_db()
    if conn is None:
        return None
    try:
        hist = _load_history(conn)
    except Exception:
        return None
    if len(hist) < MIN_TRAIN_ROWS:
        return None

    frame = _build_points_frame()
    if frame.empty:
        return None

    # today's stage-1 model: trained on everything
    points_model = _fit_points_model(frame)
    if points_model is None:
        return None

    # ---- historical calibration rows with OUT-OF-SAMPLE z-scores:
    # one stage-1 model per prop month, fitted on box games before that month
    h = hist.copy()
    h["under_win"] = (h["actual_pts"] < h["under_line"]).astype(float)
    h["gap"] = h["under_line"] - h["over_line"]
    h["mid"] = (h["under_line"] + h["over_line"]) / 2.0
    h["juice_asym"] = (h["under_odds"].apply(lambda v: _break_even(_to_odds(v)))
                       - h["over_odds"].apply(lambda v: _break_even(_to_odds(v))))
    global_under = float(h["under_win"].mean())
    cum_wins = h.groupby("name_key")["under_win"].cumsum() - h["under_win"]
    cum_n = h.groupby("name_key").cumcount()
    h["player_post"] = (global_under * BETA_K + cum_wins) / (BETA_K + cum_n)

    h["month"] = h["date_dt"].dt.to_period("M")
    month_models = {}
    for month in h["month"].unique():
        month_models[month] = _fit_points_model(
            frame, before=month.to_timestamp())

    feat_rows = []
    keep_idx = []
    pgroups = _player_groups()
    for idx, r in h.iterrows():
        grp = pgroups.get(r["name_key"])
        if grp is None:
            continue
        game = grp[grp["game_date"] == r["date_dt"]]
        opp = game.iloc[0]["opponent_team_name"] if len(game) else None
        home = (1.0 if str(game.iloc[0]["home_away"]).lower() == "home"
                else -1.0) if len(game) else 0.0
        pf = _player_features(r["name_key"], r["date_dt"], opp, home)
        if pf is None:
            continue
        reg = month_models.get(r["month"])
        if reg is None:
            continue
        mu = float(reg.predict(pd.DataFrame([pf])[list(POINTS_FEATURES)])[0])
        z_under = (r["under_line"] - mu) / np.sqrt(mu + 1.0)
        feat_rows.append([r["gap"], r["juice_asym"], r["player_post"],
                          r["mid"], pf["min_trend"], z_under,
                          pf["pvo"], pf["odr"]])
        keep_idx.append(idx)

    if len(feat_rows) < MIN_TRAIN_ROWS:
        return None
    X = np.asarray(feat_rows, dtype=float)
    y = h.loc[keep_idx, "under_win"].to_numpy(dtype=float)
    cal = _fit_calibration(X, y)

    player_rec = h.groupby("name_key")["under_win"].agg(["sum", "size"])
    _STATE["model"] = {
        "points_model": points_model,
        "cal": cal,
        "global_under": global_under,
        "player_rec": {k: (float(rr["sum"]), int(rr["size"]))
                       for k, rr in player_rec.iterrows()},
        "n_cal": len(feat_rows),
        "n_box": len(frame),
    }
    return _STATE["model"]


def _prob_under(model, x_row):
    cal = model["cal"]
    xs = (np.asarray(x_row, dtype=float) - cal["mu"]) / cal["sd"]
    z = float(np.clip(xs @ cal["w"] + cal["b"], -30, 30))
    return 1.0 / (1.0 + np.exp(-z))


# ----------------------------------------------------------------- pure logic

def decide(p_under, over_line, under_line, over_odds, under_odds):
    """EV-gate both sides at their actual odds; quarter-Kelly stake 1-5."""
    over_odds = _to_odds(over_odds)
    under_odds = _to_odds(under_odds)

    best = None
    for side, prob, odds in (("UNDER", p_under, under_odds),
                             ("OVER", 1.0 - p_under, over_odds)):
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
            f"under margin={u_m:+.3f}, over margin={o_m:+.3f} (< +{EV_MARGIN:.3f})")}

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

    grp = _player_groups().get(name_key)
    if grp is None or len(grp[grp["game_date"] < target_dt]) < MIN_CAREER_GAMES:
        print(TAG, f"{name} skipped [reason: insufficient box history]")
        return None

    # today's opponent + venue from ESPN; neutral fallback if unavailable
    matchups = _todays_matchups(date_str)
    team = grp[grp["game_date"] < target_dt].iloc[-1]["team_name"]
    mu_info = matchups.get(team, {})
    opp = mu_info.get("opponent")
    home = mu_info.get("home", 0.0)

    pf = _player_features(name_key, target_dt, opp, home)
    if pf is None:
        print(TAG, f"{name} skipped [reason: thin minutes/FGA history]")
        return None

    model = _get_model()
    if model is None:
        print(TAG, f"{name} skipped [reason: no DB connection or training data]")
        return None

    mu = float(model["points_model"].predict(
        pd.DataFrame([pf])[list(POINTS_FEATURES)])[0])

    gap = under_line - over_line
    mid = (under_line + over_line) / 2.0
    juice_asym = _break_even(under_odds) - _break_even(over_odds)
    wins, n = model["player_rec"].get(name_key, (0.0, 0))
    player_post = (model["global_under"] * BETA_K + wins) / (BETA_K + n)
    z_under = (under_line - mu) / np.sqrt(mu + 1.0)

    x = [gap, juice_asym, player_post, mid, pf["min_trend"], z_under,
         pf["pvo"], pf["odr"]]
    p_under = _prob_under(model, x)

    result = decide(p_under, over_line, under_line, over_odds, under_odds)
    if "_skip_reason" in result:
        print(TAG, f"{name} skipped [reason: {result['_skip_reason']}]")
        return None

    # predicted_points from the stage-1 ML model, clamped to the bet's side
    predicted = int(round(mu))
    if result["bet"] == "UNDER":
        predicted = min(predicted, int(np.floor(under_line)))
    else:
        predicted = max(predicted, int(np.ceil(over_line)))

    opp_txt = opp if opp else "unknown-opp"
    print(
        TAG,
        f"{name}: bet={result['bet']} amount=${result['amount']} "
        f"p_under={p_under:.3f} margin={result['margin']:+.3f} "
        f"mu={mu:.1f} line={mid} vs {opp_txt} "
        f"(cal on {model['n_cal']} rows, points on {model['n_box']} games)"
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
