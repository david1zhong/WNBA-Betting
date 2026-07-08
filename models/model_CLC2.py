import pandas as pd
import numpy as np
import math
from datetime import datetime
import json
import warnings

try:
    from . import _schedule
except ImportError:  # run directly, not as models.*
    import _schedule


TAG = "[CLC2]"
warnings.filterwarnings('ignore')

np.random.seed(42)


# All seasons through the current year; files that don't exist yet are
# skipped at load time.
YEARS_FILES = {
    year: f"playerboxes/player_box_{year}.csv"
    for year in range(datetime.now().year, 2008, -1)
}

MIN_CAREER_GAMES = 20
LOW_OUTPUT_SD = 1.5
LOW_OUTPUT_CONF = 0.85
RIDGE_LAMBDA = 1.5
LOOKBACK_TRAIN_GAMES = 80

_CACHE = {}


def _normal_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _load_all_data():
    if "df" in _CACHE:
        return (_CACHE["df"], _CACHE["opp_pos_def"], _CACHE["league_pos_avg"],
                _CACHE["league_career_mean"])

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
        _CACHE.update({
            "df": empty, "opp_pos_def": {}, "league_pos_avg": {}, "league_career_mean": {}
        })
        return empty, {}, {}, {}

    df = pd.concat(frames, ignore_index=True)
    df["did_not_play"] = df["did_not_play"].astype(str).str.upper().eq("TRUE")
    df = df[~df["did_not_play"]].copy()

    df = df.dropna(subset=["athlete_display_name", "game_date", "points"])
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
    df = df.dropna(subset=["game_date"])

    df["points"] = pd.to_numeric(df["points"], errors="coerce")
    df["minutes"] = pd.to_numeric(df["minutes"], errors="coerce").fillna(0.0)
    df = df.dropna(subset=["points"])

    df["pos"] = df["athlete_position_abbreviation"].fillna("U").astype(str)
    df = df.sort_values(["athlete_display_name", "game_date"]).reset_index(drop=True)

    seasons_present = sorted(df["Year"].unique())
    recent_years = seasons_present[-3:] if len(seasons_present) >= 3 else seasons_present
    recent = df[df["Year"].isin(recent_years)]

    opp_pos_def = recent.groupby(["opponent_team_name", "pos"])["points"].mean().to_dict()
    league_pos_avg = recent.groupby("pos")["points"].mean().to_dict()

    # League-wide mean used for Bayesian shrinkage with sparse data
    league_career_mean = float(recent["points"].mean()) if not recent.empty else 10.0

    _CACHE.update({
        "df": df,
        "opp_pos_def": opp_pos_def,
        "league_pos_avg": league_pos_avg,
        "league_career_mean": league_career_mean,
    })
    return df, opp_pos_def, league_pos_avg, league_career_mean


def _ridge_solve(X, y, alpha):
    """Closed-form ridge: beta = (X'X + alpha*I)^-1 X'y. Don't penalize intercept (col 0)."""
    n, p = X.shape
    A = X.T @ X
    reg = alpha * np.eye(p)
    reg[0, 0] = 0.0
    try:
        return np.linalg.solve(A + reg, X.T @ y)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(X, y, rcond=None)[0]


def _build_training_features(player_df, opp_pos_def, league_pos_avg):
    """For each game with sufficient prior history, compute (features, target)."""
    pdf = player_df.sort_values("game_date").reset_index(drop=True).copy()
    pdf["rest"] = pdf["game_date"].diff().dt.days
    pdf["home_ind"] = (pdf["home_away"].astype(str).str.lower() == "home").astype(float)

    pts = pdf["points"].values.astype(float)
    mins = pdf["minutes"].values.astype(float)
    rests = pdf["rest"].values
    home = pdf["home_ind"].values
    months = pdf["game_date"].dt.month.values
    opp = pdf["opponent_team_name"].values
    positions = pdf["pos"].values

    rows = []
    targets = []
    for i in range(5, len(pdf)):
        recent5 = pts[max(0, i - 5):i]
        recent10 = pts[max(0, i - 10):i]
        if len(recent5) < 3:
            continue
        rec5_mean = float(np.mean(recent5))
        rec10_mean = float(np.mean(recent10)) if len(recent10) >= 3 else rec5_mean
        rec5_std = float(np.std(recent5, ddof=0))
        min5_mean = float(np.mean(mins[max(0, i - 5):i]))

        rest_today = rests[i]
        if pd.isna(rest_today):
            rest_today = 3.0
        rest_today = float(min(7.0, max(0.0, rest_today)))

        home_ind = float(home[i])
        m_today = int(months[i])
        if m_today <= 6:
            phase = -1.0
        elif m_today <= 8:
            phase = 0.0
        else:
            phase = 1.0

        league_avg = league_pos_avg.get(positions[i], None)
        opp_avg = opp_pos_def.get((opp[i], positions[i]), None)
        if league_avg is None or opp_avg is None or league_avg <= 0:
            opp_strength = 0.0
        else:
            opp_strength = (opp_avg - league_avg) / league_avg

        rows.append([
            1.0,           # intercept
            rec5_mean,
            rec10_mean,
            rec5_std,
            min5_mean,
            home_ind,
            rest_today,
            phase,
            float(opp_strength),
        ])
        targets.append(float(pts[i]))

    if not rows:
        return None, None
    return np.array(rows, dtype=float), np.array(targets, dtype=float)


def _features_for_today(player_df, target_date, today_home, today_opp, today_pos,
                        opp_pos_def, league_pos_avg):
    target = pd.to_datetime(target_date)
    history = player_df[player_df["game_date"] < target].sort_values("game_date")
    if history.empty:
        return None
    pts = history["points"].values.astype(float)
    mins = history["minutes"].values.astype(float)

    rec5_mean = float(np.mean(pts[-5:])) if len(pts) >= 1 else 0.0
    rec10_mean = float(np.mean(pts[-10:])) if len(pts) >= 1 else rec5_mean
    rec5_std = float(np.std(pts[-5:], ddof=0)) if len(pts) >= 2 else 0.0
    min5_mean = float(np.mean(mins[-5:])) if len(mins) >= 1 else 0.0

    last_date = history["game_date"].max()
    rest_days = (target - last_date).days if pd.notna(last_date) else 3
    rest_days = float(min(7.0, max(0.0, rest_days)))

    if today_home is None:
        home_split = (history["home_away"].astype(str).str.lower() == "home").mean()
        home_ind = float(home_split) if pd.notna(home_split) else 0.5
    else:
        home_ind = 1.0 if today_home == "home" else 0.0

    m = target.month
    if m <= 6:
        phase = -1.0
    elif m <= 8:
        phase = 0.0
    else:
        phase = 1.0

    league_avg = league_pos_avg.get(today_pos, None)
    opp_avg = opp_pos_def.get((today_opp, today_pos), None)
    if today_opp is None or league_avg is None or opp_avg is None or league_avg <= 0:
        opp_strength = 0.0
    else:
        opp_strength = (opp_avg - league_avg) / league_avg

    return np.array([
        1.0, rec5_mean, rec10_mean, rec5_std, min5_mean,
        home_ind, rest_days, phase, float(opp_strength)
    ], dtype=float)


def _quantile_from_residuals(residuals, q):
    if residuals.size == 0:
        return 0.0
    return float(np.quantile(residuals, q))


def _confidence_to_amount(confidence):
    if confidence >= 0.70:
        return 5
    if confidence >= 0.62:
        return 3
    if confidence >= 0.55:
        return 1
    return None


def _categorize(deviation_pct):
    if deviation_pct <= -0.20:
        return "Very Bad Game"
    if deviation_pct <= -0.08:
        return "Below Average"
    if deviation_pct < 0.05:
        return "Average Game"
    if deviation_pct < 0.15:
        return "Good Game"
    return "Very Good Game"


def _detect_monthly_dip(player_df, target_date, baseline, dip_pct=0.20):
    if baseline <= 0:
        return False
    target = pd.to_datetime(target_date)
    history = player_df[player_df["game_date"] < target].copy()
    if history.empty:
        return False
    history["ym"] = history["game_date"].dt.to_period("M")
    last_months = sorted(history["ym"].unique())[-3:]
    if len(last_months) < 2:
        return False
    threshold = baseline * (1.0 - dip_pct)
    months_with_dip = 0
    for ym in last_months:
        m = history[history["ym"] == ym]
        nearby = m[(m["game_date"].dt.day - target.day).abs() <= 3]
        if not nearby.empty and (nearby["points"] < threshold).any():
            months_with_dip += 1
    return months_with_dip >= 2


def _find_todays_opponent(df, player_name, target_date):
    target = pd.to_datetime(target_date)
    rows = df[(df["athlete_display_name"] == player_name) & (df["game_date"] == target)]
    if not rows.empty:
        r = rows.iloc[0]
        opp = r.get("opponent_team_name")
        ha = r.get("home_away")
        return (opp if pd.notna(opp) else None,
                str(ha).lower() if pd.notna(ha) else None)

    # Pre-game the box CSVs can't contain today's matchup — ask the ESPN
    # schedule for the player's team; degrades to (None, None) if the
    # source is unavailable, keeping factors neutral.
    hist = df[df["athlete_display_name"] == player_name]
    if hist.empty:
        return None, None
    team = hist.sort_values("game_date")["team_name"].iloc[-1]
    return _schedule.opponent_for_team(team, target_date)


def predict(player):
    name = player["name"]
    date_str = player["date"]
    over_line = float(player["over_line"])
    under_line = float(player["under_line"])

    df, opp_pos_def, league_pos_avg, league_career_mean = _load_all_data()
    if df.empty:
        print(TAG, f"{name} not in dip results")
        return None

    pdata = df[df["athlete_display_name"] == name].copy()
    if len(pdata) < MIN_CAREER_GAMES:
        print(TAG, f"{name} not in dip results")
        return None

    target = pd.to_datetime(date_str)
    pdata = pdata[pdata["game_date"] < target].copy()
    if len(pdata) < MIN_CAREER_GAMES:
        print(TAG, f"{name} not in dip results")
        return None

    train_data = pdata.tail(LOOKBACK_TRAIN_GAMES) if len(pdata) > LOOKBACK_TRAIN_GAMES else pdata
    X_train, y_train = _build_training_features(train_data, opp_pos_def, league_pos_avg)
    if X_train is None or len(X_train) < 10:
        print(TAG, f"{name} not in dip results")
        return None

    pos_mode = pdata["pos"].mode()
    today_pos = pos_mode.iloc[0] if not pos_mode.empty else "U"
    today_opp, today_ha = _find_todays_opponent(df, name, date_str)

    n_train = len(X_train)
    alpha = RIDGE_LAMBDA * (1.0 + max(0.0, (40 - n_train) / 40.0) * 4.0)
    beta = _ridge_solve(X_train, y_train, alpha)

    train_pred = X_train @ beta
    residuals = y_train - train_pred

    # Bayesian shrinkage of predicted mean toward career & league mean when data is thin
    career_mean = float(pdata["points"].mean())
    if n_train < 30:
        shrink_w = max(0.0, (30 - n_train) / 30.0) * 0.4
        shrunk_intercept_target = 0.7 * career_mean + 0.3 * league_career_mean
        # Pull predictions toward shrunk_target for cold-start
    else:
        shrink_w = 0.0
        shrunk_intercept_target = career_mean

    x_today = _features_for_today(pdata, date_str, today_ha, today_opp, today_pos,
                                  opp_pos_def, league_pos_avg)
    if x_today is None:
        print(TAG, f"{name} not in dip results")
        return None

    raw_pred = float(x_today @ beta)
    predicted_raw = (1.0 - shrink_w) * raw_pred + shrink_w * shrunk_intercept_target

    # Distribution from training residuals
    sigma = float(np.std(residuals, ddof=0))
    if sigma <= 1e-6:
        sigma = max(2.0, career_mean * 0.18)

    # Quantile spread (distributional output, not just mean)
    q25_offset = _quantile_from_residuals(residuals, 0.25)
    q50_offset = _quantile_from_residuals(residuals, 0.50)
    q75_offset = _quantile_from_residuals(residuals, 0.75)
    median_pred = predicted_raw + q50_offset
    q25_pred = predicted_raw + q25_offset
    q75_pred = predicted_raw + q75_offset

    # Predicted distribution width
    iqr = max(1e-6, q75_pred - q25_pred)
    # Effective sigma blends Gaussian residual std with IQR-based scale (~IQR/1.349)
    sigma_eff = max(sigma, iqr / 1.349)

    predicted_points = int(round(median_pred))
    bet = "OVER" if predicted_points > over_line else "UNDER"

    # Confidence: P(model's pick is correct) under predicted distribution
    z = (over_line - median_pred) / sigma_eff
    p_under = _normal_cdf(z)
    p_over = 1.0 - p_under
    confidence = max(p_over, p_under)
    confidence = float(min(0.95, max(0.0, confidence)))

    deviation_pct = (median_pred - career_mean) / max(1.0, career_mean)
    note = _categorize(deviation_pct)

    # Low Output: 1.5 SD below recent rolling mean + recurring monthly dip + top 10% confidence
    rec10 = pdata.tail(10)["points"]
    rec_mean = float(rec10.mean()) if len(rec10) >= 3 else career_mean
    rec_std = float(rec10.std(ddof=0)) if len(rec10) >= 3 else sigma
    if rec_std and rec_std > 0:
        sd_below = (rec_mean - median_pred) / rec_std
        dip_recurs = _detect_monthly_dip(pdata, date_str, career_mean)
        if sd_below >= LOW_OUTPUT_SD and dip_recurs and confidence >= LOW_OUTPUT_CONF:
            # Low Output calls from this model underperform badly; abstain rather
            # than insert a bet we don't trust.
            print(TAG, f"{name} skipped (Low Output suppressed)")
            return None

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
