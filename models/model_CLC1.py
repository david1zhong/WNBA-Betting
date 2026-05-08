import pandas as pd
import numpy as np
import math
from datetime import datetime, timedelta
import json
import warnings

warnings.filterwarnings('ignore')

np.random.seed(42)


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

MIN_CAREER_GAMES = 20
LOW_OUTPUT_SD = 1.5
LOW_OUTPUT_CONF = 0.85

# Module-level cache so the test loop does not reread CSVs per player
_CACHE = {}


def _normal_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _load_all_data():
    if "df" in _CACHE:
        return (_CACHE["df"], _CACHE["opp_pos_def"], _CACHE["opp_pace"],
                _CACHE["league_pos_avg"])

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
        _CACHE.update({"df": empty, "opp_pos_def": {}, "opp_pace": {}, "league_pos_avg": {}})
        return empty, {}, {}, {}

    df = pd.concat(frames, ignore_index=True)

    df["did_not_play"] = df["did_not_play"].astype(str).str.upper().eq("TRUE")
    df = df[~df["did_not_play"]].copy()

    df = df.dropna(subset=["athlete_display_name", "game_date", "points"])
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
    df = df.dropna(subset=["game_date"])

    df["points"] = pd.to_numeric(df["points"], errors="coerce")
    df["minutes"] = pd.to_numeric(df["minutes"], errors="coerce")
    df["team_score"] = pd.to_numeric(df["team_score"], errors="coerce")
    df["opponent_team_score"] = pd.to_numeric(df["opponent_team_score"], errors="coerce")
    df = df.dropna(subset=["points"])

    df["day_of_month"] = df["game_date"].dt.day
    df["month"] = df["game_date"].dt.month
    df["pos"] = df["athlete_position_abbreviation"].fillna("U").astype(str)
    df = df.sort_values(["athlete_display_name", "game_date"]).reset_index(drop=True)

    # Last three seasons used for external-factor lookups
    seasons_present = sorted(df["Year"].unique())
    recent_years = seasons_present[-3:] if len(seasons_present) >= 3 else seasons_present
    recent = df[df["Year"].isin(recent_years)]

    opp_pos_def = recent.groupby(["opponent_team_name", "pos"])["points"].mean().to_dict()
    league_pos_avg = recent.groupby("pos")["points"].mean().to_dict()

    pace_df = recent.groupby("opponent_team_name")[["team_score", "opponent_team_score"]].mean()
    pace_df["pace"] = pace_df["team_score"] + pace_df["opponent_team_score"]
    opp_pace = pace_df["pace"].to_dict()

    _CACHE.update({
        "df": df,
        "opp_pos_def": opp_pos_def,
        "opp_pace": opp_pace,
        "league_pos_avg": league_pos_avg,
    })
    return df, opp_pos_def, opp_pace, league_pos_avg


def _player_history(df, player_name):
    return df[df["athlete_display_name"] == player_name].copy()


def _find_todays_opponent(df, player_name, target_date):
    target = pd.to_datetime(target_date)
    rows = df[(df["athlete_display_name"] == player_name) & (df["game_date"] == target)]
    if rows.empty:
        return None, None
    r = rows.iloc[0]
    opp = r.get("opponent_team_name")
    ha = r.get("home_away")
    return (opp if pd.notna(opp) else None,
            str(ha).lower() if pd.notna(ha) else None)


def _last_game_before(player_df, target_date):
    target = pd.to_datetime(target_date)
    prior = player_df[player_df["game_date"] < target]
    if prior.empty:
        return None
    return prior["game_date"].max()


def _recency_weighted_baseline(player_df):
    if player_df.empty:
        return 0.0
    max_year = int(player_df["Year"].max())
    weights = player_df["Year"].apply(
        lambda y: 3.0 if y >= max_year - 1 else (2.0 if y >= max_year - 3 else 1.0)
    ).astype(float).values
    pts = player_df["points"].values
    return float((pts * weights).sum() / weights.sum())


def _detect_monthly_dip_at_dom(player_df, target_date, baseline, dip_pct=0.20):
    """Recurring monthly dip at similar day-of-month. Returns (recurs_2of3, severity)."""
    if baseline <= 0:
        return False, 0.0
    target = pd.to_datetime(target_date)
    target_dom = target.day

    history = player_df[player_df["game_date"] < target].copy()
    if history.empty:
        return False, 0.0

    history["ym"] = history["game_date"].dt.to_period("M")
    last_months = sorted(history["ym"].unique())[-3:]
    if len(last_months) < 2:
        return False, 0.0

    threshold = baseline * (1.0 - dip_pct)
    months_with_dip = 0
    severities = []
    for ym in last_months:
        m = history[history["ym"] == ym]
        nearby = m[(m["day_of_month"] - target_dom).abs() <= 3]
        if nearby.empty:
            continue
        dips = nearby[nearby["points"] < threshold]
        if not dips.empty:
            months_with_dip += 1
            severities.append(float((baseline - dips["points"].mean()) / baseline))

    return months_with_dip >= 2, (float(np.mean(severities)) if severities else 0.0)


def _pattern_strength(player_df):
    """Career-wide regularity of dips across months — proxy for predictability."""
    if player_df.empty:
        return 0.0
    baseline = float(player_df["points"].mean())
    if baseline <= 0:
        return 0.0
    threshold = baseline * 0.80
    by_month = player_df.assign(is_dip=player_df["points"] < threshold).groupby("month")["is_dip"].mean()
    if by_month.empty or len(by_month) < 2:
        return 0.0
    return float(min(1.0, by_month.std() * 2.5))


def _home_away_factor(player_df, today_ha):
    if today_ha not in ("home", "away"):
        return 0.0
    overall = float(player_df["points"].mean())
    if overall <= 0:
        return 0.0
    bucket = player_df[player_df["home_away"].astype(str).str.lower() == today_ha]
    if len(bucket) < 5:
        return 0.0
    return float((bucket["points"].mean() - overall) / overall)


def _rest_factor(player_df, last_game_date, target_date):
    if last_game_date is None:
        return 0.0
    target = pd.to_datetime(target_date)
    rest_days = (target - pd.to_datetime(last_game_date)).days
    if rest_days < 0:
        return 0.0

    pdf = player_df.copy().sort_values("game_date")
    pdf["rest"] = pdf["game_date"].diff().dt.days
    pdf = pdf.dropna(subset=["rest"])
    if pdf.empty:
        return 0.0
    overall = float(pdf["points"].mean())
    if overall <= 0:
        return 0.0

    if rest_days <= 1:
        bucket = pdf[pdf["rest"] <= 1]
    elif rest_days == 2:
        bucket = pdf[pdf["rest"] == 2]
    elif rest_days <= 4:
        bucket = pdf[(pdf["rest"] >= 3) & (pdf["rest"] <= 4)]
    else:
        bucket = pdf[pdf["rest"] >= 5]

    if len(bucket) < 3:
        return 0.0
    return float((bucket["points"].mean() - overall) / overall)


def _recent_form_factor(player_df, target_date, n_recent=5):
    target = pd.to_datetime(target_date)
    history = player_df[player_df["game_date"] < target]
    if len(history) < n_recent:
        return 0.0
    recent = float(history.tail(n_recent)["points"].mean())
    season_window = history.tail(min(len(history), 30))
    season = float(season_window["points"].mean())
    if season <= 0:
        return 0.0
    return float((recent - season) / season)


def _season_phase_factor(player_df, target_date):
    target = pd.to_datetime(target_date)
    if target.month <= 6:
        phase = "early"
    elif target.month <= 8:
        phase = "mid"
    else:
        phase = "late"
    pdf = player_df.copy()
    pdf["phase"] = pdf["month"].apply(
        lambda m: "early" if m <= 6 else ("mid" if m <= 8 else "late")
    )
    overall = float(pdf["points"].mean())
    bucket = pdf[pdf["phase"] == phase]["points"]
    if bucket.empty or overall <= 0:
        return 0.0
    return float((bucket.mean() - overall) / overall)


def _starter_factor(player_df, target_date):
    target = pd.to_datetime(target_date)
    history = player_df[player_df["game_date"] < target].copy()
    if history.empty:
        return 0.0
    starter_recent = history.tail(10)["starter"].astype(str).str.upper().eq("TRUE").mean()
    starter_overall = history["starter"].astype(str).str.upper().eq("TRUE").mean()
    if pd.isna(starter_recent) or pd.isna(starter_overall):
        return 0.0
    return float((starter_recent - starter_overall) * 0.10)


def _opponent_def_factor(opponent_name, position, opp_pos_def, league_pos_avg):
    if opponent_name is None:
        return 0.0
    league = league_pos_avg.get(position)
    opp = opp_pos_def.get((opponent_name, position))
    if league is None or opp is None or league <= 0:
        return 0.0
    return float((opp - league) / league)


def _opp_pace_factor(opponent_name, opp_pace_dict):
    if opponent_name is None or not opp_pace_dict:
        return 0.0
    p = opp_pace_dict.get(opponent_name)
    if p is None:
        return 0.0
    league_pace = float(np.mean(list(opp_pace_dict.values())))
    if league_pace <= 0:
        return 0.0
    return float((p - league_pace) / league_pace) * 0.30


def _line_implied_factor(over_line, baseline):
    if baseline <= 0:
        return 0.0
    return float((over_line - baseline) / baseline) * 0.25


def _recent_mean_std(player_df, target_date, n=10):
    target = pd.to_datetime(target_date)
    history = player_df[player_df["game_date"] < target]
    if len(history) < 3:
        return None, None
    rec = history.tail(n)["points"]
    return float(rec.mean()), float(rec.std(ddof=0))


def _categorize(deviation_pct):
    if deviation_pct <= -0.20:
        return "very bad game"
    if deviation_pct <= -0.08:
        return "below average"
    if deviation_pct < 0.05:
        return "average game"
    if deviation_pct < 0.15:
        return "good game"
    return "very good game"


def _confidence_to_amount(confidence):
    if confidence >= 0.70:
        return 5
    if confidence >= 0.62:
        return 3
    if confidence >= 0.55:
        return 1
    return None


def predict(player):
    name = player["name"]
    date_str = player["date"]
    over_line = float(player["over_line"])
    under_line = float(player["under_line"])

    df, opp_pos_def, opp_pace_dict, league_pos_avg = _load_all_data()
    if df.empty:
        print(f"{name} not in dip results")
        return None

    pdata = _player_history(df, name)
    if len(pdata) < MIN_CAREER_GAMES:
        print(f"{name} not in dip results")
        return None

    baseline = _recency_weighted_baseline(pdata)
    if baseline <= 0:
        print(f"{name} not in dip results")
        return None

    pattern_strength = _pattern_strength(pdata)
    if pattern_strength < 0.05:
        print(f"{name} not in dip results")
        return None

    opponent_name, today_ha = _find_todays_opponent(df, name, date_str)
    last_game = _last_game_before(pdata, date_str)
    pos_mode = pdata["pos"].mode()
    position = pos_mode.iloc[0] if not pos_mode.empty else "U"

    factors = {
        "home_away": _home_away_factor(pdata, today_ha),
        "rest": _rest_factor(pdata, last_game, date_str),
        "recent_form": _recent_form_factor(pdata, date_str),
        "season_phase": _season_phase_factor(pdata, date_str),
        "starter": _starter_factor(pdata, date_str),
        "opp_def": _opponent_def_factor(opponent_name, position, opp_pos_def, league_pos_avg),
        "opp_pace": _opp_pace_factor(opponent_name, opp_pace_dict),
        "line_implied": _line_implied_factor(over_line, baseline),
    }

    dip_recurs, dip_severity = _detect_monthly_dip_at_dom(pdata, date_str, baseline)
    if dip_recurs:
        factors["monthly_dip"] = -dip_severity

    factor_sum = float(np.clip(sum(factors.values()), -0.40, 0.40))
    predicted_raw = baseline * (1.0 + factor_sum)
    predicted_points = int(round(predicted_raw))

    deviation_pct = (predicted_raw - baseline) / max(1.0, baseline)
    bet = "OVER" if predicted_points > over_line else "UNDER"

    rec_mean, rec_std = _recent_mean_std(pdata, date_str, n=10)
    sigma = rec_std if rec_std and rec_std > 0 else float(pdata["points"].std(ddof=0))
    if sigma is None or sigma <= 0:
        sigma = max(2.0, baseline * 0.20)

    if bet == "OVER":
        p_correct = 1.0 - _normal_cdf((over_line - predicted_raw) / sigma)
    else:
        p_correct = _normal_cdf((over_line - predicted_raw) / sigma)

    sign_pred = 1 if predicted_raw > over_line else -1
    agreement = 0.0
    for v in factors.values():
        if v == 0:
            continue
        if (v > 0 and sign_pred > 0) or (v < 0 and sign_pred < 0):
            agreement += abs(v)
        else:
            agreement -= abs(v) * 0.5
    agreement_score = float(np.tanh(agreement * 3.0)) * 0.5 + 0.5

    blended = 0.65 * p_correct + 0.20 * agreement_score + 0.15 * pattern_strength
    confidence = float(max(0.0, min(0.95, blended)))

    note = _categorize(deviation_pct)
    if rec_mean is not None and rec_std and rec_std > 0:
        sd_below = (rec_mean - predicted_raw) / rec_std
        if sd_below >= LOW_OUTPUT_SD and dip_recurs and confidence >= LOW_OUTPUT_CONF:
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

        print(f"\n--- Running prediction for {player['name']} on {player['date']} ---")
        result = predict(player)
        if result is None:
            print(f"Prediction not generated for {player['name']}")
            continue
        print(f"Prediction successful for {player['name']}: {result['predicted_points']} pts")
        print(f"{player['name']} predicted points: {result['predicted_points']}")
        print(f"Bet: {result['bet']}, Over line: {result['over_line']}, Under line: {result['under_line']}")
"""
