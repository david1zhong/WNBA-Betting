"""
Walk-forward backtester for the WNBA pipeline.

Two reports:
  1. Per-source-model historical performance — hit rate, ROI, calibration by
     stake tier, OVER vs UNDER, line-gap effect.
  2. CLC4_SELECTIVE walk-forward simulation — for each (player, date) where
     enough source models predicted and actuals exist, recompute what CLC4
     would have bet using ONLY rolling win rates from dates strictly prior.

Why we read from the predictions table instead of re-running each source
model's predict() with truncated playerbox data:
  - Rows in `predictions` were written when the source models ran the morning
    of the game — they had no future data. Reading the persisted rows gives
    the same walk-forward semantics the production pipeline already produced.
  - Re-running the source models against truncated CSVs would require each
    model to accept a target-date cutoff. The project spec forbids changes to
    CL1/CL2/CL3_LEARN/CH1_LEARN/CLC1/CLC2/CLC3_LEARN, so we read instead.

Read-only. No DB writes. Run via:  python backtest.py
"""
import os
import sys
import math
import pandas as pd
import numpy as np
import psycopg2
from dotenv import load_dotenv

load_dotenv()


BREAK_EVEN = 0.524           # -110/-110 break-even
LOW_N_THRESHOLD = 50         # any stat computed on fewer rows is flagged low-confidence
TARGET_BETS_PER_DAY = (3, 10)  # CLC4 simulation success window from the spec


def _connect():
    return psycopg2.connect(
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        host=os.getenv("DB_HOST"),
        port=int(os.getenv("DB_PORT", 5432)),
    )


def _load_predictions():
    """All graded predictions. profit and amount preserved as-is from DB."""
    conn = _connect()
    try:
        df = pd.read_sql(
            """
            SELECT player_name, model_name, date,
                   predicted_pts, actual_pts,
                   over_line, under_line, over_odds, under_odds,
                   bet, result, amount, profit
            FROM predictions
            WHERE bet IN ('OVER', 'UNDER')
            ORDER BY date
            """,
            conn,
        )
    finally:
        conn.close()

    df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0)
    df["profit"] = pd.to_numeric(df["profit"], errors="coerce").fillna(0)
    for col in ("predicted_pts", "actual_pts", "over_line", "under_line", "over_odds", "under_odds"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


def _fmt_hit_rate(df, low_n=LOW_N_THRESHOLD):
    """`52.4% (n=145)` with `[low-N]` suffix if sample is small."""
    n = len(df)
    if n == 0:
        return "n/a"
    won = (df["result"] == "WON").sum()
    lost = (df["result"] == "LOST").sum()
    graded = won + lost
    if graded == 0:
        return f"no graded bets (n={n})"
    rate = won / graded
    flag = " [LOW-N]" if graded < low_n else ""
    return f"{rate*100:.1f}% ({won}/{graded}){flag}"


# ----------------------------------------------------------------- per-model

def _per_model_report(graded, model):
    sub = graded[graded["model_name"] == model].copy()
    if sub.empty:
        return f"\n=== {model} ===\nNo graded bets.\n"

    won = (sub["result"] == "WON").sum()
    lost = (sub["result"] == "LOST").sum()
    n = won + lost
    if n == 0:
        return f"\n=== {model} ===\nHas rows but none graded.\n"

    hit = won / n
    profit = sub["profit"].sum()
    staked = sub["amount"].sum()
    roi = (profit / staked * 100) if staked > 0 else float("nan")

    lines = []
    lines.append(f"\n=== {model} ===")
    lines.append(f"  Bets graded: {n}  ({won}W / {lost}L)")
    lines.append(f"  Hit rate: {hit*100:.1f}%   Break-even at -110: {BREAK_EVEN*100:.1f}%   "
                 f"{'✓ above' if hit > BREAK_EVEN else '✗ below'} break-even")
    lines.append(f"  Total staked: ${staked:,.2f}   Profit: ${profit:+,.2f}   ROI: {roi:+.2f}%")
    lines.append(f"  Sample size flag: {'OK' if n >= LOW_N_THRESHOLD else f'LOW (<{LOW_N_THRESHOLD} bets)'}")

    # By stake tier
    lines.append("\n  By stake tier:")
    tier_rates = []
    for tier in [1, 2, 3, 4, 5]:
        t = sub[sub["amount"] == tier]
        if len(t) > 0:
            rate_str = _fmt_hit_rate(t)
            graded_n = (t["result"].isin(["WON", "LOST"])).sum()
            lines.append(f"    ${tier}: {rate_str}")
            if graded_n >= 10:
                t_won = (t["result"] == "WON").sum()
                t_n = (t["result"].isin(["WON", "LOST"])).sum()
                tier_rates.append((tier, t_n, t_won / t_n if t_n > 0 else 0.0))
        else:
            lines.append(f"    ${tier}: no bets")

    # Calibration: do higher stakes win more often?
    if len(tier_rates) >= 2:
        # Spearman-ish rank check: are the tier rates monotone non-decreasing?
        non_decreasing = all(
            tier_rates[i][2] <= tier_rates[i + 1][2] + 1e-6
            for i in range(len(tier_rates) - 1)
        )
        if non_decreasing:
            lines.append("    Calibration: ✓ rates non-decreasing across tiers")
        else:
            lines.append("    Calibration: ✗ NOT monotone — model's amount signal may be noise")
    else:
        lines.append("    Calibration: insufficient data (<10 bets in 2+ tiers)")

    # By direction
    lines.append("\n  By direction:")
    for direction in ["OVER", "UNDER"]:
        d = sub[sub["bet"] == direction]
        lines.append(f"    {direction}: {_fmt_hit_rate(d)}")

    # By line gap
    sub["_gap"] = sub["under_line"].fillna(0) - sub["over_line"].fillna(0)
    no_gap = sub[sub["_gap"].between(-0.01, 0.01)]
    with_gap = sub[sub["_gap"] >= 1.0]
    lines.append("\n  By line gap:")
    lines.append(f"    Same line (O=U):  {_fmt_hit_rate(no_gap)}")
    lines.append(f"    Split (gap≥1.0):  {_fmt_hit_rate(with_gap)}")

    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------- CLC4 sim

def _simulate_clc4(graded):
    """
    Walk-forward simulation of CLC4_SELECTIVE on historical data.

    For each (player, date) where ≥3 source models predicted and actuals
    exist, compute what CLC4 would have decided using ONLY rolling win
    rates from prior dates. Tally hypothetical profit/loss.
    """
    try:
        from models.model_CLC4_SELECTIVE import (
            decide, SOURCE_MODELS, MIN_SOURCE_PREDS, ROLLING_HISTORY_N,
            MIN_CAREER_GAMES, _american_to_decimal,
        )
    except Exception as e:
        return None, f"Could not import CLC4_SELECTIVE: {e}"

    g = graded[graded["model_name"].isin(SOURCE_MODELS)].copy()
    g["date"] = pd.to_datetime(g["date"])
    g = g.sort_values("date").reset_index(drop=True)
    if g.empty:
        return None, "No source-model rows found."

    # Career-games guard: use the predictions table itself as a proxy for
    # player tenure (a player with ≥40 graded prediction rows has at least
    # 40 games of pro history). This is a backtester-only approximation —
    # production CLC4 uses the playerbox CSV count, which is the real source.
    career_counts = g.groupby(g["player_name"].str.lower().str.strip())["date"].nunique().to_dict()

    grouped = g.groupby([
        g["player_name"].str.lower().str.strip(),
        g["date"].dt.date,
    ])

    results = []
    skipped = {
        "too_few_sources": 0,
        "rookie": 0,
        "missing_actual": 0,
        "skip_decision": 0,
    }

    for (player_lower, date_val), group in grouped:
        if career_counts.get(player_lower, 0) < MIN_CAREER_GAMES:
            skipped["rookie"] += 1
            continue

        source_preds = {}
        for _, row in group.iterrows():
            if row["bet"] not in ("OVER", "UNDER"):
                continue
            if pd.isna(row["predicted_pts"]):
                continue
            source_preds[row["model_name"]] = {
                "pred": float(row["predicted_pts"]),
                "bet": row["bet"],
            }
        if len(source_preds) < MIN_SOURCE_PREDS:
            skipped["too_few_sources"] += 1
            continue

        # Win rates AS OF this date — strictly prior
        target_dt = pd.Timestamp(date_val)
        prior = g[g["date"] < target_dt]
        prior = prior[prior["result"].isin(["WON", "LOST"])]
        source_win_rates = {}
        for model in SOURCE_MODELS:
            mp = prior[prior["model_name"] == model].tail(ROLLING_HISTORY_N)
            if mp.empty:
                source_win_rates[model] = None
            else:
                source_win_rates[model] = (mp["result"] == "WON").mean()

        first = group.iloc[0]
        try:
            over_line = float(first["over_line"])
            under_line = float(first["under_line"])
        except (TypeError, ValueError):
            continue
        over_odds = first["over_odds"] if pd.notna(first["over_odds"]) else -110
        under_odds = first["under_odds"] if pd.notna(first["under_odds"]) else -110

        decision = decide(source_preds, source_win_rates, over_line, under_line, over_odds, under_odds)
        if "_skip_reason" in decision:
            skipped["skip_decision"] += 1
            continue

        # Need an actual to score the hypothetical bet
        actuals = group["actual_pts"].dropna()
        if actuals.empty:
            skipped["missing_actual"] += 1
            continue
        actual = float(actuals.iloc[0])

        bet = decision["bet"]
        amount = decision["amount"]
        if bet == "OVER":
            line, odds = over_line, over_odds
            won = actual > line
            push = abs(actual - line) < 1e-9
        else:
            line, odds = under_line, under_odds
            won = actual < line
            push = abs(actual - line) < 1e-9

        decimal_odds = _american_to_decimal(odds)
        if push:
            profit = 0.0
            outcome = "PUSH"
        elif won:
            profit = amount * (decimal_odds - 1)
            outcome = "WON"
        else:
            profit = -float(amount)
            outcome = "LOST"

        results.append({
            "player": player_lower,
            "date": pd.Timestamp(date_val),
            "bet": bet,
            "amount": amount,
            "note": decision["note"],
            "result": outcome,
            "profit": profit,
            "ensemble_prob": decision["_diag"]["ensemble_prob"],
            "break_even": decision["_diag"]["break_even"],
            "margin": decision["_diag"]["margin"],
            "n_voting": decision["_diag"]["n_voting"],
            "predicted_points": decision["predicted_points"],
            "actual_pts": actual,
            "over_line": over_line,
            "under_line": under_line,
        })

    sim_df = pd.DataFrame(results)
    return sim_df, skipped


def _clc4_report(sim_df, skipped):
    if sim_df is None:
        return "\n=== CLC4_SELECTIVE walk-forward sim ===\nUnavailable.\n"
    if sim_df.empty:
        return (
            "\n=== CLC4_SELECTIVE walk-forward sim ===\n"
            f"Zero bets generated. Skips: {skipped}\n"
            "If too restrictive, lower EV_MARGIN (0.04 default) in the model.\n"
        )

    n = len(sim_df)
    won = (sim_df["result"] == "WON").sum()
    lost = (sim_df["result"] == "LOST").sum()
    push = (sim_df["result"] == "PUSH").sum()
    graded = won + lost
    hit = won / graded if graded > 0 else float("nan")
    profit = sim_df["profit"].sum()
    staked = sim_df["amount"].sum()
    roi = (profit / staked * 100) if staked > 0 else float("nan")

    days = sim_df["date"].dt.normalize().nunique()
    bets_per_day = n / days if days > 0 else 0.0

    lines = []
    lines.append("\n=== CLC4_SELECTIVE walk-forward sim ===")
    lines.append(f"  Bets simulated: {n}  ({won}W / {lost}L / {push}P)")
    lines.append(f"  Hit rate: {hit*100:.1f}%   Break-even at -110: {BREAK_EVEN*100:.1f}%")
    lines.append(f"  Total staked: ${staked:,.2f}   Profit: ${profit:+,.2f}   ROI: {roi:+.2f}%")
    lines.append(f"  Days with bets: {days}   Avg bets/day: {bets_per_day:.1f}")

    target_lo, target_hi = TARGET_BETS_PER_DAY
    if not (target_lo <= bets_per_day <= target_hi):
        if bets_per_day > target_hi:
            lines.append(f"  ⚠ Bets/day above target ({target_hi}). EV_MARGIN may be too lenient — try +0.06.")
        else:
            lines.append(f"  ⚠ Bets/day below target ({target_lo}). EV_MARGIN may be too strict — try +0.03.")

    # By edge tier
    lines.append("\n  By edge tier:")
    for note in ["High Edge", "Medium Edge", "Small Edge"]:
        t = sim_df[sim_df["note"] == note]
        if len(t) > 0:
            tw = (t["result"] == "WON").sum()
            tl = (t["result"] == "LOST").sum()
            tn = tw + tl
            rate = tw / tn if tn > 0 else float("nan")
            tp = t["profit"].sum()
            lines.append(f"    {note}: {tw}/{tn} ({rate*100:.1f}%)   profit ${tp:+,.2f}")

    # By stake
    lines.append("\n  By stake tier:")
    for tier in [1, 2, 3, 4, 5]:
        t = sim_df[sim_df["amount"] == tier]
        if len(t) > 0:
            tw = (t["result"] == "WON").sum()
            tl = (t["result"] == "LOST").sum()
            tn = tw + tl
            rate = tw / tn if tn > 0 else float("nan")
            tp = t["profit"].sum()
            lines.append(f"    ${tier}: {tw}/{tn} ({rate*100:.1f}%)   profit ${tp:+,.2f}")

    # By direction
    lines.append("\n  By direction:")
    for direction in ["OVER", "UNDER"]:
        d = sim_df[sim_df["bet"] == direction]
        if len(d) > 0:
            tw = (d["result"] == "WON").sum()
            tn = (d["result"].isin(["WON", "LOST"])).sum()
            rate = tw / tn if tn > 0 else float("nan")
            tp = d["profit"].sum()
            lines.append(f"    {direction}: {tw}/{tn} ({rate*100:.1f}%)   profit ${tp:+,.2f}")

    lines.append(f"\n  Skips: {skipped}")
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------- main

def main():
    print("Loading predictions from database...")
    df = _load_predictions()
    graded = df[df["result"].isin(["WON", "LOST"])].copy()
    if graded.empty:
        print("No graded predictions found. Nothing to backtest.")
        return

    print(f"Loaded {len(df)} total prediction rows, {len(graded)} graded.")
    print(f"Date range: {graded['date'].min().date()} to {graded['date'].max().date()}")

    print("\n" + "=" * 72)
    print("PER-MODEL HISTORICAL PERFORMANCE")
    print("=" * 72)
    models = sorted(df["model_name"].unique())
    for m in models:
        print(_per_model_report(graded, m))

    # Combined summary
    print("\n" + "-" * 72)
    print("Models clearing -110 break-even (≥50 graded bets):")
    any_clearing = False
    for m in models:
        sub = graded[graded["model_name"] == m]
        n = (sub["result"].isin(["WON", "LOST"])).sum()
        if n < LOW_N_THRESHOLD:
            continue
        rate = (sub["result"] == "WON").mean()
        if rate > BREAK_EVEN:
            any_clearing = True
            print(f"  ✓ {m}: {rate*100:.1f}% on {n} bets")
    if not any_clearing:
        print("  (none with sufficient sample) — CLC4 will weight every model 0 until one clears")

    print("\n" + "=" * 72)
    print("CLC4_SELECTIVE WALK-FORWARD SIMULATION")
    print("=" * 72)
    sim_df, skipped = _simulate_clc4(graded)
    print(_clc4_report(sim_df, skipped))


if __name__ == "__main__":
    main()
