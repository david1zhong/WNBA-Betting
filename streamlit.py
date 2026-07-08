import streamlit as st
import pandas as pd
import pytz
import numpy as np
from datetime import datetime, timedelta
import os
from sqlalchemy import create_engine

DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(DATABASE_URL)


@st.cache_data(ttl=60)
def load_data():
    return pd.read_sql("SELECT * FROM predictions", engine)


df = load_data()


def highlight_result(val):
    if val == "WON":
        return "color: green; font-weight: bold;"
    elif val == "LOST":
        return "color: red; font-weight: bold;"
    elif val == "VOID":
        return "color: gray; font-weight: bold;"
    return ""


num_cols = df.select_dtypes(include="number").columns
df[num_cols] = df[num_cols].astype(float)

st.set_page_config(layout="wide")
st.title("WNBA Betting")

if "id" in df.columns:
    df = df.drop(columns=["id"])

def smart_format(x):
    if pd.isna(x):
        return ""
    if float(x).is_integer():
        return str(int(x))
    return str(x)


def render_season_table(season_df, key_prefix):
    """Render a season's predictions table with combinable filters.

    Each multiselect defaults to empty = no filter on that column. Pick one
    or more values to narrow. Multiple filters AND together (model AND bet
    AND result AND player AND date). Click any column header in the table
    itself to sort — sort + filter combine."""
    with st.expander("Filters", expanded=False):
        # Two rows of three filters each — keeps each multiselect usable width
        # instead of cramming six into one row.
        r1c1, r1c2, r1c3 = st.columns(3)
        r2c1, r2c2, r2c3 = st.columns(3)
        with r1c1:
            models_avail = sorted(season_df["model_name"].dropna().unique())
            models_sel = st.multiselect(
                "Model(s)", models_avail, default=[],
                placeholder="All models", key=f"{key_prefix}_models",
            )
        with r1c2:
            # Dates as YYYY-MM-DD strings, recent first — typical use is
            # "today" or "yesterday" which sit at the top.
            dates_avail = sorted(
                pd.to_datetime(season_df["date"]).dt.strftime("%Y-%m-%d").unique(),
                reverse=True,
            )
            dates_sel = st.multiselect(
                "Date(s)", dates_avail, default=[],
                placeholder="All dates", key=f"{key_prefix}_dates",
            )
        with r1c3:
            players_avail = sorted(season_df["player_name"].dropna().unique())
            players_sel = st.multiselect(
                "Player(s)", players_avail, default=[],
                placeholder="All players", key=f"{key_prefix}_players",
            )
        with r2c1:
            bets_sel = st.multiselect(
                "Bet", ["OVER", "UNDER"], default=[],
                placeholder="All bets", key=f"{key_prefix}_bets",
            )
        with r2c2:
            results_avail = [
                r for r in ["WON", "LOST", "DNP", "VOID"]
                if r in season_df["result"].dropna().unique()
            ]
            results_sel = st.multiselect(
                "Result", results_avail, default=[],
                placeholder="All results", key=f"{key_prefix}_results",
            )
        with r2c3:
            _amt_numeric = pd.to_numeric(season_df["amount"], errors="coerce")
            amounts_avail = sorted(_amt_numeric.dropna().astype(int).unique().tolist())
            amount_options = [f"${a}" for a in amounts_avail]
            if _amt_numeric.isna().any():
                amount_options.append("No Bet")
            amount_sel = st.multiselect(
                "Amount", amount_options, default=[],
                placeholder="All amounts", key=f"{key_prefix}_amount",
            )

    filtered = season_df
    if models_sel:
        filtered = filtered[filtered["model_name"].isin(models_sel)]
    if dates_sel:
        date_strs = pd.to_datetime(filtered["date"]).dt.strftime("%Y-%m-%d")
        filtered = filtered[date_strs.isin(dates_sel)]
    if bets_sel:
        filtered = filtered[filtered["bet"].isin(bets_sel)]
    if results_sel:
        filtered = filtered[filtered["result"].isin(results_sel)]
    if players_sel:
        filtered = filtered[filtered["player_name"].isin(players_sel)]
    if amount_sel:
        # "$1" -> 1, ..., "No Bet" -> include NaN rows
        selected_amts = []
        include_none = False
        for s in amount_sel:
            if s == "No Bet":
                include_none = True
            else:
                try:
                    selected_amts.append(int(s.lstrip("$")))
                except ValueError:
                    continue
        amt_col = pd.to_numeric(filtered["amount"], errors="coerce")
        mask = amt_col.isin(selected_amts)
        if include_none:
            mask = mask | amt_col.isna()
        filtered = filtered[mask]

    # Summary metrics for the current view — most useful when filters are
    # combined (e.g. one model + one date), but useful unfiltered too.
    n_total = len(season_df)
    n_filt = len(filtered)
    n_players = filtered["player_name"].nunique()
    n_overs = int((filtered["bet"] == "OVER").sum())
    n_unders = int((filtered["bet"] == "UNDER").sum())
    wagered = float(pd.to_numeric(filtered["amount"], errors="coerce").fillna(0).sum())
    n_won = int((filtered["result"] == "WON").sum())
    n_lost = int((filtered["result"] == "LOST").sum())
    n_dnp = int((filtered["result"] == "DNP").sum())

    if n_filt != n_total:
        prefix = f"Showing {n_filt:,} of {n_total:,} rows"
    else:
        prefix = f"Showing all {n_total:,} rows"
    st.caption(
        " • ".join([
            prefix,
            f"{n_players} player{'s' if n_players != 1 else ''}",
            f"{n_overs} OVER / {n_unders} UNDER",
            f"${wagered:,.2f} wagered",
            f"WON: {n_won}/{n_filt}",
            f"LOST: {n_lost}/{n_filt}",
            f"DNP: {n_dnp}/{n_filt}",
        ])
    )

    styled = (
        filtered.style
        .map(highlight_result, subset=["result"])
        .format({col: smart_format for col in num_cols})
        .hide(axis="index")
    )
    st.dataframe(styled, use_container_width=True, height=600)


_dates = pd.to_datetime(df["date"])
_years = _dates.dt.year
df_2026 = df[_years == 2026]
df_2025 = df[_years == 2025]

_today_date = datetime.now(pytz.timezone("US/Eastern")).date()
_today_df = df[_dates.dt.date == _today_date]
_today_rows = len(_today_df)
_today_players = _today_df["player_name"].nunique()
_today_unders = int((_today_df["bet"] == "UNDER").sum())
_today_overs = int((_today_df["bet"] == "OVER").sum())
_today_wagered = float(pd.to_numeric(_today_df["amount"], errors="coerce").fillna(0).sum())

_today_summary = (
    f" [Today: {_today_rows} Rows | {_today_players} Players | "
    f"{_today_unders} Unders | {_today_overs} Overs | "
    f"${_today_wagered:,.2f} Wagered]"
)

st.subheader("2026 Season" + _today_summary)
if df_2026.empty:
    st.write("No predictions yet.")
else:
    render_season_table(df_2026, key_prefix="season_2026")

st.subheader("2025 Season")
if df_2025.empty:
    st.write("No predictions yet.")
else:
    render_season_table(df_2025, key_prefix="season_2025")


st.subheader("Wins and Losses per Model Yesterday")

eastern = pytz.timezone("US/Eastern")
yesterday_date = (datetime.now(eastern) - timedelta(days=1)).date()

# Rows with a real stake vs "paper picks" (row inserted with a bet but no
# amount). Paper picks are shown separately so W/L records stay comparable
# across models — some models emit many unstaked picks, others none.
_staked_mask = pd.to_numeric(df["amount"], errors="coerce").fillna(0) > 0


def _wl_counts_table(frame):
    counts = frame.groupby(["model_name", "result"]).size().unstack(fill_value=0)
    for col in ["WON", "LOST"]:
        if col not in counts.columns:
            counts[col] = 0
    counts = counts[["WON", "LOST"]]
    totals = counts.sum(axis=1).replace(0, np.nan)
    pct = (counts.div(totals, axis=0) * 100).round(1).fillna(0).astype(str) + "%"
    return counts.astype(str) + " (" + pct + ")"


yesterday_df = df[
    (pd.to_datetime(df["date"]).dt.date == yesterday_date) &
    (df["result"].isin(["WON", "LOST"]))
]

_yest_staked = yesterday_df[_staked_mask.reindex(yesterday_df.index, fill_value=False)]
if _yest_staked.empty:
    model_names = df["model_name"].unique()
    combined_yesterday = pd.DataFrame(
        {"model_name": model_names, "WON": "0 (0.0%)", "LOST": "0 (0.0%)"}
    ).set_index("model_name")
else:
    combined_yesterday = _wl_counts_table(_yest_staked)

st.caption("Staked bets only.")
st.table(combined_yesterday)

_yest_paper = yesterday_df[~_staked_mask.reindex(yesterday_df.index, fill_value=False)]
with st.expander("Paper picks yesterday (no stake)"):
    if _yest_paper.empty:
        st.write("None.")
    else:
        st.table(_wl_counts_table(_yest_paper))



st.subheader("Wins and Losses per Model")

_wl_all = df[df["result"].isin(["WON", "LOST"])].copy()
_wl_all["_year"] = pd.to_datetime(_wl_all["date"]).dt.year
_wl_models = sorted(_wl_all["model_name"].unique())
_wl_filtered = _wl_all[_staked_mask.reindex(_wl_all.index, fill_value=False)]
_wl_paper = _wl_all[~_staked_mask.reindex(_wl_all.index, fill_value=False)]


def _wl_breakdown(season_df, models_index):
    counts = (
        season_df.groupby(["model_name", "result"])
        .size()
        .unstack(fill_value=0)
        .reindex(models_index, fill_value=0)
    )
    for col in ["WON", "LOST"]:
        if col not in counts.columns:
            counts[col] = 0
    counts = counts[["WON", "LOST"]]
    raw_totals = counts.sum(axis=1)
    totals_safe = raw_totals.replace(0, np.nan)
    pct = (counts.div(totals_safe, axis=0) * 100).round(1).fillna(0).astype(str) + "%"
    out = counts.astype(str) + " (" + pct + ")"
    no_data = raw_totals == 0
    if no_data.any():
        out.loc[no_data, :] = "—"
    return out


_wl_combined = pd.concat(
    {
        "2026": _wl_breakdown(_wl_filtered[_wl_filtered["_year"] == 2026], _wl_models),
        "2025": _wl_breakdown(_wl_filtered[_wl_filtered["_year"] == 2025], _wl_models),
        "Total": _wl_breakdown(_wl_filtered, _wl_models),
    },
    axis=1,
)
st.caption("Staked bets only — paper picks (no bet amount) are in the expander below.")
st.table(_wl_combined)

with st.expander("Paper picks (no stake)"):
    if _wl_paper.empty:
        st.write("None.")
    else:
        _wl_paper_combined = pd.concat(
            {
                "2026": _wl_breakdown(_wl_paper[_wl_paper["_year"] == 2026], _wl_models),
                "2025": _wl_breakdown(_wl_paper[_wl_paper["_year"] == 2025], _wl_models),
                "Total": _wl_breakdown(_wl_paper, _wl_models),
            },
            axis=1,
        )
        st.table(_wl_paper_combined)



df['profit'] = df['profit'].fillna(0)
df['amount'] = df['amount'].fillna(0)
df['date'] = pd.to_datetime(df['date'])
# Same Eastern-time "yesterday" as the win/loss tables above — a naive
# datetime.now() on a UTC host flips to the wrong day after 8 PM ET.
yesterday = yesterday_date
df_yesterday = df[df['date'].dt.date == yesterday]

def summarize(df_input):
    grouped = df_input.groupby('model_name').agg(
        bet_amount=('amount', 'sum'),
        winnings_amount=('profit', lambda x: x[x > 0].sum()),
        losses_amount=('profit', lambda x: x[x < 0].sum()),
        total_profit=('profit', 'sum')
    ).reset_index()

    for col in ['bet_amount', 'winnings_amount', 'losses_amount', 'total_profit']:
        if col not in grouped.columns:
            grouped[col] = 0

    grouped = grouped[['model_name', 'bet_amount', 'winnings_amount', 'losses_amount', 'total_profit']]

    def currency(x):
        return f"${x:,.2f}"

    numeric_cols = ['bet_amount', 'winnings_amount', 'losses_amount', 'total_profit']
    styled_df = grouped.style.format({col: currency for col in numeric_cols})

    def highlight_positive(val):
        return "color: green; font-weight: bold;" if val > 0 else ''

    def highlight_negative(val):
        return "color: red; font-weight: bold;" if val < 0 else ''

    def highlight_profit(val):
        if val > 0:
            return "color: green; font-weight: bold;"
        elif val < 0:
            return "color: red; font-weight: bold;"
        return ''

    styled_df = styled_df.map(highlight_positive, subset=['winnings_amount'])
    styled_df = styled_df.map(highlight_negative, subset=['losses_amount'])
    styled_df = styled_df.map(highlight_profit, subset=['total_profit'])

    return styled_df


profit_per_model_yesterday = summarize(df_yesterday)
st.subheader(f"Profit Per Model - {yesterday}")
st.dataframe(profit_per_model_yesterday)


def _profit_raw(season_df, models_idx):
    cols = ["bet_amount", "winnings_amount", "losses_amount", "total_profit"]
    if season_df.empty:
        return pd.DataFrame(
            np.nan,
            index=pd.Index(models_idx, name="model_name"),
            columns=cols,
        )
    g = season_df.groupby("model_name").agg(
        bet_amount=("amount", "sum"),
        winnings_amount=("profit", lambda x: x[x > 0].sum()),
        losses_amount=("profit", lambda x: x[x < 0].sum()),
        total_profit=("profit", "sum"),
    )
    g = g.reindex(models_idx)
    zero_mask = (
        g["bet_amount"].fillna(0).eq(0)
        & g["winnings_amount"].fillna(0).eq(0)
        & g["losses_amount"].fillna(0).eq(0)
    )
    g.loc[zero_mask, :] = np.nan
    g.index.name = "model_name"
    return g[cols]


_models_idx = sorted(df["model_name"].unique())
_year_series = df["date"].dt.year

_profit_combined = pd.concat(
    {
        "2026": _profit_raw(df[_year_series == 2026], _models_idx),
        "2025": _profit_raw(df[_year_series == 2025], _models_idx),
        "Total": _profit_raw(df, _models_idx),
    },
    axis=1,
)


def _currency(v):
    if not isinstance(v, (int, float, np.floating)) or pd.isna(v):
        return "—"
    return f"${v:,.2f}"


def _hl_pos(v):
    return "color: green; font-weight: bold;" if isinstance(v, (int, float, np.floating)) and v > 0 else ""


def _hl_neg(v):
    return "color: red; font-weight: bold;" if isinstance(v, (int, float, np.floating)) and v < 0 else ""


def _hl_profit(v):
    if not isinstance(v, (int, float, np.floating)) or pd.isna(v):
        return ""
    if v > 0:
        return "color: green; font-weight: bold;"
    if v < 0:
        return "color: red; font-weight: bold;"
    return ""


_winnings_cols = [c for c in _profit_combined.columns if c[1] == "winnings_amount"]
_losses_cols = [c for c in _profit_combined.columns if c[1] == "losses_amount"]
_profit_cols = [c for c in _profit_combined.columns if c[1] == "total_profit"]

_styled_profit = (
    _profit_combined.style
    .format(_currency)
    .map(_hl_pos, subset=_winnings_cols)
    .map(_hl_neg, subset=_losses_cols)
    .map(_hl_profit, subset=_profit_cols)
)
st.subheader("Profit Per Model")
st.dataframe(_styled_profit, use_container_width=True)


def _build_fade_df(source_df):
    """Return a fade view of the predictions: bet flipped, result and profit
    recomputed against the opposite side's line and odds."""
    fade = source_df.copy()

    fade_bet = source_df["bet"].map({"OVER": "UNDER", "UNDER": "OVER"})
    fade["bet"] = fade_bet
    is_over = fade_bet == "OVER"

    fade_line = pd.to_numeric(
        pd.Series(
            np.where(is_over, source_df["over_line"], source_df["under_line"]),
            index=source_df.index,
        ),
        errors="coerce",
    )
    fade_odds = pd.to_numeric(
        pd.Series(
            np.where(is_over, source_df["over_odds"], source_df["under_odds"]),
            index=source_df.index,
        ),
        errors="coerce",
    ).fillna(-110).replace(0, -110)

    actual = pd.to_numeric(source_df["actual_pts"], errors="coerce")

    res = pd.Series(np.nan, index=source_df.index, dtype=object)
    gradable = actual.notna() & fade_line.notna() & fade_bet.notna()
    over_g = gradable & is_over
    under_g = gradable & ~is_over
    res[over_g & (actual > fade_line)] = "WON"
    res[over_g & (actual < fade_line)] = "LOST"
    res[under_g & (actual < fade_line)] = "WON"
    res[under_g & (actual > fade_line)] = "LOST"
    # Push: exactly on the line is a void, not a loss.
    res[gradable & (actual == fade_line)] = "VOID"

    _carry = source_df["result"].isin(["DNP", "VOID"])
    res[_carry] = source_df["result"][_carry]
    fade["result"] = res

    amount = pd.to_numeric(source_df["amount"], errors="coerce").fillna(0)
    dec = pd.Series(
        np.where(fade_odds > 0, fade_odds / 100.0 + 1.0, 100.0 / fade_odds.abs() + 1.0),
        index=source_df.index,
    )
    profit = pd.Series(np.nan, index=source_df.index, dtype=float)
    profit[res == "WON"] = amount[res == "WON"] * (dec[res == "WON"] - 1.0)
    profit[res == "LOST"] = -amount[res == "LOST"]
    fade["profit"] = profit.fillna(0.0).round(2)

    return fade


fade_df = _build_fade_df(df)

st.subheader("Wins and Losses per Model Yesterday FADE")
_fade_staked = pd.to_numeric(fade_df["amount"], errors="coerce").fillna(0) > 0
_fade_yesterday_df = fade_df[
    (fade_df["date"].dt.date == yesterday_date)
    & (fade_df["result"].isin(["WON", "LOST"]))
    & _fade_staked
]
if _fade_yesterday_df.empty:
    _fade_counts_yest = pd.DataFrame(
        {"model_name": df["model_name"].unique(), "WON": 0, "LOST": 0}
    ).set_index("model_name")
else:
    _fade_counts_yest = (
        _fade_yesterday_df.groupby(["model_name", "result"]).size().unstack(fill_value=0)
    )
    for col in ["WON", "LOST"]:
        if col not in _fade_counts_yest.columns:
            _fade_counts_yest[col] = 0
    _fade_counts_yest = _fade_counts_yest[["WON", "LOST"]]
_fade_totals_yest = _fade_counts_yest.sum(axis=1)
_fade_pct_yest = (
    _fade_counts_yest.div(_fade_totals_yest.replace(0, np.nan), axis=0)
    .multiply(100)
    .round(1)
    .fillna(0)
    .astype(str)
    + "%"
)
st.table(_fade_counts_yest.astype(str) + " (" + _fade_pct_yest + ")")

st.subheader("Wins and Losses per Model FADE")
_fade_wl_filtered = fade_df[fade_df["result"].isin(["WON", "LOST"]) & _fade_staked].copy()
_fade_wl_filtered["_year"] = pd.to_datetime(_fade_wl_filtered["date"]).dt.year
_fade_wl_combined = pd.concat(
    {
        "2026": _wl_breakdown(_fade_wl_filtered[_fade_wl_filtered["_year"] == 2026], _wl_models),
        "2025": _wl_breakdown(_fade_wl_filtered[_fade_wl_filtered["_year"] == 2025], _wl_models),
        "Total": _wl_breakdown(_fade_wl_filtered, _wl_models),
    },
    axis=1,
)
st.table(_fade_wl_combined)

st.subheader(f"Profit per Model Yesterday FADE - {yesterday}")
_fade_df_yesterday = fade_df[fade_df["date"].dt.date == yesterday]
st.dataframe(summarize(_fade_df_yesterday))

st.subheader("Profit Per Model FADE")
_fade_year_series = fade_df["date"].dt.year
_fade_profit_combined = pd.concat(
    {
        "2026": _profit_raw(fade_df[_fade_year_series == 2026], _models_idx),
        "2025": _profit_raw(fade_df[_fade_year_series == 2025], _models_idx),
        "Total": _profit_raw(fade_df, _models_idx),
    },
    axis=1,
)
_fade_winnings_cols = [c for c in _fade_profit_combined.columns if c[1] == "winnings_amount"]
_fade_losses_cols = [c for c in _fade_profit_combined.columns if c[1] == "losses_amount"]
_fade_profit_cols = [c for c in _fade_profit_combined.columns if c[1] == "total_profit"]
_styled_fade_profit = (
    _fade_profit_combined.style
    .format(_currency)
    .map(_hl_pos, subset=_fade_winnings_cols)
    .map(_hl_neg, subset=_fade_losses_cols)
    .map(_hl_profit, subset=_fade_profit_cols)
)
st.dataframe(_styled_fade_profit, use_container_width=True)


st.subheader("Daily Profit per Model")
df['date'] = pd.to_datetime(df['date'])

# Chart only days with graded bets, but keep genuine break-even ($0) days —
# filtering on profit != 0 silently dropped them.
_graded_rows = df[df["result"].isin(["WON", "LOST", "VOID"])]
daily_profit = _graded_rows.groupby(["model_name", "date"])["profit"].sum().reset_index()
daily_profit["_year"] = daily_profit["date"].dt.year

_fade_graded_rows = fade_df[fade_df["result"].isin(["WON", "LOST", "VOID"])]
fade_daily = _fade_graded_rows.groupby(["model_name", "date"])["profit"].sum().reset_index()

_all_models = sorted(set(daily_profit["model_name"]) | set(fade_daily["model_name"]))

for model in _all_models:
    orig = (
        daily_profit[daily_profit["model_name"] == model]
        .set_index("date")[["profit"]]
        .rename(columns={"profit": "Original"})
    )
    fade = (
        fade_daily[fade_daily["model_name"] == model]
        .set_index("date")[["profit"]]
        .rename(columns={"profit": "FADE"})
    )
    combined = orig.join(fade, how="outer").sort_index()
    if combined.empty:
        continue
    combined["_year"] = combined.index.year
    m_2026 = combined[combined["_year"] == 2026].drop(columns=["_year"])
    m_2025 = combined[combined["_year"] == 2025].drop(columns=["_year"])

    st.write(f"**{model}**")
    cols = st.columns(2)
    with cols[0]:
        st.caption("2026 Season")
        if m_2026.empty:
            st.write("No data.")
        else:
            st.line_chart(m_2026, use_container_width=True)
    with cols[1]:
        st.caption("2025 Season")
        if m_2025.empty:
            st.write("No data.")
        else:
            st.line_chart(m_2025, use_container_width=True)





# All graded model-bets per player. The old drop_duplicates on
# (player, date) kept one arbitrary model's row per game — since models bet
# opposite directions on the same player, that sampled a random pick per
# game instead of measuring anything.
result_df = df[df["result"].isin(["WON", "LOST"])]

grouped = result_df.groupby("player_name").apply(
    lambda g: pd.Series({
        "total_bets": len(g),
        "correct_bets": (g["result"] == "WON").sum(),

        "over_bets": (g["bet"] == "OVER").sum(),
        "under_bets": (g["bet"] == "UNDER").sum(),

        "over_win_rate": (
            (g[g["bet"] == "OVER"]["result"] == "WON").mean()
            if (g["bet"] == "OVER").any() else None
        ),
        "under_win_rate": (
            (g[g["bet"] == "UNDER"]["result"] == "WON").mean()
            if (g["bet"] == "UNDER").any() else None
        ),
    })
).reset_index()

for _c in ["total_bets", "correct_bets", "over_bets", "under_bets"]:
    grouped[_c] = grouped[_c].astype(int)

grouped["accuracy"] = (grouped["correct_bets"] / grouped["total_bets"]).apply(
    lambda x: f"{x*100:.2f}%" if pd.notnull(x) else "-"
)

grouped["over_win_rate"] = grouped["over_win_rate"].apply(
    lambda x: f"{x*100:.2f}%" if pd.notnull(x) else "-"
)
grouped["under_win_rate"] = grouped["under_win_rate"].apply(
    lambda x: f"{x*100:.2f}%" if pd.notnull(x) else "-"
)

grouped["over_bets"] = grouped["over_bets"].astype(str) + " / " + grouped["total_bets"].astype(str)
grouped["under_bets"] = grouped["under_bets"].astype(str) + " / " + grouped["total_bets"].astype(str)

st.subheader("Most Correct Bet Players (All Model Bets)")
st.dataframe(grouped[
    ["player_name", "correct_bets", "total_bets", "accuracy",
     "over_bets", "under_bets", "over_win_rate", "under_win_rate"]
])






st.subheader("Over/Under Bets Summary per Model")

_ou_df_all = df[(df["bet"].isin(["OVER", "UNDER"])) & (df["result"].isin(["WON", "LOST"]))].copy()
_ou_year = _ou_df_all["date"].dt.year
_ou_models = sorted(_ou_df_all["model_name"].unique()) if not _ou_df_all.empty else _models_idx


def _ou_breakdown(season_df, models_idx):
    cols = ["Unders", "Overs", "U %", "O %", "O Win %", "U Win %"]
    EM = "—"
    if season_df.empty:
        return pd.DataFrame(
            {c: [EM] * len(models_idx) for c in cols},
            index=pd.Index(models_idx, name="model_name"),
        )
    counts = season_df.groupby(["model_name", "bet"]).size().unstack(fill_value=0)
    counts = counts.rename(columns={"OVER": "Overs", "UNDER": "Unders"})
    for c in ["Overs", "Unders"]:
        if c not in counts.columns:
            counts[c] = 0
    counts = counts[["Overs", "Unders"]].reindex(models_idx, fill_value=0)
    total = counts.sum(axis=1)

    win_counts = (
        season_df[season_df["result"] == "WON"]
        .groupby(["model_name", "bet"]).size()
        .unstack(fill_value=0)
        .rename(columns={"OVER": "Over_Wins", "UNDER": "Under_Wins"})
        .reindex(models_idx, fill_value=0)
    )
    for c in ["Over_Wins", "Under_Wins"]:
        if c not in win_counts.columns:
            win_counts[c] = 0

    over_pct = (win_counts["Over_Wins"] / counts["Overs"].replace(0, np.nan) * 100).round(1).fillna(0)
    under_pct = (win_counts["Under_Wins"] / counts["Unders"].replace(0, np.nan) * 100).round(1).fillna(0)
    total_safe = total.replace(0, np.nan)

    out = pd.DataFrame({
        "Unders": counts["Unders"].astype(int).astype(str),
        "Overs": counts["Overs"].astype(int).astype(str),
        "U %": (counts["Unders"] / total_safe * 100).round(1).fillna(0).astype(str) + "%",
        "O %": (counts["Overs"] / total_safe * 100).round(1).fillna(0).astype(str) + "%",
        "O Win %": over_pct.astype(str) + "%",
        "U Win %": under_pct.astype(str) + "%",
    })

    no_data = total == 0
    if no_data.any():
        out.loc[no_data, :] = EM

    return out


_ou_combined = pd.concat(
    {
        "2026": _ou_breakdown(_ou_df_all[_ou_year == 2026], _ou_models),
        "2025": _ou_breakdown(_ou_df_all[_ou_year == 2025], _ou_models),
        "Total": _ou_breakdown(_ou_df_all, _ou_models),
    },
    axis=1,
)
st.table(_ou_combined)






st.subheader("Average Model Accuracy (PTS Differential)")
_acc_year = df["date"].dt.year

# These models don't store a genuine point estimate in predicted_pts
# (CLCF1/CLCF2/CLCF3 clamp it to the bet's side of the line, CLC4 stores the
# voting side's median), so predicted-vs-actual comparisons are meaningless
# for them.
_FABRICATED_PRED_MODELS = {"model_CLCF1", "model_CLCF2", "model_CLCF3",
                           "model_CLC4_SELECTIVE"}
_acc_models_idx = [m for m in _models_idx if m not in _FABRICATED_PRED_MODELS]


def _acc_series(season_df, models_idx):
    if season_df.empty:
        return pd.Series(0.0, index=pd.Index(models_idx, name="model_name"))
    s = season_df.groupby("model_name")["pts_differential"].mean()
    return s.reindex(models_idx, fill_value=0.0).round(2)


_accuracy_combined = pd.DataFrame({
    "2026": _acc_series(df[_acc_year == 2026], _acc_models_idx),
    "2025": _acc_series(df[_acc_year == 2025], _acc_models_idx),
    "Total": _acc_series(df, _acc_models_idx),
})
st.caption(
    "CLCF1, CLCF2 and CLC4_SELECTIVE are excluded: they store a bet signal, "
    "not a point estimate, in predicted_pts."
)
st.bar_chart(_accuracy_combined)





st.subheader("Low Output Stats")

low_output_games = df[df["note"].str.contains("Low Output", case=False, na=False)].copy()
low_output_games["_year"] = pd.to_datetime(low_output_games["date"]).dt.year
low_output_games["date"] = pd.to_datetime(low_output_games["date"]).dt.date

def highlight_pg_result(val):
    if val == "WON":
        return "color: green; font-weight: bold;"
    elif val == "LOST":
        return "color: red; font-weight: bold;"
    return ""

def smart_num_format(x, col=None):
    if pd.isna(x):
        return ""
    try:
        val = float(x)
        if col in ["over_line", "under_line"]:
            if val.is_integer():
                return str(int(val))
            else:
                return str(val)
        else:
            return str(int(round(val)))
    except Exception:
        return str(x)

low_output_display = low_output_games.drop(columns=["_year"]).copy()
numeric_cols = low_output_display.select_dtypes(include=[np.number]).columns.tolist()
fmt_dict = {col: (lambda x, c=col: smart_num_format(x, c)) for col in numeric_cols}

styled_low_output = (
    low_output_display.style
    .map(highlight_pg_result, subset=["result"])
    .format(fmt_dict)
)

st.write("### Low Output Entries")
_lo_filter = st.radio(
    "Filter entries by season:",
    ("All", "2026", "2025"),
    horizontal=True,
    key="low_output_entries_filter",
)
_lo_view = low_output_display
if _lo_filter == "2026":
    _lo_view = low_output_display[low_output_games["_year"] == 2026]
elif _lo_filter == "2025":
    _lo_view = low_output_display[low_output_games["_year"] == 2025]

_styled_lo_view = (
    _lo_view.style
    .map(highlight_pg_result, subset=["result"])
    .format(fmt_dict)
)
st.dataframe(_styled_lo_view, use_container_width=True, height=400)


def _lo_stats_row(season_df):
    if season_df.empty:
        return {
            "WON": "0 (0.0%)", "LOST": "0 (0.0%)",
            "Total Bet Amount": 0.0, "Winnings": 0.0, "Losses": 0.0,
            "Net Profit": None, "MAE": None, "RMSE": None, "STD": None,
        }
    graded = season_df[season_df["result"].isin(["WON", "LOST"])]
    w = (graded["result"] == "WON").sum()
    l = (graded["result"] == "LOST").sum()
    n = w + l
    wp = (w / n * 100) if n > 0 else 0.0
    lp = (l / n * 100) if n > 0 else 0.0
    return {
        "WON": f"{w} ({wp:.1f}%)",
        "LOST": f"{l} ({lp:.1f}%)",
        "Total Bet Amount": season_df["amount"].sum(),
        "Winnings": season_df.loc[season_df["profit"] > 0, "profit"].sum(),
        "Losses": season_df.loc[season_df["profit"] < 0, "profit"].sum(),
        "Net Profit": season_df["profit"].sum(),
        "MAE": np.mean(np.abs(season_df["pts_differential"])),
        "RMSE": np.sqrt(np.mean(season_df["pts_differential"] ** 2)),
        "STD": np.std(season_df["pts_differential"]),
    }


stats = pd.DataFrame(
    [
        _lo_stats_row(low_output_games[low_output_games["_year"] == 2026]),
        _lo_stats_row(low_output_games[low_output_games["_year"] == 2025]),
        _lo_stats_row(low_output_games),
    ],
    index=["2026", "2025", "Total"],
)

def format_currency_2(x):
    if pd.isna(x):
        return "—"
    return f"${x:,.2f}"

def format_float_max3(x):
    if pd.isna(x):
        return "—"
    return f"{x:.3f}"

styled_stats = (
    stats.style
    .map(lambda v: "color: green; font-weight: bold;" if (isinstance(v, (int, float, np.floating)) and v > 0) else ("color: red; font-weight: bold;" if (isinstance(v, (int, float, np.floating)) and v < 0) else ""), subset=["Winnings", "Losses", "Net Profit"])
    .format({
        "Total Bet Amount": format_currency_2,
        "Winnings": format_currency_2,
        "Losses": format_currency_2,
        "Net Profit": format_currency_2,
        "MAE": format_float_max3,
        "RMSE": format_float_max3,
        "STD": format_float_max3
    })
)

st.write("### Low Output Stats")
st.dataframe(styled_stats, use_container_width=True)












metrics_df = df.groupby("model_name").apply(lambda x: pd.Series({
    "MAE": np.mean(np.abs(x["pts_differential"])),
    "RMSE": np.sqrt(np.mean((x["pts_differential"])**2)),
    "STD": np.std(x["pts_differential"])
})).reset_index()

metrics_df = metrics_df.round(3)


#st.subheader("Model Error Metrics (Points Differential)")
#st.dataframe(metrics_df)





#st.subheader("Sportsbook Error Metrics (Points Differential)")

def american_odds_to_probability(odds):
    if odds > 0:
        return 100 / (odds + 100)
    else:
        return abs(odds) / (abs(odds) + 100)

odds_columns = [col for col in df.columns if 'odds' in col.lower()]

if len(odds_columns) >= 2:
    over_odds_col = [col for col in odds_columns if 'over' in col.lower()][0]
    under_odds_col = [col for col in odds_columns if 'under' in col.lower()][0]
    
    df['over_prob'] = df[over_odds_col].apply(american_odds_to_probability)
    df['under_prob'] = df[under_odds_col].apply(american_odds_to_probability)
    
    df['total_prob'] = df['over_prob'] + df['under_prob']
    df['over_prob_no_vig'] = df['over_prob'] / df['total_prob']
    df['under_prob_no_vig'] = df['under_prob'] / df['total_prob']
    
    df['sportsbook_prediction'] = (df['over_line'] * df['under_prob_no_vig'] + 
                                   df['under_line'] * df['over_prob_no_vig'])
    
    
elif 'over_line' in df.columns and 'under_line' in df.columns:
    df['sportsbook_prediction'] = (df['over_line'] + df['under_line']) / 2
    

df['sportsbook_differential'] = df['sportsbook_prediction'] - df['actual_pts']

sportsbook_metrics = pd.Series({
    "MAE": np.mean(np.abs(df["sportsbook_differential"])),
    "RMSE": np.sqrt(np.mean((df["sportsbook_differential"])**2)),
    "STD": np.std(df["sportsbook_differential"])
})

sportsbook_metrics = sportsbook_metrics.round(3)
sportsbook_metrics_df = pd.DataFrame([sportsbook_metrics])
sportsbook_metrics_df.index = ['Sportsbook']

#st.dataframe(sportsbook_metrics_df)


#if len(odds_columns) >= 2:
#    st.write(f"Average sportsbook edge removed: {((df['total_prob'] - 1) * 100).mean():.1f}%")


st.subheader("Model vs Sportsbook Error Comparison")
st.caption(
    "CLCF1, CLCF2 and CLC4_SELECTIVE are excluded: they store a bet signal, "
    "not a point estimate, in predicted_pts."
)


def _err_metrics(values):
    if values.empty or values.dropna().empty:
        return {"MAE": np.nan, "RMSE": np.nan, "STD": np.nan}
    v = values.dropna()
    return {
        "MAE": float(np.mean(np.abs(v))),
        "RMSE": float(np.sqrt(np.mean(v ** 2))),
        "STD": float(np.std(v)),
    }


def _err_breakdown(season_df, models_idx):
    rows = {}
    for m in models_idx:
        rows[m] = _err_metrics(season_df.loc[season_df["model_name"] == m, "pts_differential"])
    rows["Sportsbook"] = _err_metrics(season_df["sportsbook_differential"])
    return pd.DataFrame.from_dict(rows, orient="index")[["MAE", "RMSE", "STD"]].round(3)


_err_year = df["date"].dt.year
_err_combined = pd.concat(
    {
        "2026": _err_breakdown(df[_err_year == 2026], _acc_models_idx),
        "2025": _err_breakdown(df[_err_year == 2025], _acc_models_idx),
        "Total": _err_breakdown(df, _acc_models_idx),
    },
    axis=1,
)
_err_combined = _err_combined.sort_values(("Total", "MAE"))
def _fmt_err(v):
    if v is None or pd.isna(v):
        return "—"
    try:
        return f"{float(v):.3f}"
    except (TypeError, ValueError):
        return "—"


_styled_err = _err_combined.style.format(_fmt_err, na_rep="—")
st.dataframe(_styled_err, use_container_width=True)
