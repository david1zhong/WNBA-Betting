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

st.subheader("All Predictions")


def smart_format(x):
    if pd.isna(x):
        return ""
    if float(x).is_integer():
        return str(int(x))
    return str(x)
    
styled_df = (
    df.style
    .map(highlight_result, subset=["result"])
    .format({col: smart_format for col in num_cols})
    .hide(axis="index")
)

st.dataframe(
    styled_df,
    use_container_width=True,
    height=600
)


st.subheader("Wins and Losses per Model Yesterday")

eastern = pytz.timezone("US/Eastern")
yesterday_date = (datetime.now(eastern) - timedelta(days=1)).date()

yesterday_df = df[
    (pd.to_datetime(df["date"]).dt.date == yesterday_date) &
    (df["result"].isin(["WON", "LOST"]))
]

if yesterday_df.empty:
    model_names = df["model_name"].unique()
    counts_yesterday = pd.DataFrame({"model_name": model_names, "WON": 0, "LOST": 0})
    counts_yesterday = counts_yesterday.set_index("model_name")
else:
    counts_yesterday = (
        yesterday_df.groupby(["model_name", "result"])
        .size()
        .unstack(fill_value=0)
    )
    for col in ["WON", "LOST"]:
        if col not in counts_yesterday.columns:
            counts_yesterday[col] = 0
    counts_yesterday = counts_yesterday[["WON", "LOST"]]

totals_yesterday = counts_yesterday.sum(axis=1)
percent_df_yesterday = counts_yesterday.div(totals_yesterday, axis=0).multiply(100)
percent_df_yesterday = percent_df_yesterday.round(1).astype(str) + "%"

combined_yesterday = counts_yesterday.astype(str) + " (" + percent_df_yesterday + ")"

st.table(combined_yesterday)



st.subheader("Wins and Losses per Model Total")
filtered = df[df["result"].isin(["WON", "LOST"])]
counts = filtered.groupby(["model_name", "result"]).size().unstack(fill_value=0)
counts = counts[["WON", "LOST"]]
totals = counts.sum(axis=1)
percent_df = counts.div(totals, axis=0).multiply(100).round(1).astype(str) + "%"
combined = counts.astype(str) + " (" + percent_df + ")"
st.table(combined)



df['profit'] = df['profit'].fillna(0)
df['amount'] = df['amount'].fillna(0)
df['date'] = pd.to_datetime(df['date'])
yesterday = datetime.now().date() - timedelta(days=1)
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
profit_per_model_total = summarize(df)
st.subheader(f"Profit Per Model - {yesterday}")
st.dataframe(profit_per_model_yesterday)
st.subheader("Profit Per Model - All Time")
st.dataframe(profit_per_model_total)




st.subheader("Daily Profit per Model")
df['date'] = pd.to_datetime(df['date'])
daily_profit = df.groupby(["model_name", "date"])["profit"].sum().reset_index()
daily_profit = daily_profit[daily_profit["profit"] != 0]

for model in daily_profit["model_name"].unique():
    model_data = daily_profit[daily_profit["model_name"] == model].sort_values("date")
    
    st.write(f"**{model}**")
    st.line_chart(
        model_data.set_index("date")["profit"],
        use_container_width=True
    )





result_df = df[df["result"].isin(["WON", "LOST"])]
dedup = result_df.drop_duplicates(subset=["player_name", "date"])

grouped = dedup.groupby("player_name").apply(
    lambda g: pd.Series({
        "total_games": len(g),
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

grouped["accuracy"] = (grouped["correct_bets"] / grouped["total_games"]).apply(
    lambda x: f"{x*100:.2f}%" if pd.notnull(x) else "-"
)

grouped["over_win_rate"] = grouped["over_win_rate"].apply(
    lambda x: f"{x*100:.2f}%" if pd.notnull(x) else "-"
)
grouped["under_win_rate"] = grouped["under_win_rate"].apply(
    lambda x: f"{x*100:.2f}%" if pd.notnull(x) else "-"
)

grouped["over_bets"] = grouped["over_bets"].astype(str) + " / " + grouped["total_games"].astype(str)
grouped["under_bets"] = grouped["under_bets"].astype(str) + " / " + grouped["total_games"].astype(str)

st.subheader("Most Correct Bet Players (No Duplicate Games)")
st.dataframe(grouped[
    ["player_name", "correct_bets", "total_games", "accuracy",
     "over_bets", "under_bets", "over_win_rate", "under_win_rate"]
])






st.subheader("Over/Under Bets Summary per Model")
ou_df = df[(df["bet"].isin(["OVER", "UNDER"])) & (df["result"].isin(["WON", "LOST"]))]
counts = ou_df.groupby(["model_name", "bet"]).size().unstack(fill_value=0)
counts = counts.rename(columns={"OVER": "Overs", "UNDER": "Unders"})
counts["Total"] = counts["Overs"] + counts["Unders"]

win_counts = (
    ou_df[ou_df["result"] == "WON"]
    .groupby(["model_name", "bet"])
    .size()
    .unstack(fill_value=0)
    .rename(columns={"OVER": "Over_Wins", "UNDER": "Under_Wins"})
)

over_win_pct = (win_counts["Over_Wins"] / counts["Overs"].replace(0, pd.NA) * 100).round(1)
under_win_pct = (win_counts["Under_Wins"] / counts["Unders"].replace(0, pd.NA) * 100).round(1)

summary_df = pd.DataFrame({
    "Number of Unders": counts["Unders"],
    "Number of Overs": counts["Overs"],
    "Unders % of All Bets": (counts["Unders"] / counts["Total"] * 100).round(1).astype(str) + "%",
    "Overs % of All Bets": (counts["Overs"] / counts["Total"] * 100).round(1).astype(str) + "%",
    "Over Win %": over_win_pct.fillna(0).astype(str) + "%",
    "Under Win %": under_win_pct.fillna(0).astype(str) + "%"
})

st.table(summary_df)






st.subheader("Average Model Accuracy (PTS Differential)")
accuracy = df.groupby("model_name")["pts_differential"].mean().reset_index()
accuracy["pts_differential"] = accuracy["pts_differential"].round(2)
st.bar_chart(accuracy.set_index("model_name"))





st.subheader("Period Game Stats")

period_games = df[df["note"].str.contains("Period Game", case=False, na=False)].copy()
period_games["date"] = pd.to_datetime(period_games["date"]).dt.date

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

period_games_display = period_games.copy()
numeric_cols = period_games_display.select_dtypes(include=[np.number]).columns.tolist()
fmt_dict = {col: (lambda x, c=col: smart_num_format(x, c)) for col in numeric_cols}

styled_period_games = (
    period_games_display.style
    .format(fmt_dict)
)

st.write("### Period Game Entries")
st.dataframe(styled_period_games, use_container_width=True, height=400)

pg_filtered = period_games[period_games["result"].isin(["WON", "LOST"])]
won_count = (pg_filtered["result"] == "WON").sum()
lost_count = (pg_filtered["result"] == "LOST").sum()
total_played = won_count + lost_count
win_pct = (won_count / total_played * 100) if total_played > 0 else 0
loss_pct = (lost_count / total_played * 100) if total_played > 0 else 0

stats = pd.DataFrame({
    "WON": [f"{won_count} ({win_pct:.1f}%)"],
    "LOST": [f"{lost_count} ({loss_pct:.1f}%)"],
    "Total Bet Amount": [period_games["amount"].sum()],
    "Winnings": [period_games.loc[period_games["profit"] > 0, "profit"].sum()],
    "Losses": [period_games.loc[period_games["profit"] < 0, "profit"].sum()],
    "Net Profit": [period_games["profit"].sum() if not period_games.empty else None],
    "MAE": [np.mean(np.abs(period_games["pts_differential"])) if not period_games.empty else None],
    "RMSE": [np.sqrt(np.mean(period_games["pts_differential"]**2)) if not period_games.empty else None],
    "STD": [np.std(period_games["pts_differential"]) if not period_games.empty else None]
})

def format_currency_2(x):
    if pd.isna(x):
        return "—"
    return f"${x:,.2f}"

def format_float_max3(x):
    if pd.isna(x):
        return "—"
    return f"{x:.3f}"

styled_stats = (
    stats.reset_index(drop=True)
    .style
    .applymap(lambda v: "color: green; font-weight: bold;" if (isinstance(v, (int, float, np.floating)) and v > 0) else ("color: red; font-weight: bold;" if (isinstance(v, (int, float, np.floating)) and v < 0) else ""), subset=["Winnings", "Losses", "Net Profit"])
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

st.write("### Period Game Stats")
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
comparison_df = metrics_df.copy()
sportsbook_row = pd.DataFrame({
    'model_name': ['Sportsbook'],
    'MAE': [sportsbook_metrics['MAE']],
    'RMSE': [sportsbook_metrics['RMSE']],
    'STD': [sportsbook_metrics['STD']]
})
comparison_df = pd.concat([comparison_df, sportsbook_row], ignore_index=True)
comparison_df = comparison_df.sort_values('MAE')
st.dataframe(comparison_df)
