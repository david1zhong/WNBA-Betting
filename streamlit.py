import streamlit as st
import pandas as pd
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
    .applymap(highlight_result, subset=["result"])
    .format({col: smart_format for col in num_cols})
    .hide(axis="index")
)

st.dataframe(styled_df, use_container_width=True, height=600)

def payout_from_odds(odds):
    return odds / 100 if odds > 0 else 100 / abs(odds)

def calc_profit(row):
    odds_used = row['under_line'] if row['bet'] == "UNDER" else row['over_line']
    payout = payout_from_odds(odds_used)
    if row['result'] == row['bet']:
        return payout
    elif row['result'] in ["WON", "LOST"]:
        return -payout
    return 0

df['profit'] = df.apply(calc_profit, axis=1)

df['winnings'] = df['profit'].apply(lambda x: x if x > 0 else 0)
df['losses'] = df['profit'].apply(lambda x: -x if x < 0 else 0)

total_winnings = df['winnings'].sum()
total_losses = df['losses'].sum()
total_net_profit = total_winnings - total_losses

df['year_month'] = pd.to_datetime(df['date']).dt.to_period("M").astype(str)
monthly_stats = df.groupby('year_month').agg({
    'winnings': 'sum',
    'losses': 'sum',
    'profit': 'sum'
}).reset_index()

st.subheader("Net Profit Overview")
col1, col2, col3 = st.columns(3)
col1.metric("Total Winnings ($)", f"{total_winnings:.2f}")
col2.metric("Total Losses ($)", f"{total_losses:.2f}")
col3.metric("Net Profit ($)", f"{total_net_profit:.2f}")

st.subheader("Monthly Winnings, Losses, and Net Profit ($)")
st.dataframe(monthly_stats.rename(columns={
    'winnings': 'Winnings ($)',
    'losses': 'Losses ($)',
    'profit': 'Net Profit ($)',
    'year_month': 'Month'
}))

st.bar_chart(monthly_stats.set_index('year_month')[['winnings', 'losses', 'profit']])

st.subheader("Wins and Losses per Model")
filtered = df[df["result"].isin(["WON", "LOST"])]
counts = filtered.groupby(["model_name", "result"]).size().unstack(fill_value=0)
counts = counts[["WON", "LOST"]]
totals = counts.sum(axis=1)
percent_df = counts.div(totals, axis=0).multiply(100).round(1).astype(str) + "%"
combined = counts.astype(str) + " (" + percent_df + ")"
st.table(combined)
st.bar_chart(counts)

st.subheader("Average Model Accuracy (PTS Differential)")
accuracy = df.groupby("model_name")["pts_differential"].mean().reset_index()
accuracy["pts_differential"] = accuracy["pts_differential"].round(2)
st.bar_chart(accuracy.set_index("model_name"))
