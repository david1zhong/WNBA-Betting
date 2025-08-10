import streamlit as st
import pandas as pd
import pytz
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
    .applymap(highlight_result, subset=["result"])
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
st.bar_chart(counts_yesterday)



st.subheader("Wins and Losses per Model Total")
filtered = df[df["result"].isin(["WON", "LOST"])]
counts = filtered.groupby(["model_name", "result"]).size().unstack(fill_value=0)
counts = counts[["WON", "LOST"]]
totals = counts.sum(axis=1)
percent_df = counts.div(totals, axis=0).multiply(100).round(1).astype(str) + "%"
combined = counts.astype(str) + " (" + percent_df + ")"
st.table(combined)
st.bar_chart(counts)



df['profit'] = df['profit'].fillna(0)
df['amount'] = df['amount'].fillna(0)
df['date'] = pd.to_datetime(df['date'])
yesterday = datetime.now().date() - timedelta(days=1)
df_yesterday = df[df['date'].dt.date == yesterday]

def summarize(df_input):
    grouped = df_input.groupby('model_name').agg(
        total_bet_amount=('amount', 'sum'),
        total_profit=('profit', 'sum'),
        winnings_amount=('profit', lambda x: x[x > 0].sum()),
        losses_amount=('profit', lambda x: x[x < 0].sum())
    ).reset_index()

    grouped = group[['model_name', 'total_bet_amount', 'winnings_amount', 'losses_amount', 'total_profit']]
    return grouped.sort_values(by='total_profit', ascending=False)

profit_per_model_yesterday = summarize(df_yesterday)
profit_per_model_total = summarize(df)
st.subheader(f"Profit Per Model - {yesterday}")
st.dataframe(profit_per_model_yesterday)
st.subheader("Profit Per Model - All Time")
st.dataframe(profit_per_model_total)



st.subheader("Average Model Accuracy (PTS Differential)")
accuracy = df.groupby("model_name")["pts_differential"].mean().reset_index()
accuracy["pts_differential"] = accuracy["pts_differential"].round(2)
st.bar_chart(accuracy.set_index("model_name"))
