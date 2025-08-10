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




st.subheader("Over/Under Bets Summary per Model")
ou_df = df[(df["bet"].isin(["OVER", "UNDER"])) & (df["result"].isin(["WON", "LOST"]))]

ou_counts = (
    ou_df.groupby(["model_name", "bet", "result"])
    .size()
    .unstack(fill_value=0)
)

for col in ["WON", "LOST"]:
    if col not in ou_counts.columns:
        ou_counts[col] = 0

ou_counts["TOTAL"] = ou_counts["WON"] + ou_counts["LOST"]
ou_counts["WON %"] = (ou_counts["WON"] / ou_counts["TOTAL"] * 100).round(1)
over_summary = ou_counts.loc[pd.IndexSlice[:, "OVER"], :].reset_index(level=1, drop=True)
under_summary = ou_counts.loc[pd.IndexSlice[:, "UNDER"], :].reset_index(level=1, drop=True)

summary_df = pd.DataFrame({
    "Number of Overs": over_summary["TOTAL"],
    "Number of Unders": under_summary["TOTAL"],
    "Over Won": over_summary["WON"],
    "Under Won": under_summary["WON"],
    "Over Lost": over_summary["LOST"],
    "Under Lost": under_summary["LOST"],
    "Over Won %": over_summary["WON %"].map(lambda x: f"{x:.1f}%" if pd.notna(x) else "0%"),
    "Under Won %": under_summary["WON %"].map(lambda x: f"{x:.1f}%" if pd.notna(x) else "0%"),
})

summary_df = summary_df.fillna(0).astype({
    "Number of Overs": int,
    "Number of Unders": int,
    "Over Won": int,
    "Under Won": int,
    "Over Lost": int,
    "Under Lost": int,
})

st.table(summary_df)





st.subheader("Average Model Accuracy (PTS Differential)")
accuracy = df.groupby("model_name")["pts_differential"].mean().reset_index()
accuracy["pts_differential"] = accuracy["pts_differential"].round(2)
st.bar_chart(accuracy.set_index("model_name"))
