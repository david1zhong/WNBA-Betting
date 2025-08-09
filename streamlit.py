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

st.table(counts_yesterday)
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


st.subheader("Average Model Accuracy (PTS Differential)")
accuracy = df.groupby("model_name")["pts_differential"].mean().reset_index()
accuracy["pts_differential"] = accuracy["pts_differential"].round(2)
st.bar_chart(accuracy.set_index("model_name"))
