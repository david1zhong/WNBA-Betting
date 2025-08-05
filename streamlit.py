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

st.subheader("Average Model Accuracy (PTS Differential)")
accuracy = df.groupby("model_name")["pts_differential"].mean().reset_index()
accuracy["pts_differential"] = accuracy["pts_differential"].round(2)
st.bar_chart(accuracy.set_index("model_name"))
