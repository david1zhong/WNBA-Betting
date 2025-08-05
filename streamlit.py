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
    return ""


num_cols = df.select_dtypes(include="number").columns
df[num_cols] = df[num_cols].round(0).astype("Int64")


st.set_page_config(layout="wide")
st.title("WNBA Betting")


if "id" in df.columns:
    df = df.drop(columns=["id"])


st.subheader("All Predictions")
styled_df = (
    df.style
    .applymap(highlight_result, subset=["result"])
    .hide(axis="index")
)

def format_dynamic_numbers(df):
    df_copy = df.copy()
    for col in df_copy.select_dtypes(include=['float']):
        df_copy[col] = df_copy[col].apply(lambda x: int(x) if x.is_integer() else x)
    return df_copy

formatted_df = format_dynamic_numbers(df)

st.dataframe(
    formatted_df,
    use_container_width=True,
    height=600
)

st.subheader("Average Model Accuracy (PTS Differential)")
accuracy = df.groupby("model_name")["pts_differential"].mean().reset_index()
accuracy["pts_differential"] = accuracy["pts_differential"].round(2)
st.bar_chart(accuracy.set_index("model_name"))
