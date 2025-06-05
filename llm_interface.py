#!/usr/bin/env python3
"""
Restaurant Sales Streamlit Assistant (Key in Code)
===================================================

Streamlit web app for querying restaurant sales data (March 1–9, 2025) using LangChain and OpenAI.

Setup:
1. pip install streamlit langchain openai pandas
2. Place all output_2025-03-*.json files in same folder.
3. Run with: streamlit run restaurant_streamlit_app.py
"""

import os
import glob
import json
from typing import List

import streamlit as st
import pandas as pd
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.llms import OpenAI
from langchain_community.chat_models import ChatOpenAI

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

DATA_PATH_PATTERN = "output_2025-03-*.json"
TEMPERATURE = 0.14

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def load_json_records(pattern: str) -> List[dict]:
    records: List[dict] = []
    for path in sorted(glob.glob(pattern)):
        with open(path, "r", encoding="utf-8") as f:
            doc = json.load(f)
            if isinstance(doc, dict) and "data" in doc:
                records.extend(doc["data"])
            elif isinstance(doc, list):
                records.extend(doc)
    return records


def build_dataframe(records: List[dict]) -> pd.DataFrame:
    df = pd.json_normalize(records, sep=".")
    for col in ["billAmount", "grossAmount", "netAmount", "totalDiscountAmount", "taxAmount"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "invoiceDay" in df.columns:
        df["invoiceDay"] = pd.to_datetime(df["invoiceDay"]).dt.date
    return df


def create_agent(df: pd.DataFrame):
    llm = ChatOpenAI(model_name="gpt-4.1",temperature=TEMPERATURE)
    return create_pandas_dataframe_agent(llm, df, verbose=False, allow_dangerous_code=True)

# ------------------------------------------------------------------
# Streamlit App
# ------------------------------------------------------------------

def main():
    st.set_page_config(page_title="Restaurant Sales Assistant", layout="wide")
    st.title("📊 Papaya BKC Restaurant Sales Assistant (1–9 March 2025)")

    with st.spinner("Loading sales data..."):
        records = load_json_records(DATA_PATH_PATTERN)
        if not records:
            st.error("No JSON records found. Check file pattern.")
            return
        df = build_dataframe(records)
        agent = create_agent(df)

    st.success("Data loaded. You can now ask questions!")
    query = st.text_input("Ask a question about your sales data:", placeholder="e.g., What was the gross revenue on March 5th?")

    if query:
        with st.spinner("Thinking..."):
            try:
                response = agent.run(query)
                st.markdown(f"### 🤖 Response\n{response}")
            except Exception as e:
                st.error(f"❌ Error: {e}")

    with st.expander("📋 Preview sales data"):
        st.dataframe(df.head(100))

if __name__ == "__main__":
    main()
