"""
Data loading and preparation module.

Handles:
- CSV loading with column standardization
- Derived time columns (year, month, quarter, shipping_days)
- Summary statistics for the system prompt
- Knowledge base text loading for RAG
"""

import pandas as pd
import streamlit as st
from src.config import DATA_PATH, KNOWLEDGE_BASE_PATH


@st.cache_data
def load_data() -> pd.DataFrame:
    """
    Load the Superstore dataset and add derived columns.
    Cached by Streamlit so it only runs once per session.
    """
    df = pd.read_csv(DATA_PATH, encoding="latin1")

    # Standardize column names: "Order Date" -> "order_date"
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    # Parse dates
    df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
    df["ship_date"] = pd.to_datetime(df["ship_date"], errors="coerce")

    # Derived time features
    df["year"] = df["order_date"].dt.year
    df["month"] = df["order_date"].dt.month
    df["quarter"] = df["order_date"].dt.quarter
    df["shipping_days"] = (df["ship_date"] - df["order_date"]).dt.days

    return df


@st.cache_data
def load_knowledge_base() -> str:
    """Load the RAG knowledge base text file."""
    return KNOWLEDGE_BASE_PATH.read_text(encoding="utf-8")


@st.cache_data
def compute_data_summary(_df: pd.DataFrame) -> dict:
    """
    Pre-compute summary statistics injected into the system prompt.

    The underscore prefix on _df tells Streamlit not to hash
    this argument (DataFrames are expensive to hash).
    """
    return {
        "total_orders": len(_df),
        "total_sales": _df["sales"].sum(),
        "avg_sales": _df["sales"].mean(),
        "median_sales": _df["sales"].median(),
        "total_profit": _df["profit"].sum(),
        "avg_profit": _df["profit"].mean(),
        "categories": _df["category"].value_counts().to_dict(),
        "regions": _df["region"].value_counts().to_dict(),
        "segments": _df["segment"].value_counts().to_dict(),
        "years": sorted(_df["year"].dropna().unique().tolist()),
        "top_subcats": (
            _df.groupby("sub-category")["sales"]
            .median()
            .sort_values(ascending=False)
            .head(5)
            .to_dict()
        ),
        "bottom_subcats": (
            _df.groupby("sub-category")["sales"]
            .median()
            .sort_values()
            .head(5)
            .to_dict()
        ),
    }
