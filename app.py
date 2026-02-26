"""
Superstore Sales Analyst - Streamlit Application
=================================================
AI-powered chatbot for exploring the Superstore Sales Forecasting project.
Uses OpenAI GPT with RAG (Retrieval-Augmented Generation) to answer
questions grounded in actual data and analysis results.

Run with: streamlit run app.py
"""

import streamlit as st
from src.config import APP_TITLE, APP_ICON, OPENAI_API_KEY
from src.data_loader import load_data, load_knowledge_base, compute_data_summary
from src.chat_engine import get_openai_client, build_system_prompt, get_chat_response


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(page_title=APP_TITLE, page_icon=APP_ICON, layout="wide")


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("Configuration")

    # Allow override of .env key via sidebar input
    api_key_input = st.text_input(
        "OpenAI API Key",
        type="password",
        value=OPENAI_API_KEY or "",
        help="Loaded from .env file. You can override it here.",
    )

    st.divider()
    st.markdown("### Example Questions")
    st.markdown(
        "- What is the best model and why?\n"
        "- Which sub-categories are most profitable?\n"
        "- What happens to profit when discount exceeds 30%?\n"
        "- How many orders are in the West region?\n"
        "- Show me total sales by category and year\n"
        "- What is the unit price proxy?\n"
        "- Compare Phase 1 vs Phase 2 results\n"
        "- Cual es la subcategoria con mas ventas?\n"
    )

    st.divider()
    st.markdown("### About")
    st.markdown(
        "RAG-powered assistant that answers questions using the "
        "actual Superstore dataset and the ML analysis results. "
        "Built with OpenAI GPT + Streamlit."
    )


# ---------------------------------------------------------------------------
# Validate API key
# ---------------------------------------------------------------------------
active_key = api_key_input or OPENAI_API_KEY

if not active_key:
    st.warning(
        "No API key found. Either:\n"
        "1. Create a `.env` file with `OPENAI_API_KEY=sk-...`\n"
        "2. Or paste your key in the sidebar."
    )
    st.stop()


# ---------------------------------------------------------------------------
# Load data and knowledge base
# ---------------------------------------------------------------------------
try:
    df = load_data()
    knowledge_base = load_knowledge_base()
    data_summary = compute_data_summary(df)
except FileNotFoundError as e:
    st.error(f"File not found: {e}")
    st.info(
        "Make sure `data/Sample - Superstore.csv` and "
        "`src/knowledge_base.txt` exist in the project directory."
    )
    st.stop()


# ---------------------------------------------------------------------------
# Header and metrics
# ---------------------------------------------------------------------------
st.title(f"{APP_ICON} {APP_TITLE}")
st.caption("Ask questions about the Superstore dataset, the EDA, the ML models, or request live data queries.")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Orders", f"{len(df):,}")
col2.metric("Total Sales", f"${df['sales'].sum():,.0f}")
col3.metric("Total Profit", f"${df['profit'].sum():,.0f}")
col4.metric("Best Model R²", "0.925")

st.divider()


# ---------------------------------------------------------------------------
# Chat interface
# ---------------------------------------------------------------------------

# Initialize OpenAI client
from openai import OpenAI
client = OpenAI(api_key=active_key)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask me anything about the Superstore analysis..."):

    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get and show assistant response
    with st.chat_message("assistant"):
        with st.spinner("Analyzing..."):
            try:
                system_prompt = build_system_prompt(knowledge_base, data_summary)
                answer = get_chat_response(
                    client=client,
                    messages=st.session_state.messages,
                    system_prompt=system_prompt,
                    df=df,
                    user_prompt=prompt,
                )
                st.markdown(answer)
            except Exception as e:
                answer = f"Error: {str(e)}"
                st.error(answer)

    # Save to history
    st.session_state.messages.append({"role": "assistant", "content": answer})
