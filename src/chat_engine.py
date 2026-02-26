"""
Chat engine module.

Handles:
- OpenAI client creation
- System prompt construction with RAG context
- Natural language -> pandas query translation and execution
- Natural language -> matplotlib/seaborn chart generation
- Full chat response pipeline
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import streamlit as st
from openai import OpenAI
from src.config import (
    OPENAI_API_KEY,
    OPENAI_MODEL,
    CHAT_TEMPERATURE,
    MAX_TOKENS,
)

# Dark style matching Spotify theme
matplotlib.rcParams.update({
    "figure.facecolor": "#121212",
    "axes.facecolor": "#1E1E1E",
    "axes.edgecolor": "#333333",
    "axes.labelcolor": "#FFFFFF",
    "text.color": "#FFFFFF",
    "xtick.color": "#B3B3B3",
    "ytick.color": "#B3B3B3",
    "grid.color": "#333333",
    "legend.facecolor": "#1E1E1E",
    "legend.edgecolor": "#333333",
    "legend.labelcolor": "#FFFFFF",
})

CHART_COLORS = ["#1DB954", "#1ED760", "#A0E77D", "#509BF5", "#F573A0",
                "#E8125C", "#FFA42B", "#F5E6C8", "#B49BC8", "#CDF564"]


def get_openai_client() -> OpenAI:
    """Create an OpenAI client using the API key from .env."""
    return OpenAI(api_key=OPENAI_API_KEY)


def build_system_prompt(knowledge_base: str, data_summary: dict) -> str:
    """Construct the system prompt with RAG context."""
    return f"""You are an AI Sales Analyst assistant for the Superstore Sales Forecasting project.

RULES:
1. Ground every answer in the DATA SUMMARY and KNOWLEDGE BASE below. Never invent numbers.
2. When the user asks something that requires querying the raw data, respond ONLY with: [QUERY_NEEDED] followed by a rephrased version of what they want.
3. When the user asks for a chart, graph, plot, visualization, or says words like "show me", "graph", "plot", "chart", "grafica", "muestra", "visualiza", respond ONLY with: [CHART_NEEDED] followed by a description of what chart to create.
4. Be concise. Use numbers and percentages.
5. Answer in the same language the user writes (Spanish or English).
6. When discussing models, cite the exact metrics from the knowledge base.

DATA SUMMARY:
- Total orders: {data_summary['total_orders']:,}
- Total sales: ${data_summary['total_sales']:,.2f}
- Average sale: ${data_summary['avg_sales']:.2f} | Median sale: ${data_summary['median_sales']:.2f}
- Total profit: ${data_summary['total_profit']:,.2f} | Average profit: ${data_summary['avg_profit']:.2f}
- Categories: {data_summary['categories']}
- Regions: {data_summary['regions']}
- Segments: {data_summary['segments']}
- Years: {data_summary['years']}
- Top 5 sub-categories by median sales: {data_summary['top_subcats']}
- Bottom 5 sub-categories by median sales: {data_summary['bottom_subcats']}

PROJECT KNOWLEDGE BASE:
{knowledge_base}
"""


def run_data_query(df: pd.DataFrame, question: str, client: OpenAI) -> str:
    """Translate a natural language question into pandas code and execute it."""
    columns_info = ", ".join([f"{c} ({df[c].dtype})" for c in df.columns])

    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a pandas code generator. Given a question about a "
                    "DataFrame called `df`, generate ONLY a single Python expression. "
                    "Return ONLY the code. No explanation, no markdown, no backticks. "
                    "Use .to_string() for DataFrames.\n\n"
                    f"Columns: {columns_info}\n\n"
                    "Column names are lowercase with underscores. "
                    "Category: 'Technology', 'Furniture', 'Office Supplies'. "
                    "Segment: 'Consumer', 'Corporate', 'Home Office'. "
                    "Region: 'West', 'East', 'Central', 'South'. "
                    "Derived columns: year, month, quarter, shipping_days."
                ),
            },
            {"role": "user", "content": question},
        ],
    )

    code = response.choices[0].message.content.strip()
    code = code.replace("```python", "").replace("```", "").strip()

    try:
        result = eval(code, {"df": df, "pd": pd, "np": np})
        return (
            f"**Query:**\n```python\n{code}\n```\n"
            f"**Result:**\n```\n{str(result)}\n```"
        )
    except Exception as e:
        return f"Query failed. Error: {e}\nCode: `{code}`"


def generate_chart(df: pd.DataFrame, description: str, client: OpenAI):
    """Generate a matplotlib/seaborn chart from a natural language description."""
    columns_info = ", ".join([f"{c} ({df[c].dtype})" for c in df.columns])

    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a matplotlib/seaborn code generator. "
                    "Generate Python code that creates a chart from a DataFrame called `df`.\n"
                    "RULES:\n"
                    "- Start with: fig, ax = plt.subplots(figsize=(10, 6))\n"
                    "- Use these colors: " + str(CHART_COLORS[:5]) + "\n"
                    "- End the code with just: fig\n"
                    "- Do NOT call plt.show() or plt.savefig()\n"
                    "- Do NOT use plt.style.use()\n"
                    "- Always add plt.tight_layout() before the final fig line\n"
                    "- Return ONLY code. No explanation, no backticks.\n\n"
                    f"Available columns: {columns_info}\n"
                    "Category: 'Technology', 'Furniture', 'Office Supplies'. "
                    "Region: 'West', 'East', 'Central', 'South'. "
                    "Segment: 'Consumer', 'Corporate', 'Home Office'."
                ),
            },
            {"role": "user", "content": description},
        ],
    )

    code = response.choices[0].message.content.strip()
    code = code.replace("```python", "").replace("```", "").strip()

    try:
        local_vars = {"df": df, "pd": pd, "np": np, "plt": plt, "sns": sns}
        exec(code, local_vars)
        fig = local_vars.get("fig", plt.gcf())
        return fig, code
    except Exception as e:
        return None, f"Chart generation failed. Error: {e}\nCode:\n{code}"


def get_chat_response(
    client: OpenAI,
    messages: list,
    system_prompt: str,
    df: pd.DataFrame,
    user_prompt: str,
) -> str:
    """
    Full response pipeline:
    1. Send conversation to GPT with RAG context.
    2. If [CHART_NEEDED] -> generate and display matplotlib chart.
    3. If [QUERY_NEEDED] -> execute pandas query.
    4. Feed results back to GPT for natural language answer.
    """
    api_messages = [{"role": "system", "content": system_prompt}]
    for msg in messages[-10:]:
        api_messages.append({"role": msg["role"], "content": msg["content"]})

    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=api_messages,
        temperature=CHAT_TEMPERATURE,
        max_tokens=MAX_TOKENS,
    )
    answer = response.choices[0].message.content

    # --- CHART ---
    if "[CHART_NEEDED]" in answer:
        chart_desc = answer.split("[CHART_NEEDED]")[-1].strip()
        fig, code = generate_chart(df, chart_desc, client)

        if fig is not None:
            st.pyplot(fig)
            plt.close(fig)

            followup = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                    {
                        "role": "assistant",
                        "content": f"I generated a chart with:\n```python\n{code}\n```",
                    },
                    {
                        "role": "user",
                        "content": (
                            "The chart is displayed. Give a brief interpretation "
                            "of what it shows. Same language as the user."
                        ),
                    },
                ],
                temperature=CHAT_TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )
            return followup.choices[0].message.content
        else:
            return f"No pude generar la grafica. {code}"

    # --- DATA QUERY ---
    if "[QUERY_NEEDED]" in answer:
        query_text = answer.split("[QUERY_NEEDED]")[-1].strip()
        query_result = run_data_query(df, query_text, client)

        followup = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
                {
                    "role": "assistant",
                    "content": f"Query result:\n{query_result}",
                },
                {
                    "role": "user",
                    "content": (
                        "Give a clear answer using those results. "
                        "Include key numbers. Show the pandas code. "
                        "Same language as the user."
                    ),
                },
            ],
            temperature=CHAT_TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        return followup.choices[0].message.content

    return answer
