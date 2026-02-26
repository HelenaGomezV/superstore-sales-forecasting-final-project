"""
Superstore Sales Analyst - Streamlit Application
=================================================
Multi-page app: Dashboard (notebook results) + AI Chat (GenAI)
Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

from src.config import APP_TITLE, APP_ICON, OPENAI_API_KEY, CHART_COLORS
from src.data_loader import load_data, load_knowledge_base, compute_data_summary
from src.chat_engine import build_system_prompt, get_chat_response

# ---------------------------------------------------------------------------
# Matplotlib dark style for Spotify theme
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(page_title=APP_TITLE, page_icon=APP_ICON, layout="wide")

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
try:
    df = load_data()
    knowledge_base = load_knowledge_base()
    data_summary = compute_data_summary(df)
except FileNotFoundError as e:
    st.error(f"File not found: {e}")
    st.stop()

# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title(f"{APP_ICON} {APP_TITLE}")
    page = st.radio(
        "Navigate",
        ["Dashboard", "AI Chat"],
        index=0,
    )
    st.divider()

    if page == "AI Chat":
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
            "- Show me sales by category\n"
            "- How many orders in the West region?\n"
            "- Compare Phase 1 vs Phase 2\n"
            "- Muestra ventas por subcategoria\n"
        )

# =========================================================================
# PAGE 1: DASHBOARD - Notebook Results
# =========================================================================
if page == "Dashboard":

    st.title("Superstore Sales Forecasting - Dashboard")
    st.caption("Key results from the EDA and ML analysis")

    # --- KPI Metrics ---
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Orders", f"{len(df):,}")
    col2.metric("Total Sales", f"${df['sales'].sum():,.0f}")
    col3.metric("Total Profit", f"${df['profit'].sum():,.0f}")
    col4.metric("Best Model R\u00b2", "0.925")

    st.divider()

    # --- Tab layout for different analysis sections ---
    tab1, tab2, tab3, tab4 = st.tabs([
        "EDA Overview",
        "Sales Analysis",
        "Model Results",
        "Unit Price Impact"
    ])

    # ---- TAB 1: EDA Overview ----
    with tab1:
        col_left, col_right = st.columns(2)

        with col_left:
            st.subheader("Sales Distribution")
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.histplot(df['sales'], bins=40, kde=True, color=CHART_COLORS[0], ax=ax)
            ax.set_title("Distribution of Sales")
            ax.set_xlabel("Sales ($)")
            ax.set_ylabel("Frequency")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            st.markdown(
                "**Skewness:** 12.97 | **Kurtosis:** 305.31  \n"
                "Mean: $230 | Median: $54 | Max: $22,638  \n"
                "Only 1.4% of orders exceed $2,000"
            )

        with col_right:
            st.subheader("Correlation with Sales")
            num_df = df[['sales', 'quantity', 'discount', 'profit', 'shipping_days']].copy()
            corr = num_df.corr()
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.heatmap(corr, annot=True, cmap='RdYlGn', center=0, ax=ax)
            ax.set_title("Correlation Matrix")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            st.markdown(
                "**Profit** has the strongest correlation (r=0.479)  \n"
                "**Quantity** is moderate (r=0.201)  \n"
                "**Discount** is negligible (r=-0.028)"
            )

        # Category and Sub-Category
        col_left2, col_right2 = st.columns(2)

        with col_left2:
            st.subheader("Orders by Category")
            cat_counts = df['category'].value_counts()
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.barh(cat_counts.index, cat_counts.values, color=CHART_COLORS[:3])
            ax.set_xlabel("Number of Orders")
            ax.set_title("Orders by Category")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        with col_right2:
            st.subheader("Median Sales by Sub-Category")
            subcat = df.groupby('sub-category')['sales'].median().sort_values()
            fig, ax = plt.subplots(figsize=(8, 6))
            colors = [CHART_COLORS[0] if v > 100 else CHART_COLORS[4] for v in subcat.values]
            ax.barh(subcat.index, subcat.values, color=colors)
            ax.set_xlabel("Median Sales ($)")
            ax.set_title("Median Sales by Sub-Category")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

    # ---- TAB 2: Sales Analysis ----
    with tab2:
        col_left, col_right = st.columns(2)

        with col_left:
            st.subheader("Monthly Sales Trend")
            monthly = df.groupby(['year', 'month'])['sales'].sum().reset_index()
            monthly['date'] = pd.to_datetime(monthly[['year', 'month']].assign(day=1))
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(monthly['date'], monthly['sales'] / 1000, marker='o',
                    color=CHART_COLORS[0], linewidth=2)
            ax.set_ylabel("Sales ($K)")
            ax.set_title("Total Monthly Sales (2014-2017)")
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        with col_right:
            st.subheader("Annual Sales by Category")
            cat_yr = df.groupby(['year', 'category'])['sales'].sum().unstack()
            fig, ax = plt.subplots(figsize=(10, 5))
            cat_yr.div(1000).plot(kind='bar', ax=ax, color=CHART_COLORS[:3])
            ax.set_ylabel("Sales ($K)")
            ax.set_title("Annual Sales by Category")
            ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        # Discount impact
        col_left2, col_right2 = st.columns(2)

        with col_left2:
            st.subheader("Mean Sales by Discount")
            disc_stats = df.groupby('discount')[['sales', 'profit']].mean().round(2)
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.bar(disc_stats.index.astype(str), disc_stats['sales'], color=CHART_COLORS[0])
            ax.set_xlabel("Discount")
            ax.set_ylabel("Mean Sales ($)")
            ax.set_title("Mean Sales by Discount Level")
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        with col_right2:
            st.subheader("Mean Profit by Discount")
            fig, ax = plt.subplots(figsize=(8, 5))
            colors_profit = [CHART_COLORS[0] if v >= 0 else CHART_COLORS[5] for v in disc_stats['profit']]
            ax.bar(disc_stats.index.astype(str), disc_stats['profit'], color=colors_profit)
            ax.axhline(0, linestyle='--', color='white', alpha=0.5)
            ax.set_xlabel("Discount")
            ax.set_ylabel("Mean Profit ($)")
            ax.set_title("Mean Profit by Discount Level")
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            st.markdown(
                ":red[Discounts >= 30% cause negative profit on average]"
            )

        # Seasonality
        st.subheader("Seasonality: Mean Sales by Month")
        seas = df.groupby('month')['sales'].mean()
        fig, ax = plt.subplots(figsize=(12, 4))
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        colors_month = [CHART_COLORS[0] if m >= 9 else CHART_COLORS[3] for m in seas.index]
        ax.bar(month_names, seas.values, color=colors_month)
        ax.set_ylabel("Mean Sales ($)")
        ax.set_title("Q4 (Sep-Dec) shows the highest average sales")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # ---- TAB 3: Model Results ----
    with tab3:
        st.subheader("Phase 1: Model Comparison (without Unit Price Proxy)")

        results_phase1 = pd.DataFrame({
            'Model': ['RF Base', 'RF Tuned', 'RF Tuned + log(y)',
                      'XGB Base', 'XGB Tuned', 'XGB Early Stop',
                      'XGB Huber', 'XGB + log(y)'],
            'R2': [0.699, 0.723, 0.509, 0.717, 0.588, 0.594, 0.542, 0.671],
            'MAE': [96.34, 87.45, 91.47, 81.35, 88.46, 88.14, 92.80, 80.99],
            'SMAPE': [56.4, 45.4, 36.1, 45.0, 45.6, 44.9, 40.2, 35.1],
        })
        st.dataframe(results_phase1.style.highlight_min(subset=['MAE'], color='#1DB954')
                     .highlight_max(subset=['R2'], color='#1DB954'),
                     use_container_width=True)

        col_left, col_right = st.columns(2)

        with col_left:
            fig, ax = plt.subplots(figsize=(10, 5))
            y_pos = range(len(results_phase1))
            ax.barh(results_phase1['Model'], results_phase1['MAE'], color=CHART_COLORS[0])
            ax.set_xlabel("MAE ($)")
            ax.set_title("Phase 1: MAE by Model")
            ax.invert_yaxis()
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        with col_right:
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.barh(results_phase1['Model'], results_phase1['R2'], color=CHART_COLORS[3])
            ax.set_xlabel("R\u00b2")
            ax.set_title("Phase 1: R\u00b2 by Model")
            ax.set_xlim(0, 1)
            ax.invert_yaxis()
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        st.markdown(
            "**Best MAE:** XGB + log(y) = $80.99  \n"
            "**Best R\u00b2:** RF Tuned = 0.723  \n"
            "**Key insight:** Log target helps XGBoost but hurts Random Forest"
        )

        st.divider()

        # Error by segment
        st.subheader("Error Analysis by Price Segment (Phase 1)")
        segments = pd.DataFrame({
            'Segment': ['Low ($0-50)', 'Medium ($50-200)', 'High ($200-1K)', 'Premium ($1K+)'],
            'Orders': [1617, 857, 691, 147],
            'MAE ($)': [12.20, 42.75, 144.97, 928.12],
            'R2': [-2.59, -1.37, 0.05, 0.05],
        })
        st.dataframe(segments, use_container_width=True)
        st.markdown(":red[Premium orders (MAE = $928) show the model lacks a price signal]")

    # ---- TAB 4: Unit Price Impact ----
    with tab4:
        st.subheader("Phase 2: Unit Price Proxy - Transformative Results")

        col_left, col_right = st.columns(2)

        with col_left:
            st.markdown("### Before vs After")
            comparison = pd.DataFrame({
                'Metric': ['R\u00b2', 'MAE ($)', 'SMAPE (%)'],
                'RF Tuned (Phase 1)': [0.723, 87.45, 45.4],
                'RF + UnitPrice (Phase 2)': [0.925, 18.41, 5.6],
                'Improvement': ['+28%', '-79%', '-88%'],
            })
            st.dataframe(comparison, use_container_width=True)

        with col_right:
            st.markdown("### MAE Reduction by Segment")
            seg_comparison = pd.DataFrame({
                'Segment': ['Low', 'Medium', 'High', 'Premium'],
                'Before': [16.18, 49.30, 141.17, 841.30],
                'After': [1.92, 10.76, 23.14, 222.18],
            })
            fig, ax = plt.subplots(figsize=(8, 5))
            x = range(len(seg_comparison))
            width = 0.35
            ax.bar([i - width/2 for i in x], seg_comparison['Before'],
                   width, label='Before', color=CHART_COLORS[5])
            ax.bar([i + width/2 for i in x], seg_comparison['After'],
                   width, label='After', color=CHART_COLORS[0])
            ax.set_xticks(x)
            ax.set_xticklabels(seg_comparison['Segment'])
            ax.set_ylabel("MAE ($)")
            ax.set_title("MAE Before vs After Unit Price Proxy")
            ax.legend()
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        # Feature importance comparison
        st.subheader("Feature Importance: Phase 1 vs Phase 2")
        col_left2, col_right2 = st.columns(2)

        with col_left2:
            st.markdown("**Phase 1 - Top 5 Features**")
            fi_p1 = pd.DataFrame({
                'Feature': ['product_te', 'customer_freq', 'log_quantity', 'shipping_days', 'quantity'],
                'Importance': [0.2554, 0.0637, 0.0562, 0.0532, 0.0510],
            }).sort_values('Importance')
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.barh(fi_p1['Feature'], fi_p1['Importance'], color=CHART_COLORS[3])
            ax.set_title("Phase 1: Top Features")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        with col_right2:
            st.markdown("**Phase 2 - Top 5 Features**")
            fi_p2 = pd.DataFrame({
                'Feature': ['estimated_sales', 'log_est_sales', 'product_te', 'unit_price_proxy', 'subcat_te'],
                'Importance': [0.3672, 0.3543, 0.1373, 0.0892, 0.0135],
            }).sort_values('Importance')
            fig, ax = plt.subplots(figsize=(8, 4))
            colors_fi = [CHART_COLORS[5] if f in ['estimated_sales', 'log_est_sales', 'unit_price_proxy']
                         else CHART_COLORS[3] for f in fi_p2['Feature']]
            ax.barh(fi_p2['Feature'], fi_p2['Importance'], color=colors_fi)
            ax.set_title("Phase 2: Top Features (red = new)")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        st.divider()
        st.subheader("All Models - Final Comparison")
        all_results = pd.DataFrame({
            'Model': ['RF Base', 'RF Tuned', 'XGB Base', 'XGB + log(y)',
                      'RF + UnitPrice', 'RF + UP + log(y)',
                      'XGB + UnitPrice', 'XGB + UP + log(y)'],
            'Phase': ['1', '1', '1', '1', '2', '2', '2', '2'],
            'R2': [0.699, 0.723, 0.717, 0.671, 0.925, 0.910, 0.848, 0.844],
            'MAE': [96.34, 87.45, 81.35, 80.99, 18.41, 20.70, 27.25, 28.31],
            'SMAPE': [56.4, 45.4, 45.0, 35.1, 5.6, 5.3, 8.0, 6.5],
        })
        st.dataframe(
            all_results.style
            .highlight_min(subset=['MAE'], color='#1DB954')
            .highlight_max(subset=['R2'], color='#1DB954'),
            use_container_width=True
        )

        st.success(
            "**Conclusion:** Feature engineering (unit price proxy) produced a 79% MAE reduction. "
            "Algorithm choice (RF vs XGB) made marginal difference with the same features."
        )


# =========================================================================
# PAGE 2: AI CHAT
# =========================================================================
elif page == "AI Chat":

    st.title(f"{APP_ICON} AI Sales Analyst Chat")
    st.caption("Ask questions about the data, EDA, models, or request charts")

    # KPI row
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Orders", f"{len(df):,}")
    col2.metric("Total Sales", f"${df['sales'].sum():,.0f}")
    col3.metric("Total Profit", f"${df['profit'].sum():,.0f}")
    col4.metric("Best Model R\u00b2", "0.925")

    st.divider()

    # Validate API key
    active_key = api_key_input or OPENAI_API_KEY
    if not active_key:
        st.warning(
            "No API key found. Either:\n"
            "1. Create a `.env` file with `OPENAI_API_KEY=sk-...`\n"
            "2. Or paste your key in the sidebar."
        )
        st.stop()

    # Initialize OpenAI client
    from openai import OpenAI
    client = OpenAI(api_key=active_key)

    # Chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask me anything about the Superstore analysis..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

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

        st.session_state.messages.append({"role": "assistant", "content": answer})