# Superstore Sales Forecasting

An end-to-end data science pipeline for order-level revenue forecasting, from exploratory analysis through machine learning to a GenAI-powered chatbot. Built with the Superstore dataset (9,994 transactions, 2014–2017).

The project demonstrates that **feature engineering matters more than algorithm selection**: a unit price proxy reduced prediction error by 79%, while switching between Random Forest and XGBoost only changed MAE by ~$6.

## Results Summary

| Phase | Best Model | R² | MAE | SMAPE |
|-------|-----------|-----|-----|-------|
| Baseline | XGBoost Base | 0.717 | $81.35 | 45.0% |
| Tuned | RF Tuned | 0.723 | $87.45 | 45.4% |
| + Unit Price Proxy | **RF + UnitPrice** | **0.925** | **$18.41** | **5.6%** |

## Project Structure

```
├── data/
│   └── Sample - Superstore.csv          # Source dataset (Kaggle)
│
├── notebooks/
│   └── 02_EDA_and_Forecasting_Sales.ipynb  # Full analysis: EDA → Features → Models
│
├── src/
│   ├── __init__.py
│   ├── config.py                        # App settings, API key loading, constants
│   ├── data_loader.py                   # CSV loading, derived columns, data summary
│   ├── chat_engine.py                   # GPT integration, RAG, query/chart generation
│   └── knowledge_base.txt              # RAG context: all findings, metrics, conclusions
│
├── app.py                               # Streamlit application entry point
├── requirements.txt
├── .env                                 # API key (not tracked in git)
└── README.md
```

## Dataset

The Superstore dataset contains transaction records from a US retail company spanning 2014 to 2017. Each row represents a single order line with attributes including product category, customer segment, geographic region, shipping mode, quantity, discount, and profit.

**Target variable:** `Sales` (revenue per order line, in USD)

The target distribution is heavily right-skewed (skewness = 12.97), with a median of $54 and a maximum exceeding $22,000. This characteristic drives several design decisions, including log-transformed targets and robust loss functions.

## Methodology

### 1. Exploratory Data Analysis

- Univariate analysis of all numeric and categorical features
- Outlier detection via the IQR method (1,167 flagged observations, retained as legitimate high-value orders)
- Bivariate analysis: correlation matrices, scatter plots, categorical boxplots, discount–profit interaction
- Temporal decomposition revealing year-over-year growth and Q4 seasonality
- Key finding: same quantity produces wildly different sales due to missing unit price signal

### 2. Feature Engineering

27 features organized in four groups, each motivated by specific EDA findings:

**Temporal** — capture seasonality and time patterns:
- `shipping_days`: days between order and ship date (urgency signal)
- `month_sin`, `month_cos`: cyclic encoding so Dec and Jan are adjacent
- `is_q4`: binary flag for Q4 (peak season identified in EDA)
- `quarter`, `order_dow`, `day_of_month`, `year_norm`, `is_month_end`

**Business logic** — capture discount dynamics:
- `disc_x_qty`: discount × quantity interaction (50% off on 14 units ≠ on 1 unit)
- `heavy_discount`: binary flag for discounts ≥30% (profit turns negative at this threshold)
- `discount_flag`: binary flag for any discount > 0
- `log_quantity`: log-transform to compress skewed distribution

**Aggregation** — capture order and customer context:
- `order_size`: line items per order (bulk vs single purchase)
- `customer_freq`: total unique orders per customer (repeat buyer signal)

**Encoding** — convert categories to price-level numbers:
- `product_te` (r = 0.868): median Sales per product from training data
- `subcat_te` (r = 0.422): median Sales per sub-category
- `category_te` (r = 0.217): median Sales per category
- `state_te`: median Sales per state

**Removed for cause:**
- `profit_margin`: data leakage (computed from the target variable)
- `region_te`, `segment_te`, `customer_te`, `shipmode_te`: correlation with Sales below 0.01

### 3. Train/Test Split

Temporal partitioning: 2014–2016 for training (6,682 rows), 2017 for testing (3,312 rows). Random splits were avoided because they would allow the model to learn from future transactions, inflating performance estimates. Hyperparameter tuning used `TimeSeriesSplit` cross-validation.

### 4. Models Evaluated

**Random Forest** (bagging): base, tuned (RandomizedSearchCV + TimeSeriesSplit), and log-target variants.

**XGBoost** (gradient boosting): base, tuned, early stopping, Huber loss (robust to outliers), and log-target variants.

All seven configurations plateaued around $80 MAE — the limitation was in the features, not the algorithms.

### 5. Unit Price Proxy Experiment

Error analysis revealed the model's primary weakness: no direct product price signal. The `product_te` feature conflated unit price with quantity and discount effects.

**Solution — three steps:**

1. **Compute unit price**: median of `Sales / (Quantity × (1 - Discount))` per product, using training data only
2. **Estimate expected sales**: `UnitPrice × Quantity × (1 - Discount)` → achieves r = 0.986 with actual Sales
3. **Handle unseen products**: hierarchical fallback (product → sub-category → global median) + `price_confidence` flag

**Impact by price segment:**

| Segment | MAE Before | MAE After | Reduction |
|---------|-----------|----------|-----------|
| Low ($0–50) | $16.18 | $1.92 | −88% |
| Medium ($50–200) | $49.30 | $10.76 | −78% |
| High ($200–1K) | $141.17 | $23.14 | −84% |
| Premium ($1K+) | $841.30 | $222.18 | −74% |

### 6. GenAI Integration — Streamlit Chatbot

A RAG-powered chatbot that allows non-technical users to explore the data and findings through natural language.

**Architecture:**

```
User Question → Streamlit (app.py) → GPT-4o-mini + RAG context
                                          │
                    ┌─────────────────────┼─────────────────────┐
                    │                     │                     │
              Direct Answer        [QUERY_NEEDED]        [CHART_NEEDED]
           (from knowledge base)   GPT → pandas code     GPT → matplotlib code
                                   → eval(code, df)      → exec(code)
                                   → natural language     → st.pyplot(fig)
```

**Three capabilities:**

| Capability | Example Query | How It Works |
|-----------|--------------|-------------|
| RAG Knowledge | *"What is the best model and why?"* | GPT responds with exact metrics from `knowledge_base.txt` |
| Live Data Query | *"How many orders are in California?"* | GPT generates pandas code, executes against real CSV |
| Dynamic Charts | *"Show me sales by category and year"* | GPT generates matplotlib code with dark theme |

**Key design decisions:**
- **RAG over fine-tuning**: all findings and metrics injected into the system prompt via `knowledge_base.txt`
- **Code generation + execution**: GPT generates pandas/matplotlib code that runs against the real DataFrame
- **Bilingual**: responds in the same language the user writes (English or Spanish)
- **Low temperature (0.3)**: minimizes hallucination while keeping responses natural

## Setup & Usage

### Prerequisites

- Python 3.9+
- OpenAI API key ([get one here](https://platform.openai.com/api-keys))

### Installation

```bash
git clone https://github.com/HelenaGomezV/retail-data-platform-final-project.git
cd retail-data-platform-final-project
pip install -r requirements.txt
```

### Running the Analysis Notebook

```bash
jupyter notebook notebooks/02_EDA_and_Forecasting_Sales.ipynb
```

### Running the Streamlit Chatbot

```bash
echo "OPENAI_API_KEY=sk-your-key-here" > .env
streamlit run app.py
# Open http://localhost:8501
```

### Example Queries

```
What is the best model and why?
Which sub-categories are most profitable?
What happens to profit when discount exceeds 30%?
How many orders are in the West region?
Show me total sales by category and year
¿Cuál es la subcategoría con más ventas?
```

## Key Findings

1. **Feature engineering > algorithm selection.** The unit price proxy improved R² from 0.723 to 0.925, while switching between RF and XGBoost changed MAE by only ~$6. The two unit price features account for 72% of total feature importance.

2. **Log-target helps XGBoost, hurts Random Forest.** XGBoost's sequential boosting learns residuals more effectively in log-space. RF's averaging + expm1 back-transformation amplifies errors for high-value orders.

3. **Discounts ≥30% destroy profitability.** Mean profit turns negative beyond the 30% threshold — an actionable business insight.

4. **Product identity is the strongest predictor.** Target encoding at the product level (r = 0.868) far exceeds all other features, because products have relatively stable price points.

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Data Analysis | pandas, NumPy, matplotlib, seaborn |
| Machine Learning | scikit-learn, XGBoost |
| GenAI | OpenAI GPT-4o-mini |
| RAG Context | knowledge_base.txt |
| Web App | Streamlit |
| Config | python-dotenv |

## License

This project uses the [Superstore Sales dataset](https://www.kaggle.com/datasets/vivek468/superstore-dataset-final), publicly available on Kaggle for educational and analytical purposes.