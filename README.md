# Superstore Sales Forecasting

Predictive modeling pipeline for order-level revenue forecasting using the Superstore dataset (9,994 transactions, 21 features). The analysis progresses from exploratory data analysis through feature engineering and model tuning, culminating in a unit price proxy experiment that reduces prediction error by 79%.

## Results Summary

| Phase | Best Model | R² | MAE | SMAPE |
|-------|-----------|-----|-----|-------|
| Baseline | XGBoost Base | 0.717 | $81.35 | 45.0% |
| Tuned | RF Tuned | 0.723 | $87.45 | 45.4% |
| + Unit Price Proxy | **RF + UnitPrice** | **0.925** | **$18.41** | **5.6%** |

## Project Structure

```
├── 02_EDA_and_Forecasting_Sales.ipynb   # Main analysis notebook
├── Sample - Superstore.csv              # Source dataset
└── README.md
```

## Dataset

The Superstore dataset contains transaction records from a US retail company spanning 2014 to 2017. Each row represents a single order line with attributes including product category, customer segment, geographic region, shipping mode, quantity, discount, and profit.

**Target variable:** `Sales` (revenue per order line, in USD).

The target distribution is heavily right-skewed (skewness = 12.97), with a median of $54 and a maximum exceeding $22,000. This characteristic drives several design decisions throughout the analysis, including the evaluation of log-transformed targets and robust loss functions.

## Methodology

### 1. Exploratory Data Analysis

- Univariate analysis of all numeric and categorical features.
- Outlier detection via the IQR method (1,167 flagged observations, retained as legitimate high-value orders).
- Bivariate analysis including correlation matrices, scatter plots, categorical boxplots, and discount-profit interaction analysis.
- Temporal decomposition revealing year-over-year growth and Q4 seasonality.

### 2. Feature Engineering

27 features organized in four groups:

- **Temporal:** shipping days, cyclic month encoding (sin/cos), quarter flags, day of week, normalized year.
- **Business logic:** discount-quantity interaction, binary flags for discount presence and heavy discount (>=30%), log-transformed quantity.
- **Aggregation:** order size (line items per order), customer purchase frequency, day of month, month-end flag.
- **Encoding:** label encoding for tree-based models, target encoding (median Sales per group) computed exclusively from training data to prevent temporal leakage.

Features removed for cause:
- `profit_margin`: data leakage (computed from the target variable).
- `region`, `segment`, `customer`, `shipmode`: correlations with Sales below 0.01.

### 3. Train/Test Split

Temporal partitioning: 2014-2016 for training (6,682 rows), 2017 for testing (3,312 rows). Random splits were avoided because they would allow the model to learn from future transactions, inflating performance estimates.

### 4. Models Evaluated

**Random Forest** (bagging):
- Base, tuned (RandomizedSearchCV + TimeSeriesSplit), and log-target variants.

**XGBoost** (gradient boosting):
- Base, tuned, early stopping, Huber loss (robust to outliers), and log-target variants.

### 5. Unit Price Proxy Experiment

The error analysis from Phase 1 revealed that the model's primary weakness was the absence of a direct product price signal. The `product_te` feature (median historical transaction value) conflated unit price with quantity and discount effects.

The experiment constructs a **unit price proxy**: the median unit price per product, computed as `Sales / (Quantity × (1 - Discount))` using only training data. A hierarchical fallback (product → sub-category → global median) handles unseen products without leaking test information.

From this proxy, an **estimated sales** feature is derived: `UnitPrice × Quantity × (1 - Discount)`. This feature achieves r = 0.986 with actual Sales, compared to r = 0.868 for `product_te`.

**Impact by price segment (RF + UnitPrice vs RF Tuned):**

| Segment | MAE Before | MAE After | Reduction |
|---------|-----------|----------|-----------|
| Low ($0-50) | $16.18 | $1.92 | -88% |
| Medium ($50-200) | $49.30 | $10.76 | -78% |
| High ($200-1K) | $141.17 | $23.14 | -84% |
| Premium ($1K+) | $841.30 | $222.18 | -74% |

## Key Findings

1. **Feature engineering matters more than algorithm selection.** The unit price proxy improved R² from 0.723 to 0.925, while switching between Random Forest and XGBoost with the same features produced marginal differences.

2. **Log-target transformation helps XGBoost but hurts Random Forest.** XGBoost's sequential boosting learns residual structure more effectively in log-space. Random Forest's averaging mechanism already provides natural outlier robustness, and the expm1 back-transformation amplifies errors for high-value orders.

3. **Aggressive discounts (>=30%) erode profitability.** Mean profit turns negative beyond the 30% discount threshold, justifying the `heavy_discount` binary feature.

4. **Product identity is the strongest predictor.** Target encoding at the product level (r = 0.868) far exceeds all other individual features, because products have relatively stable price points.

## Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
```

## Usage

1. Place `Sample - Superstore.csv` in the same directory as the notebook.
2. Run all cells sequentially in `02_EDA_and_Forecasting_Sales.ipynb`.
3. The notebook executes the complete pipeline from data loading through final model comparison.

## License

This project uses the [Superstore Sales dataset](https://www.kaggle.com/datasets/vivek468/superstore-dataset-final), publicly available on Kaggle for educational and analytical purposes.
