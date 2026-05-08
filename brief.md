# Project 21 - Stock Portfolio Recommender

**Track:** Data Scientist - **Difficulty:** 8/10 - **Status:** Phase 1 scaffold
**Domain:** wealth-tech / robo-advisor (Liora-issued from Drive Catalogue, April 2026)
**Source brief:** `/root/AI/.tmp/Liora_Catalogue_Projects/Proposal - Stock Portfolio Recommender.pdf`

## Problem statement (verbatim from the brief)

A retail investor who wants to start picking individual stocks usually has two options: pay for a broker's recommendation, or open an online screener and stare at hundreds of tickers without knowing where to begin. The tools that try to bridge this gap tend to either give the same generic "aggressive" or "moderate" bucket to everyone, or assume the user already knows the vocabulary of finance.

The idea of this project is to build a recommender that takes a short questionnaire about the user (years of investing experience, time horizon, how much loss they can tolerate, monthly budget, sector preferences) and returns a small portfolio of 5 to 10 stocks picked from a known universe such as the S&P 500 and the DAX 40, along with the risk metrics that justify each choice.

## Three-step pipeline (per brief)

1. **Cluster** stocks by historical risk and return profile (volatility, Sharpe ratio, beta against the index, maximum drawdown, sector) using K-Means or hierarchical clustering. Expected groups: stable dividend payers, growth tech, cyclicals, defensive low-beta names.
2. **Rank within cluster.** Train a regression model (linear baseline, then Random Forest or Gradient Boosting) to estimate the expected 12-month return for each stock from fundamentals plus a few technical features. Goal is intra-cluster ranking, not market timing.
3. **Map user to clusters.** Translate the questionnaire (experience, horizon, drawdown tolerance, monthly budget, sector preferences) into a target risk profile, pick the relevant clusters, and return the top-N stocks per cluster, with the constraint that no more than 30% of the portfolio sits in a single sector.

## Deliverables (Liora full-format)

| Artefact | Path | Status (Phase 1) |
|----------|------|-------------------|
| Data fetch documentation | `data/README.md` | written |
| EDA notebook (raw, not executed) | `notebooks/01_EDA.ipynb` | scaffolded, awaits user run |
| Verified references (>=20) | `reports/references.md` | live-verified |
| Baseline model (cluster + intra-cluster Sharpe rank) | `src/model_baseline.py` | code-only |
| Advanced model (autoencoder embedding + MV optimiser + LightGBM ranker) | `src/model_advanced.py` | code-only |
| Manuscript (IMRaD, 4000-5000 words) | `manuscripts/manuscript.md` | written, numbers as `<TBD after model run>` |
| Self-contained presentation | `deliverables/presentation.html` | written |
| Pipeline status JSON | `checkpoint.json` | written |

## Universe and target

- **Universe:** S&P 500 constituents + DAX 40 constituents (~540 tickers).
- **Period:** 5 years of daily OHLCV (default 2021-01-01 to fetch date).
- **Engineered features for clustering:** annualised return, annualised volatility, Sharpe ratio (risk-free = 10y US Treasury via FRED), beta versus benchmark (S&P 500 for US tickers, DAX for German tickers), maximum drawdown, sector (GICS), market capitalisation bucket.
- **Engineered features for intra-cluster ranking:** rolling 12-month return, 6-month momentum, P/E and dividend yield where available (Alpha Vantage backup), 60-day realised volatility, sector dummies.
- **Target for the ranker:** forward 12-month return (computed from price series, cropped to the most recent fully-observed window).

## Constraints from the brief (decision-support, not financial advice)

- 5 to 10 stocks per recommended portfolio.
- No more than 30% allocation in any single sector.
- The output is decision support for a beginner investor, not financial advice. The Streamlit demo (out of Phase 1 scope) shows that disclaimer prominently.

## Out of scope for Phase 1

- Streamlit web app (brief mentions it; Phase 1 is code-only deliverables, not a live demo).
- Backtesting an actual trading strategy on out-of-sample data (the deliverable is a recommender, not a portfolio manager).
- Cryptocurrency or fixed-income universes.

## Open questions for the defense

1. Should the questionnaire be Likert-scored (1-5) or use binary/categorical answers? Default = mix (sliders for tolerance/budget, multi-select for sector preferences).
2. Should the recommender update on a fixed schedule (e.g. monthly) or per user request? Default = on request, with cached features.
3. How to handle the long tail of small-cap volatility once DAX is included? Default = exclude any ticker with less than 3 years of trading history.
