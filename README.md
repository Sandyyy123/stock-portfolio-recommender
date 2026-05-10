![Python](https://img.shields.io/badge/Python-3.10%2B-blue) ![Fintech](https://img.shields.io/badge/Fintech-portfolio-gold) ![License](https://img.shields.io/badge/license-CC%20BY--NC%204.0-lightgrey)

# Stock Portfolio Recommender — Cluster-then-Rank

Cluster-then-rank recommendation system suggesting personalised stock portfolios based on investor risk profile and factor exposure.

---

## Task

**Recommendation System + Portfolio Optimization**

---

## Architecture

```
Investor Profile → Cluster Assignment → Factor Scoring → MVO Optimization → Portfolio Output
```

---

## Key Features

- Investor clustering by risk tolerance, horizon, and factor preferences (k-means)
- Per-cluster stock ranking by Sharpe, momentum, and quality factors
- Mean-variance portfolio optimisation (cvxpy) with cardinality constraints
- Cold-start handling via investor onboarding questionnaire → cluster assignment
- Backtested return and drawdown metrics vs S&P 500 benchmark

---

## Dataset

Yahoo Finance (yfinance) — S&P 500 constituents, factor data

---

## Project Structure

```
├── src/
│   ├── model_baseline.py      # Baseline model
│   └── model_advanced.py      # Advanced model
├── notebooks/
│   └── 01_EDA.ipynb           # Exploratory analysis
├── manuscripts/
│   └── manuscript.md          # IMRaD writeup
├── reports/
│   └── references.md          # Verified references
├── deliverables/
│   └── presentation.html      # Self-contained HTML
├── data/
│   └── README.md              # Dataset download instructions
└── requirements.txt
```

---

## Quick Start

```bash
git clone https://github.com/Sandyyy123/stock-portfolio-recommender.git
cd stock-portfolio-recommender
pip install -r requirements.txt

# See data/README.md for dataset download
python src/model_baseline.py
python src/model_advanced.py
```

---

## Tech Stack

`scikit-learn · cvxpy · pandas · yfinance · matplotlib`

---

## Author

**Dr. Sandeep Grover** — PhD Data Science, independent ML researcher, Mössingen, Germany.

---

## License

MIT
