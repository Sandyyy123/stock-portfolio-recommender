# Data sources - Project 21 Stock Portfolio Recommender

This project does **not** ship a bulk pre-downloaded dataset because all the data sources are fast to fetch at runtime via free APIs. The model scripts pull on demand and cache parquet files into `data/cache/`. This README documents the exact fetch commands so the scripts are reproducible.

## WARNING: survivorship bias in the universe (read before running any backtest)

The universe construction described below pulls the **current** S&P 500 and DAX 40 constituent lists from Wikipedia and then fetches five years of price history for those tickers. This is a survivorship-biased universe:

- Every ticker in the panel survived to the run date.
- Stocks that delisted, were acquired, or fell out of the index between the start of the lookback window and today are absent.
- Any backtest run on this panel inflates realised returns by roughly 1 to 2 percent per year on US equities [Brown, Goetzmann, Ross 1995], with an asymmetric bias: winners stay, losers vanish.

**Phase 1 deliverables on this universe are a methodology specification, not an empirical performance claim.** Phase 2 must replace the current-snapshot universe with point-in-time historical constituents (CRSP S&P 500 historical-constituent file, iShares IVV holdings history from BlackRock, Deutsche Borse historical DAX list) keyed by rebalance date, and run a walk-forward backtest with explicit transaction costs (5 to 10 basis points per trade, plus a turnover penalty) before any Sharpe ratio, NDCG, or 1/N comparison from this pipeline is shown to a client.

The corresponding warning banners live at the top of `src/model_baseline.py` and `src/model_advanced.py`, and `manuscripts/manuscript.md` section 5.3.1 expands the issue. Do not remove any of those banners while the universe is still built from a current-snapshot Wikipedia scrape.

## Stock universe (~540 tickers)

### S&P 500 constituents
- Source: Wikipedia article `https://en.wikipedia.org/wiki/List_of_S%26P_500_companies` (constituent list, GICS sector, headquarters).
- Mirror with stable schema: `https://datahub.io/core/s-and-p-500-companies` (CSV).
- Fetch (Python):

```python
import pandas as pd
sp = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", header=0)[0]
sp = sp.rename(columns={"Symbol": "ticker", "GICS Sector": "sector"})
sp[["ticker", "Security", "sector", "GICS Sub-Industry"]].to_csv("data/cache/sp500_universe.csv", index=False)
```

### DAX 40 constituents
- Source: Wikipedia article `https://en.wikipedia.org/wiki/DAX` (table of current constituents). Symbols are XETRA tickers, e.g. `SAP.DE`, `BMW.DE`. Add the `.DE` suffix for `yfinance` compatibility.
- Fetch (Python):

```python
import pandas as pd
dax = pd.read_html("https://en.wikipedia.org/wiki/DAX")[3]   # current constituents table
dax["ticker"] = dax["Ticker symbol"].astype(str) + ".DE"
dax[["ticker", "Company"]].to_csv("data/cache/dax40_universe.csv", index=False)
```

## Daily OHLCV history (primary source: yfinance)

`yfinance` is a free unofficial wrapper around Yahoo Finance. Five years of daily data for ~540 tickers takes about 90 seconds and roughly 30 MB of parquet on disk.

```bash
pip install yfinance pandas pyarrow
```

```python
import yfinance as yf
import pandas as pd
from pathlib import Path

CACHE = Path("data/cache"); CACHE.mkdir(parents=True, exist_ok=True)
tickers = pd.concat([
    pd.read_csv(CACHE / "sp500_universe.csv"),
    pd.read_csv(CACHE / "dax40_universe.csv"),
])["ticker"].unique().tolist()

prices = yf.download(tickers, start="2021-01-01", auto_adjust=True, group_by="ticker", progress=False, threads=True)
prices.to_parquet(CACHE / "ohlcv_2021_to_now.parquet")
```

`auto_adjust=True` rolls splits and dividends into the close series, which is the convention for return-based features.

## Fundamentals (P/E, EPS, dividend yield)

- Primary: `yf.Ticker(t).info` (free, returns trailingPE, forwardPE, dividendYield, sector, marketCap). Note: rate-limited and intermittently inconsistent, so cache once and never refetch in the same session.
- Backup: Alpha Vantage free tier (5 requests / minute, 500 / day). Set `ALPHAVANTAGE_API_KEY` in `.env` and use the `OVERVIEW` endpoint:

```bash
curl "https://www.alphavantage.co/query?function=OVERVIEW&symbol=AAPL&apikey=$ALPHAVANTAGE_API_KEY"
```

The advanced model only needs trailing P/E, dividend yield, and sector; missing values are imputed by sector median.

## Risk-free rate (FRED 10-year US Treasury)

Used to compute the Sharpe ratio. FRED series ID `DGS10` (daily, in percent).

```python
import pandas_datareader as pdr
rf = pdr.DataReader("DGS10", "fred", start="2021-01-01") / 100.0   # to decimal
rf.to_parquet("data/cache/risk_free_dgs10.parquet")
```

No API key required.

## Offline backup (Kaggle)

If Yahoo Finance rate-limits the entire run (rare), the Kaggle "Huge Stock Market Dataset" by Boris Marjanovic provides ~7,000 US tickers with daily OHLCV through 2017. It is **not** an active mirror, so it cannot be used for the most recent 12-month forward-return target. It is only kept as a fallback for the historical clustering features.

```bash
# Requires kaggle CLI configured at ~/.kaggle/kaggle.json
kaggle datasets download -d borismarjanovic/price-volume-data-for-all-us-stocks-etfs -p data/kaggle_backup --unzip
```

Approximate size: 700 MB unpacked. **Do not download by default**; only use if the runtime fetch fails twice.

## Caching policy

- All fetched data is written to `data/cache/`.
- The two model scripts (`src/model_baseline.py`, `src/model_advanced.py`) check the cache first and only refetch if files are older than 7 days.
- The cache directory is gitignored (see `data/.gitignore`).

## Data dictionary (engineered features)

| Column | Unit | Source | Used by |
|--------|------|--------|---------|
| `ann_return` | dimensionless | derived from `Close` | clustering |
| `ann_vol` | dimensionless | derived from daily log returns | clustering |
| `sharpe` | dimensionless | (`ann_return` - rf_avg) / `ann_vol` | clustering, ranking |
| `beta` | dimensionless | OLS slope against `^GSPC` (US) or `^GDAXI` (DE) | clustering |
| `max_drawdown` | dimensionless | trough/peak of cumulative return curve | clustering |
| `sector` | GICS string | Wikipedia constituent table | clustering, sector-cap constraint |
| `mkt_cap_bucket` | {micro, small, mid, large, mega} | derived from `yf.info.marketCap` | clustering |
| `pe_ttm` | ratio | yfinance / Alpha Vantage | ranking |
| `div_yield` | ratio | yfinance / Alpha Vantage | ranking |
| `momentum_6m` | dimensionless | 126-trading-day return | ranking |
| `realized_vol_60d` | dimensionless | 60-day rolling std of daily log returns | ranking |
| `fwd_return_12m` | dimensionless | TARGET, forward 12-month return | ranking (training only) |
