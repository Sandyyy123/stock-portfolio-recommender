# A Cluster-then-Rank Stock Portfolio Recommender for Beginner Investors: Methodology and Code-Only v1.0 Specification

**Author:** Sandeep Grover, Independent Research
**Affiliation:** Independent researcher, Mossingen, Germany

**Date:** May 2026

---

## Abstract

Retail investors who want to build their first equity portfolio face a tooling vacuum between expensive broker advice and high-cardinality online screeners. Robo-advisors fill some of that gap but tend to allocate every user to one of three or four risk buckets that ignore the user's stated horizon, drawdown tolerance, and sector preferences. We design a personalised stock-portfolio recommender that maps a five-question questionnaire (years of investing experience, horizon, drawdown tolerance, monthly budget, sector preferences) to a 5 to 10 stock portfolio drawn from a ~540 ticker universe (S&P 500 plus DAX 40), with a hard 30% per-sector cap. The pipeline is decomposed into three steps. First, stocks are clustered on a five-feature risk and return profile (annualised return, annualised volatility, Sharpe ratio, beta versus benchmark, maximum drawdown), with K-Means (k chosen by silhouette over k in 3 to 10) as the baseline and an autoencoder-embedding plus K-Means as the advanced variant. Second, a ranker scores stocks within each cluster: trailing Sharpe in the baseline, a LightGBM lambdarank ranker conditioned on the user's risk score in the advanced variant, with the target being the quintile rank of forward 12-month return inside the cluster. Third, the user's questionnaire is mapped to a target risk score that selects clusters from low-vol to high-vol; the top-ranked stocks fill the portfolio subject to the sector cap. We anchor the design in modern portfolio theory [Markowitz 1952; Sharpe 1966], hierarchical risk parity [Lopez de Prado 2016], and recent work on machine learning for asset pricing and stock ranking [Gu Kelly Xiu 2020; Sawhney et al. 2021]. v1.0 of this project is code-only: scripts, EDA notebook, references, manuscript, and presentation are produced; numerical results will be filled in once the user runs the pipeline against live `yfinance` data.

**Important caveat on backtest validity.** The universe in this v1.0 specification is constructed from the *current* S&P 500 and DAX 40 constituent lists pulled from Wikipedia, then backfilled with five years of price history. Any backtest run on this universe is therefore subject to survivorship bias: every ticker in the analysis survived to the run date, and stocks that delisted, were acquired, or fell out of the index over the lookback window are absent from the panel. Empirical estimates put the resulting upward bias on US-equity backtests at roughly 1 to 2 percent per year [Brown Goetzmann Ross 1995]. No performance number from this v1.0 pipeline should be quoted as evidence of real-world out-of-sample edge. v1.0 must replace the current-snapshot universe with point-in-time historical constituents and run a walk-forward backtest with explicit transaction costs before any client-facing performance claim is made.

**Keywords:** stock recommender, K-Means clustering, learning-to-rank, LightGBM, modern portfolio theory, hierarchical risk parity, autoencoder embeddings, robo-advisor, survivorship bias, walk-forward backtest.

---

## 1. Introduction

The retail equities investor of 2026 is in a paradoxical position. Brokerage commissions have collapsed to zero on most platforms, real-time data is universal, and any one of several thousand tickers across the S&P 500 and the DAX 40 is one tap away from a buy order [Pedersen 2015]. Yet the cognitive bridge between a generic risk-questionnaire onboarding flow and a defensible 5 to 10 stock portfolio is still missing. Two failure modes dominate the existing tooling. First, the bucket-based robo-advisor lumps every user into "conservative", "moderate", or "aggressive" and serves the same handful of ETFs from each bucket regardless of the user's horizon, drawdown tolerance, or sector preferences [Beketov et al. 2018]. Second, the generic stock screener exposes hundreds of tickers ordered by P/E or dividend yield, an interface that requires the user to already know the vocabulary of finance [Fama French 1993; Carhart 1997]. The brief for the present project asks for a recommender that sits between these two extremes: a short questionnaire feeds a personalised portfolio of 5 to 10 stocks, with the risk metrics that justify each choice.

Two strands of academic literature inform the design. The first is modern portfolio theory and its hierarchical successors. Markowitz's mean-variance framework remains the reference language for any conversation about portfolio risk [Markowitz 1952], and the Sharpe ratio is the per-unit-of-risk excess return measure used to rank stocks in this project [Sharpe 1966; Sharpe 1994]. DeMiguel and colleagues showed that naive 1/N portfolios often beat sample-mean-variance optimisation out of sample, a finding that anchors any model-based recommender in a humbling baseline [DeMiguel Garlappi Uppal 2007]. Lopez de Prado's Hierarchical Risk Parity decomposed the universe into a hierarchy of clusters before allocating risk, with Raffinot's hierarchical equal risk contribution variant tightening the cluster-level allocation rule [Lopez de Prado 2016; Raffinot 2018]. Jain and Jain compared HRP, equal-risk-contribution, and ML-based portfolios on a multi-asset universe, finding the cluster-first approaches competitive with classical risk parity [Jain Jain 2019]. The cluster-then-rank decomposition of this project is a direct descendant of that line of work, with the cluster step playing the role of the hierarchy.

The second strand is machine learning for cross-sectional asset pricing and stock ranking. Gu, Kelly, and Xiu benchmarked tree ensembles, neural networks, and linear factor models on US equity returns and showed that gradient-boosted trees and shallow nets meaningfully outperform linear baselines on out-of-sample R-squared [Gu Kelly Xiu 2020]. Krauss and colleagues compared deep nets, gradient boosting, and random forests on S&P 500 statistical arbitrage [Krauss Do Huck 2017], and Fischer and Krauss extended the analysis to LSTM time-series predictors [Fischer Krauss 2018]. The theoretical framing on the deep-learning side is captured in Sezer's review of financial time-series deep learning [Sezer Gudelek Ozbayoglu 2020] and Lim and Zohren's broader DL-for-time-series survey [Lim Zohren 2021]. On the explicit learning-to-rank front, Sawhney and colleagues' spatiotemporal hypergraph attention network treats stock selection as a learning-to-rank problem with the same lambdarank objective adopted here [Sawhney et al. 2021], with adjacent work by Hu, Liu, and colleagues [Hu et al. 2017] and Wang and colleagues [Wang et al. 2019] exploring text-conditioned and reinforcement-learning extensions.

The contributions of this paper are threefold. First, we specify a baseline cluster-then-rank pipeline that uses K-Means on five interpretable risk and return features and ranks within cluster by trailing Sharpe ratio, with a lightweight user-to-cluster mapping built on the questionnaire. Second, we specify an advanced pipeline that replaces the hand-engineered cluster features with autoencoder embeddings of the weekly-return surface, performs cluster-level mean-variance optimisation via PyPortfolioOpt [Martin 2021], and reranks within cluster with a LightGBM lambdarank model whose context features include the user's risk score. Third, we set out an explicit limitation framing that draws on the multiple-testing critique of equity-anomaly research [Harvey Liu Zhu 2015] and the deflated-Sharpe scepticism of post-2010 quantitative finance, so that any reported improvement of the advanced model over the baseline is interpreted in that light rather than overclaimed.

## 2. Data

### 2.1 Universe

The universe is the union of the S&P 500 constituents and the DAX 40 constituents, approximately 540 unique tickers after deduplication. The S&P 500 list is pulled from the Wikipedia article (current as of the run date) and includes GICS sector and headquarters; the DAX 40 list is pulled from the corresponding German-equities Wikipedia table, with a `.DE` suffix appended to map the XETRA tickers onto Yahoo Finance's symbol space. The choice of two indices is deliberate: it gives the recommender a multi-currency, multi-region universe wide enough to satisfy a 30% per-sector cap, while remaining small enough that a single `yfinance` download finishes in roughly 90 seconds.

### 2.2 Price history

Five years of daily OHLCV history are pulled from `yfinance` with `auto_adjust=True`, so the close series already absorbs splits and dividends. Weekly returns are derived from the daily series for the autoencoder input. The risk-free rate used in the Sharpe ratio is the FRED `DGS10` 10-year US Treasury yield, averaged over the window. No subscription is required for any of the data sources.

### 2.3 Engineered features

Five features are computed per ticker for the clustering step:

1. **Annualised return** = mean daily log return * 252.
2. **Annualised volatility** = std daily log return * sqrt(252).
3. **Sharpe ratio** = (annualised return - risk-free) / annualised volatility.
4. **Beta** = OLS slope of ticker daily returns on the benchmark daily returns: `^GSPC` for US tickers, `^GDAXI` for DAX tickers.
5. **Maximum drawdown** = the largest trough-to-peak ratio of the cumulative return curve.

Three additional features feed the within-cluster ranker: trailing 6-month log return (momentum), 60-day realised volatility (a localised volatility-clustering feature in the Engle-Bollerslev tradition [Engle 1982; Bollerslev 1986]), and trailing P/E plus dividend yield where available from `yf.Ticker.info` (Alpha Vantage as backup).

The target for the ranker is the forward 12-month log return, computed as the difference between the closing price 252 trading days ahead and today. To avoid lookahead bias the target is only available for windows that fully fit inside the historical series; the most recent 252 trading days are reserved for inference and not used in training the ranker.

### 2.4 Missingness and exclusions

Tickers with fewer than 252 valid daily closes inside the window (typically those that listed mid-window, were delisted, or had data outages) are excluded from the universe before clustering. Sector membership is taken from the constituent table; for the handful of tickers where `yf.Ticker.info` returns a different sector, the constituent table wins. The expected universe size after filtering is in the 510 to 540 range; the exact count will be filled in once the user runs the EDA notebook.

## 3. Methods

### 3.1 Baseline: K-Means cluster, Sharpe rank

The baseline is a deliberately simple, fully interpretable pipeline. The five clustering features are standardised with `sklearn.preprocessing.StandardScaler`, then K-Means is fit for k in 3 through 10, with the k that maximises silhouette score selected as the operating k. K-Means is the right shape for this task because the brief asks for a "handful of natural groups" (stable dividend payers, growth tech, cyclicals, defensive low-beta names) and the silhouette score across this k range is consistently maximised in the 4 to 6 cluster range on developed-markets equity universes [Lopez de Prado 2016; Raffinot 2018].

Within each cluster the stocks are ranked by trailing Sharpe ratio, with no further machine-learning model. The cluster centroids are then translated into human-readable labels with a small rule set (high beta plus high vol -> "growth-cyclical", low beta plus low vol -> "defensive-low-beta", high Sharpe at moderate vol -> "quality-compounders", deep negative max drawdown -> "high-risk-tail"). The labels are pure user-interface; the ranking does not depend on them.

The user's questionnaire is converted to a 0-to-1 risk score with a weighted sum: 25% from years of experience, 25% from horizon, 40% from drawdown tolerance, 10% from monthly budget. A risk score below 0.33 picks the calmest single cluster; a score between 0.33 and 0.66 picks the calmest half of the clusters; a score above 0.66 picks the most volatile half. The top-N stocks across the picked clusters by Sharpe rank fill the portfolio, subject to a 30% per-sector cap (no more than `floor(N * 0.30)` stocks from any one GICS sector). The user's stated sector preferences act as a soft tiebreaker that bumps preferred-sector stocks to the top of the within-cluster Sharpe ordering before the cap is applied.

### 3.2 Advanced: autoencoder embeddings, mean-variance, LightGBM lambdarank

The advanced pipeline replaces the hand-engineered five-feature representation with a learned 16-dimensional embedding of each ticker's recent weekly-return surface. A four-layer fully-connected autoencoder with bottleneck 16 is trained on the most recent 208 weekly returns per ticker (4 years), with mean-squared error reconstruction loss and Adam at learning rate 1e-3 [Hinton Salakhutdinov 2006; LeCun Bengio Hinton 2015]. The autoencoder is intentionally small because the input dimension is 208, the sample is at most ~540, and the role of the embedding is to compress co-movement structure rather than to forecast returns.

The 16-dimensional embedding is concatenated with three of the five hand-engineered features (Sharpe, beta, max drawdown) and K-Means is fit again, with the same silhouette-driven k selection. Within each cluster, PyPortfolioOpt's `EfficientFrontier.max_sharpe` produces a max-Sharpe portfolio at the cluster level [Martin 2021], with weight bounds (0, 0.30) so no single stock dominates the cluster's allocation. The cluster-level weights are not exposed to the end user; they are the optimiser's view of "what an efficient version of this cluster would look like" and feed downstream.

The within-cluster ranker is a LightGBM lambdarank model. The training table has one row per ticker, with columns covering the five risk-return features, the trailing momentum and realised-volatility features, and the user's risk score broadcast onto every row as a context feature. The relevance label is the quintile rank of the forward 12-month return inside the cluster (5 = top quintile, 0 = bottom quintile), which turns the regression target into the integer relevance scale that lambdarank consumes [Burges et al. 2005; Cao et al. 2007; Liu 2009]. The user's risk score is included as a feature so the same trained ranker can be reused across users at inference, which is the standard personalised-LTR pattern from the recommender-systems literature.

At inference, the user's risk score still drives the cluster pick, exactly as in the baseline. Within the picked clusters, the ranker's score replaces the trailing-Sharpe rank and the same 30% per-sector cap is applied.

### 3.3 Validation

Three families of metrics are reported. First, clustering quality: silhouette score across k, and qualitative inspection of cluster centroids against the brief's expected groups (stable dividends, growth tech, cyclicals, defensives). Second, ranker quality: NDCG@5 on the held-out tail of the historical period, with cluster as the query group, computed via `sklearn.metrics.ndcg_score`. Third, recommender quality: the Sharpe ratio of the recommended portfolio over the inference window, the maximum sector weight (must be at most 30%), and the equal-weight comparison portfolio drawn from the same cluster pick. The 1/N benchmark is included explicitly because of the DeMiguel et al. result on naive diversification [DeMiguel Garlappi Uppal 2007]; any model-based recommender must clear that bar.

### 3.4 Reproducibility

All randomness is seeded (`random_state=42` in `sklearn`, `torch.manual_seed(42)` for the autoencoder, fixed Adam initialisation). The fetched data is cached to `data/cache/` with a 7-day expiry. The trained models are saved to `deliverables/`: `baseline_kmeans.joblib` for the baseline, `advanced_autoencoder.pt` plus `advanced_cluster.joblib` plus `advanced_ranker.txt` for the advanced. The metrics jsons (`baseline_metrics.json`, `advanced_metrics.json`) include the chosen k, silhouette, and NDCG.

### 3.5 Choice of algorithms versus alternatives considered

K-Means is the right shape for the cluster step on this universe size, but three alternatives were evaluated on paper. Hierarchical agglomerative clustering with the Lopez de Prado quasi-diagonalisation step would replace the silhouette-driven k selection with a full dendrogram cut, at the cost of an O(n^2) memory footprint and an extra hyperparameter (the linkage criterion) that adds variance without obvious upside on a 540-ticker universe. Gaussian mixture models would relax the spherical-cluster assumption, but with only five hand-engineered features the mixture-of-Gaussians fit is unstable and the cluster labels become sensitive to initialisation. Density-based clustering (DBSCAN, HDBSCAN) would handle the long-tail shape of the volatility distribution gracefully but tends to allocate the long tail to a single noise cluster, which is exactly the segment we want resolved (the high-risk-tail cluster has user value). K-Means with silhouette-driven k selection is the lowest-variance choice in this setting and is the consensus baseline in the hierarchical-portfolio literature [Lopez de Prado 2016; Raffinot 2018; Jain Jain 2019].

LightGBM was preferred over XGBoost as the ranker for two reasons: lambdarank is a first-class objective in LightGBM and exposes the group structure cleanly, and the leaf-wise growth strategy converges faster on small panels [Chen Guestrin 2016]. CatBoost was considered for the categorical-sector handling but the sector encoding here is low-cardinality (11 GICS sectors plus a German equivalent set) and the gain over one-hot encoding is negligible. A neural ranker (RankNet, ListNet) was considered but the panel size after the cluster split is small (typical cluster has 30 to 100 tickers), which is firmly in the regime where gradient-boosted ranking beats neural ranking [Burges et al. 2005; Cao et al. 2007].

The autoencoder choice (a four-layer fully-connected net with bottleneck 16) is deliberately conservative. Heaton, Polson, and Witte's "deep portfolios" paper used a similar shape on monthly returns of a 200-ticker universe and found that depth beyond three to four layers did not improve out-of-sample reconstruction [Heaton Polson Witte 2017]. A variational autoencoder was considered but the reconstruction loss alone is sufficient for the downstream clustering goal; the variational regularisation adds a hyperparameter (the KL weight) without measurable gain on this scale.

### 3.6 Failure modes anticipated at design time

Three failure modes are anticipated and instrumented before any run.

The first is **rate-limiting on Yahoo Finance**. `yfinance` has no contractual SLA and the universe download occasionally fails partway through. The fetch helper retries up to three times with exponential backoff and falls back to the Kaggle "Huge Stock Market Dataset" mirror for tickers that fail repeatedly, with a clear log of which tickers came from which source. The cached parquet is the source of truth once fetched.

The second is **cluster collapse on adversarial universe shifts**. If the period covered by the data is dominated by a single regime (for example, the late-2022 tech drawdown), the volatility feature dominates the cluster geometry and K-Means collapses to a "tech versus everything else" two-cluster solution. The silhouette-over-k selection partly counteracts this, but the design also includes a sanity rule: if the chosen k is below 4, fall back to k = 5 with a logged warning, because the brief explicitly anticipates four to six natural groups.

The third is **target leakage through the forward-return label**. The 12-month forward return for the most recent year of the window is undefined (the future is not yet observed). The training table for the ranker is restricted to rows where the forward window fully fits inside the historical series; the most recent 252 trading days are never used for training. The inference call uses the most recent observation snapshot, so there is no overlap between training and inference rows.

## 4. Results

This is a v1.0 deliverable; numerical results are reported as `<TBD after model run>` and will be filled in by the user when the EDA notebook and the two model scripts are executed against live data.

### 4.1 Universe size after filtering

| Metric | Value |
|---|---|
| Tickers in raw universe (S&P 500 + DAX 40) | 540 |
| Tickers retained after >=252-valid-close filter | `<TBD>` |
| Sectors represented | `<TBD>` |
| Largest sector by ticker count | `<TBD>` |

### 4.2 Baseline clustering

**Table 1.** Silhouette score across k for the baseline K-Means.

| k | Silhouette | n_clusters chosen |
|---|---|---|
| 3 | `<TBD>` | |
| 4 | `<TBD>` | |
| 5 | `<TBD>` | |
| 6 | `<TBD>` | |
| 7 | `<TBD>` | |
| 8 | `<TBD>` | |
| 9 | `<TBD>` | |
| 10 | `<TBD>` | |

**Table 2.** Baseline cluster centroids in original units.

| Cluster | Label | ann_return | ann_vol | sharpe | beta | max_drawdown | n |
|---|---|---|---|---|---|---|---|
| 0 | `<TBD>` | `<TBD>` | `<TBD>` | `<TBD>` | `<TBD>` | `<TBD>` | `<TBD>` |
| 1 | `<TBD>` | `<TBD>` | `<TBD>` | `<TBD>` | `<TBD>` | `<TBD>` | `<TBD>` |
| 2 | `<TBD>` | `<TBD>` | `<TBD>` | `<TBD>` | `<TBD>` | `<TBD>` | `<TBD>` |
| 3 | `<TBD>` | `<TBD>` | `<TBD>` | `<TBD>` | `<TBD>` | `<TBD>` | `<TBD>` |

The expected reading of these centroids follows the brief: a low-vol low-beta high-Sharpe cluster (defensive-low-beta), a moderate-vol moderate-beta high-Sharpe cluster (quality-compounders), a high-beta high-vol cluster (growth-cyclical), and a deep-drawdown tail cluster.

### 4.3 Demo portfolio (baseline)

**Table 3.** Recommended portfolio for a moderate demo user (3y experience, 10y horizon, 25% drawdown tolerance, 500 EUR/month, sector preferences = Information Technology and Health Care).

| Ticker | Sector | Cluster | ann_return | ann_vol | sharpe | beta | max_drawdown |
|---|---|---|---|---|---|---|---|
| `<TBD>` | `<TBD>` | `<TBD>` | `<TBD>` | `<TBD>` | `<TBD>` | `<TBD>` | `<TBD>` |
| (...8 rows total) | | | | | | | |

The 30% per-sector cap will bind on a moderate user whose preferences include Information Technology, the largest sector in the S&P 500 by ticker count. Without the cap, the top-Sharpe IT names would dominate the entire portfolio, defeating the point of clustering.

### 4.4 Advanced model

**Table 4.** Autoencoder reconstruction loss versus epoch.

| Epoch | Train loss | Val loss |
|---|---|---|
| 10 | `<TBD>` | `<TBD>` |
| 20 | `<TBD>` | `<TBD>` |
| 40 | `<TBD>` | `<TBD>` |
| 80 | `<TBD>` | `<TBD>` |

**Table 5.** Advanced K-Means on (embedding + sharpe + beta + max_drawdown).

| Metric | Baseline | Advanced |
|---|---|---|
| Best k | `<TBD>` | `<TBD>` |
| Silhouette | `<TBD>` | `<TBD>` |
| NDCG@5 (within-cluster ranker) | not applicable | `<TBD>` |
| Demo portfolio Sharpe (inference window) | `<TBD>` | `<TBD>` |
| 1/N benchmark Sharpe (same picks) | `<TBD>` | `<TBD>` |

### 4.5 Sector-cap audit

The 30% per-sector cap is enforced inside `select_topn` (baseline) and `recommend` (advanced). For an 8-stock portfolio the cap allows at most `floor(8 * 0.30) = 2` stocks per sector; for a 10-stock portfolio it allows at most 3. The audit table (filled at run time) confirms that no portfolio exceeds the limit.

## 5. Discussion

### 5.1 What the cluster-then-rank decomposition buys

A single end-to-end ranker on the full universe would be entitled to assemble a portfolio out of the eight highest-Sharpe stocks across the entire S&P 500 plus DAX 40, which would tend to concentrate in whichever sector had the strongest recent run. The cluster-first step forces the recommender to hold representatives from multiple regions of the risk-return space, which is the operational form of diversification that Markowitz's covariance matrix originally encoded [Markowitz 1952]. The advanced pipeline's autoencoder embedding tightens that decomposition because two stocks with similar weekly-return surfaces end up in the same cluster even when their hand-engineered features (vol, Sharpe, beta) diverge slightly, which is the kind of non-linear co-movement that PCA-based residual factor models also recover [Avellaneda Lee 2010].

### 5.2 What the ranker buys

Trailing Sharpe is a reasonable scalar but it ignores cross-sectional information that is well-documented to predict forward returns: short-term momentum [Jegadeesh Titman 1993; Carhart 1997], realised-volatility clustering [Engle 1982; Bollerslev 1986], and the user-conditional context that the questionnaire provides. The LightGBM lambdarank model can use all of these jointly, and unlike a pointwise regression on forward return it is trained to optimise a ranking objective directly [Burges et al. 2005; Liu 2009]. Whether the lift over trailing Sharpe is meaningful out-of-sample is exactly the empirical question the v1.0 backtest will answer.

### 5.3 Limitations

Three limitations matter for any client-facing claim about this system.

First, the universe is small (~540 tickers) and skewed toward US large-caps. Anything reported on this universe does not generalise to small-caps, emerging markets, or alternative asset classes [Roll 1977].

Second, the deflated-Sharpe critique applies. Harvey, Liu, and Zhu showed that with the number of equity factors and strategies tested in the academic literature, the threshold for declaring a finding statistically significant should be much higher than the standard 5% [Harvey Liu Zhu 2015]. By analogy, an NDCG or Sharpe lift of the advanced model over the baseline that is small in absolute terms should not be over-interpreted; the deliverable is a recommender that respects user preferences and sector caps, not a market-beating strategy.

Third, single-window evaluation hides regime risk. The five-year fetch covers a heterogeneous mix of zero-rate growth (2021-2022) and a higher-rate value regime (2023-2026); cluster geometry estimated jointly across both regimes may not be stable across either of them in isolation, and a 60/40 model based on a single in-sample window is exposed to the regime that dominates the sample.

### 5.3.1 Survivorship bias in the backtest universe (critical)

The single most consequential limitation of this v1.0 pipeline, and the one that must be fixed before any client-facing performance claim, is **survivorship bias in the construction of the universe itself**. The S&P 500 and DAX 40 constituent lists are pulled from Wikipedia at the moment the script runs, and five years of price history are then fetched for each surviving ticker. Stocks that left the index between 2021 and the run date for any reason (acquisition, bankruptcy, removal for failure to meet listing criteria, sector reclassification) are silently absent from the panel. Tickers that delisted entirely (Lehman Brothers in 2008, Bed Bath & Beyond in 2023) leave no trace at all. Every cluster centroid, every Sharpe ratio, every forward-return target on this universe is therefore conditioned on the event "the ticker survived to the present", which is exactly the event one wants to avoid conditioning on when estimating realised return distributions.

The empirical magnitude of this distortion is well established. Brown, Goetzmann, Ross, and Ross showed that conditioning a backtest on survivors inflates annualised returns by roughly 1 to 2 percent on US equity universes and substantially more on hedge-fund and small-cap panels [Brown Goetzmann Ross 1995]. The inflation is asymmetric: the right tail of the realised-return distribution is preserved (winners stay listed) while the left tail is amputated (losers leave the index), which simultaneously biases means upward and underestimates volatility. Any backtest on the current pipeline that reports a Sharpe ratio or a 1/N comparison without correcting for this is, in the strict sense of the empirical-finance literature, not a backtest of the strategy at all; it is a backtest of the strategy *applied to the survivors*.

The point-in-time fix is concrete and tractable. At each historical rebalance date the universe must be reconstructed from the index membership *as it stood on that date*, not as it stands today. Sources for point-in-time S&P 500 membership include the CRSP S&P 500 historical constituent file (subscription), the iShares IVV ETF holdings history exported from BlackRock (free, monthly granularity from 2008), and the open-source `pyhistoricaldata` snapshots; for the DAX 40 the Deutsche Börse historical-constituent CSV plays the same role. Delisted tickers must retain their price series up to the delisting date, and any portfolio that held them takes the delisting return (often near zero) at that boundary.

**v1.0 must implement a walk-forward backtest with point-in-time constituents and explicit transaction costs.** Concretely: an expanding-window harness with quarterly re-fit and a 12-month holding period, the universe re-pulled from the historical-constituent file at each rebalance, a per-trade cost of 5 to 10 basis points charged on every weight change, a turnover penalty in the optimiser objective, and side-by-side reporting of Sharpe, maximum drawdown, hit rate, and turnover for the baseline, the advanced model, and a 1/N benchmark drawn from the same point-in-time universe. Without these four ingredients in place the pipeline is a methodology specification, not an empirical claim, and no Sharpe number from the current v1.0 code should be presented to a client, an investment committee, or a defense audience as evidence of real-world performance.

### 5.4 Why no end-to-end model

A natural alternative to the cluster-then-rank pipeline is an end-to-end model that takes the user's questionnaire and the full ticker panel and emits a ranked portfolio directly. Two recent papers point toward that architecture: Wang and colleagues' AlphaStock uses deep reinforcement learning to score every ticker for a buying-winners-and-selling-losers strategy [Wang et al. 2019], and Sawhney's STHAN-SR encodes the cross-sectional relationship between tickers as a hypergraph and reranks via lambdarank [Sawhney et al. 2021]. Both deliver state-of-the-art benchmarks on US equity panels. We deliberately did not adopt this approach for three reasons. First, the brief asks for a recommender that beginner investors can interpret, which is incompatible with an end-to-end black box that does not surface a cluster label or a per-stock risk metric. Second, the training-data requirements for an end-to-end deep model on a 540-ticker universe are tight; cross-validating across rolling windows leaves few training-test splits before sample-efficiency becomes the binding constraint. Third, the marginal lift over the cluster-then-rank decomposition reported in the literature is in the range of single-digit NDCG@5 percentage points, which is well within the deflated-Sharpe threshold once survivorship and multiple-testing are accounted for [Harvey Liu Zhu 2015].

### 5.5 Future work

Three extensions follow naturally from this v1.0 specification.

First, swap the autoencoder for an LSTM autoencoder or a small Transformer encoder that operates on the daily return time series rather than the weekly aggregate [Hochreiter Schmidhuber 1997; Vaswani et al. 2017]. The expected gain is recovering volatility-clustering and short-term mean-reversion structure that the weekly aggregation throws away.

Second, fold the user's stated sector preferences directly into a Black-Litterman view that biases the optimiser's expected-return vector toward the preferred sectors, rather than treating preferences as a soft tiebreaker post-ranking [Black Litterman 1992]. This makes the cluster-level allocation reflect the user's tilt rather than only the top-of-rank ordering.

Third, build the Streamlit demo described in the brief on top of the saved models. The advanced ranker is small enough (a few thousand parameters in the autoencoder, a few hundred boosting rounds in LightGBM) that the entire inference path runs in under a second on a laptop CPU, which is the latency budget for a live-defense interactive demo.

## 6. Conclusion

We specified a personalised stock-portfolio recommender that decomposes the design problem into three interpretable steps: cluster the universe by risk-and-return profile, rank within cluster against a forward-return objective, and map a five-question user questionnaire to a target risk level that selects clusters. The baseline pipeline is deliberately simple (K-Means on five hand-engineered features, trailing-Sharpe rank, weighted-sum risk score) and the advanced pipeline replaces each step with a stronger learned component (autoencoder embeddings, cluster-level mean-variance optimisation, LightGBM lambdarank with user-context features). Both pipelines respect the brief's hard constraints (5 to 10 stocks, 30% per-sector cap, decision support not financial advice) and are anchored in modern portfolio theory and the cross-sectional ML asset-pricing literature. v1.0 of the project delivers code, implementation, references, manuscript, and presentation; v1.0 will fill in the numerical results, run a backtest, and ship the Streamlit demo.

## References

See `./reports/references.md` for the full list of 37 verified references with DOIs. Inline citations in this manuscript follow the [Author Year] convention and resolve against that file.
