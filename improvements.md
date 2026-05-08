# IMPROVER report - Project 21 Stock Portfolio Recommender

**Role:** B (IMPROVER) - independent reviewer, recommendations only.
**Scope:** brief.md, data/README.md, notebooks/01_EDA.ipynb, src/model_baseline.py, src/model_advanced.py, manuscripts/manuscript.md, reports/references.md, deliverables/presentation.html, checkpoint.json.
**Output:** recommendations only. No file modifications. No GitHub push.

## Top recommendation

**Add a walk-forward backtest with explicit survivorship-bias and transaction-cost handling, evaluated against a 1/N benchmark, before any client-facing claim.**

The current pipeline trains the LightGBM lambdarank on the entire history, then uses the most recent observation snapshot for inference. There is no out-of-sample evaluation, no rolling re-fit, and no acknowledgement that the universe (S&P 500 + DAX 40 constituents pulled from Wikipedia "today") only contains stocks that survived to the run date. The DeMiguel et al. 1/N result is even cited as the benchmark in the manuscript, but no run actually computes it. Concrete next steps: (1) implement an expanding-window walk-forward harness in `src/eval_backtest.py` with quarterly re-fit and a 1-year holding period, (2) hold a frozen historical constituent list (point-in-time S&P 500 from a snapshot dataset such as CRSP or the iShares historical constituents file) so delisted tickers stay in, (3) charge a per-trade cost of 5-10 bps and a turnover penalty, (4) report Sharpe, max drawdown, hit-rate, and turnover for baseline, advanced, and 1/N on the same draws. This single change converts the project from a Phase 1 spec into a defensible empirical claim.

---

## Weaknesses, by dimension

### 1. Methodology - cluster step is brittle and dependence-blind [HIGH]

K-Means on five hand-engineered features (annualised return, annualised volatility, Sharpe, beta, max drawdown) treats the universe as point cloud in a Euclidean space the data does not actually live in. Annualised return and Sharpe are mechanically correlated (Sharpe is a function of the same numerator), max drawdown is bounded above by zero and heavy-tailed, and beta is computed against two different benchmarks (^GSPC for US, ^GDAXI for DE) which makes a direct cluster comparison across exchanges questionable. Hierarchical Risk Parity (HRP, Lopez de Prado 2016) is even cited in the references but never used.

**Action:** replace silhouette-K-Means with HRP single-linkage on a quasi-diagonalised correlation matrix of weekly returns (`scipy.cluster.hierarchy.linkage(corr_distance, method='single')` then cut at the elbow). HRP gives clusters that respect actual return co-movement rather than hand-engineered marginals, and dovetails directly with the cluster-level mean-variance step the advanced model already runs.

### 2. Methodology - autoencoder is too small, target is wrong [HIGH]

The advanced autoencoder is a four-layer fully-connected net on 208-dim weekly returns with bottleneck 16, trained with MSE reconstruction loss. On a ~540-ticker universe with potentially fewer than 540 rows after filtering, this risks memorising idiosyncratic noise. More fundamentally, MSE on raw returns is not what the downstream task needs - clustering wants a representation that preserves co-movement, not pointwise reconstruction.

**Action:** swap to a contrastive embedding (SimCLR-style with augmentations of weekly windows, or a triplet loss on (ticker, same-sector positive, random negative)). Alternative: a temporal convolutional autoencoder (1D Conv layers over the time axis) which respects temporal locality. Reduce embedding dim to 8 to match sample size; current 16-dim is borderline.

### 3. Experimental design - no ablations, no statistical test, no calibration [HIGH]

The manuscript's Results section is filled with `<TBD>` placeholders. There is no plan to compare baseline against advanced on the same draws, no plan to ablate the autoencoder vs hand-engineered features, no plan to test whether the LightGBM lift over trailing-Sharpe ranking is significantly different from zero, and no plan to calibrate the user-risk-score-to-cluster mapping (the cutoffs 0.33 and 0.66 are arbitrary).

**Action:** add an ablation table to the protocol with at least four rows: (a) baseline, (b) advanced minus autoencoder (use only hand-features in cluster step), (c) advanced minus user-context (drop user_risk feature in ranker), (d) full advanced. Use a paired Diebold-Mariano test on out-of-sample Sharpe across rolling windows to test significance. Calibrate the risk-score cutoffs by gridsearch on a holdout.

### 4. Reproducibility - no requirements.txt, no torch seed, fragile fetch [HIGH]

`src/model_advanced.py` imports torch, lightgbm, pypfopt, sklearn, joblib, pandas, numpy, but neither the project root nor `data/` ships a `requirements.txt` or `pyproject.toml`. The Wikipedia HTML scrape in the EDA notebook (`pd.read_html(...)[3]` for the DAX table) is positionally indexed and breaks the moment Wikipedia adds or reorders a table - a known-stable failure mode. `torch.manual_seed(42)` is mentioned in the manuscript but never actually called in `model_advanced.py`; only `random_state=42` for sklearn is set.

**Action:** ship a `requirements.txt` pinning torch>=2.2, lightgbm>=4.0, pypfopt>=1.5, scikit-learn>=1.4, pandas>=2.2, numpy>=1.26, yfinance>=0.2.40, pandas_datareader. Add `torch.manual_seed(42); np.random.seed(42)` at the top of `train_autoencoder`. Replace the Wikipedia scrape with a static CSV mirror (datahub.io for S&P 500, a checked-in `dax40_universe.csv` for DAX) and document the refresh cadence.

### 5. Survivorship bias and point-in-time data [HIGH]

The brief and `data/README.md` both build the universe by pulling the *current* S&P 500 + DAX 40 constituent list from Wikipedia, then fetching 5 years of price history for those tickers. This is a textbook survivorship bias: every ticker in the analysis survived to today, so any backtest on this universe overstates returns by 1-2% per year on US equities (Brown, Goetzmann, Ross 1995). The manuscript even names this in section 5.3 limitations, then proceeds to ship the pipeline as if the limitation does not bind.

**Action:** add a point-in-time historical-constituent file (the iShares ETF constituent history, the CRSP S&P 500 constituent file, or the open-source `pyhistoricaldata` snapshots) keyed by year. At each rebalance, only include tickers that were S&P 500 or DAX 40 members on that date. This single change is the difference between a defensible recommender and a finance-101 mistake.

### 6. Manuscript and presentation - results are placeholders, no figures committed [MEDIUM]

The manuscript ships with every numerical cell as `<TBD>` and the presentation has no plots. Phase 1 is described as "code-only" but the deliverable list in the brief explicitly includes "verified references (>=20)" and "self-contained presentation" - both of which are present, but with no actual cluster visualisation, no t-SNE / UMAP of the embedding, no demo-portfolio table. A defense audience cannot evaluate the design without at least one rendered cluster scatterplot.

**Action:** the moment the user runs the EDA notebook, generate three static plots and commit them under `deliverables/figures/`: (a) the silhouette-vs-k curve, (b) a 2D PCA of the standardised feature matrix coloured by K-Means cluster, (c) a t-SNE of the autoencoder embedding coloured by GICS sector to show whether the embedding rediscovers sector structure. Inline these into the manuscript Figure 1-3 placeholders and the presentation. Do not ship the manuscript with `<TBD>` cells to a client.

### 7. Domain - finance-specific gaps the brief glosses over [MEDIUM]

Several finance-specific issues are not addressed: (a) currency mismatch (DAX 40 prices are in EUR, S&P 500 in USD; combining annualised returns without an FX adjustment biases the cross-sectional comparison), (b) dividend treatment (`auto_adjust=True` rolls dividends into the close, which is the right convention for total-return clustering but is incompatible with the dividend-yield ranker feature pulled separately from `yf.Ticker.info`), (c) factor exposure (the cluster step ignores the Fama-French style/value/momentum factors that the references cite), (d) regime change (a single 5-year window through 2021-2026 covers two regimes - 2021-2022 zero-rate growth and 2023-2026 higher-rate value; one cluster solution across both regimes is suspect).

**Action:** add an FX layer that converts DAX returns into USD using FRED EURUSD before clustering. Document the dividend convention explicitly. Add a Fama-French 5-factor projection as an ablation feature set. Run the cluster step separately on 2021-2023 and 2024-2026 and check stability via the adjusted Rand index.

### 8. Constraint handling - sector cap is the only constraint [MEDIUM]

The brief specifies "no more than 30% per sector" as the only hard constraint, and the code enforces it via a greedy `floor(N * 0.30)` integer cap. This is a weak portfolio constraint set: there is no minimum-position-size, no maximum-position-size at the asset level, no liquidity floor (avg daily volume), no exchange or currency cap, and no overall risk budget (target portfolio volatility). A real robo-advisor offering would also constrain the portfolio's beta to a target level matched to the user's risk score.

**Action:** rewrite the selection step as a small linear program (cvxpy or scipy.optimize.linprog): maximise sum of ranker scores subject to (a) sum of weights = 1, (b) per-asset weight <= 0.20, (c) per-sector weight <= 0.30, (d) portfolio beta within +/- 0.15 of the user's target beta. This replaces the greedy sector-cap heuristic with a defensible optimiser pass.

### 9. User questionnaire - heuristic risk score is unjustified [MEDIUM]

The 0-to-1 risk score uses fixed weights of 25/25/40/10 across (years_experience, horizon, drawdown_tolerance, monthly_budget) with no theoretical or empirical justification. Drawdown tolerance gets the largest weight, which is reasonable, but the 25% weight on years_experience is arguable - someone with 1 year of experience and a 30-year horizon may rationally tolerate more risk than a 20-year-veteran nearing retirement. The cutoffs (<0.33, 0.33-0.66, >0.66) are also arbitrary and produce a step function that flips a user from "calmest cluster only" to "calmest half" at the 0.33 boundary.

**Action:** ground the questionnaire in the Grable-Lytton risk-tolerance scale (a 13-item validated instrument from the household-finance literature, doi:10.1023/A:1022920614803) or the Cordell FRSI. At minimum, replace the hard cutoffs with a smooth softmax over cluster volatilities weighted by user risk score, so the cluster mix changes continuously rather than in steps.

---

## Priority summary

| # | Improvement | Priority |
|---|-------------|----------|
| 1 | Replace K-Means with HRP on weekly-return correlation distance | HIGH |
| 2 | Swap MSE autoencoder for contrastive / temporal-conv encoder; reduce dim to 8 | HIGH |
| 3 | Add ablation table, paired Diebold-Mariano test, calibrate risk-score cutoffs | HIGH |
| 4 | Ship requirements.txt; set torch seed; replace Wikipedia scrape with static mirror | HIGH |
| 5 | Use point-in-time historical constituents to kill survivorship bias | HIGH |
| 6 | Generate cluster, embedding, and demo-portfolio figures; remove `<TBD>` cells before defense | MEDIUM |
| 7 | FX conversion for DAX, document dividend convention, factor projection, regime stability | MEDIUM |
| 8 | Replace greedy sector cap with cvxpy LP including beta target and asset-weight bounds | MEDIUM |
| 9 | Ground questionnaire in Grable-Lytton FRSI; smooth cluster picks | MEDIUM |
| - | Walk-forward backtest with survivorship-corrected universe and transaction costs | TOP |

---

## Notes on what is already strong

- The reference list (37 entries) is broad, properly DOI-anchored, and well-mapped to manuscript sections.
- The cluster-then-rank decomposition is a legitimate architectural choice and dovetails with the HRP literature.
- The "decision support, not financial advice" framing and the 30% sector cap are correct guardrails.
- The IMRaD manuscript structure is complete and the narrative is anchored in the right citations.
- The `data/README.md` is exemplary for reproducibility - explicit fetch commands, cache policy, fallback Kaggle dataset, data dictionary.
- The `failure modes anticipated at design time` section (3.6) is a rare and valuable inclusion for a Phase 1 spec.

These strengths are why the eight improvements above are high-leverage rather than throwaway: the foundation is sound, the gaps are at the level of empirical rigour and finance-specific defaults.
