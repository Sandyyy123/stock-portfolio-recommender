# References - Stock Portfolio Recommender

Verified against CrossRef and OpenAlex (May 2026). Each reference has a resolvable DOI; arXiv-only items are listed with the arXiv identifier, which OpenAlex returns as a canonical entry. Volume, issue, and pages are deliberately stripped per project rule (citation-strip-volume-pages).

## Modern portfolio theory and risk-adjusted metrics

1. **Markowitz, H.** (1952). Portfolio Selection. *The Journal of Finance*. DOI: [10.2307/2975974](https://doi.org/10.2307/2975974). The mean-variance foundation: a portfolio is the linear combination of assets that minimises variance for a given expected return. Used here to motivate the cluster-then-rank decomposition and the variance term in the advanced optimiser.

2. **Sharpe, W. F.** (1966). Mutual Fund Performance. *The Journal of Business*. DOI: [10.1086/294846](https://doi.org/10.1086/294846). Original definition of the Sharpe ratio, the per-unit-of-risk excess return measure used to rank stocks within a cluster in the baseline model.

3. **Sharpe, W. F.** (1994). The Sharpe Ratio. *The Journal of Portfolio Management*. DOI: [10.3905/jpm.1994.409501](https://doi.org/10.3905/jpm.1994.409501). The author's revision and clarification of the 1966 ratio, including the ex ante versus ex post distinction. Cited when justifying the use of trailing Sharpe as an estimator of forward risk-adjusted return.

4. **Fama, E. F., & French, K. R.** (1993). Common risk factors in the returns on stocks and bonds. *Journal of Financial Economics*. DOI: [10.1016/0304-405X(93)90023-5](https://doi.org/10.1016/0304-405X(93)90023-5). The three-factor model (market, size, value) that any contemporary risk-adjusted return analysis benchmarks against.

5. **Carhart, M. M.** (1997). On Persistence in Mutual Fund Performance. *The Journal of Finance*. DOI: [10.1111/j.1540-6261.1997.tb03808.x](https://doi.org/10.1111/j.1540-6261.1997.tb03808.x). Adds the momentum factor to Fama-French. Justifies the inclusion of `momentum_6m` as a ranker feature.

6. **Jegadeesh, N., & Titman, S.** (1993). Returns to Buying Winners and Selling Losers: Implications for Stock Market Efficiency. *The Journal of Finance*. DOI: [10.1111/j.1540-6261.1993.tb04702.x](https://doi.org/10.1111/j.1540-6261.1993.tb04702.x). The original momentum-anomaly paper, basis for momentum-style ranking in the recommender.

7. **Black, F., & Litterman, R.** (1992). Global Portfolio Optimization. *Financial Analysts Journal*. DOI: [10.2469/faj.v48.n5.28](https://doi.org/10.2469/faj.v48.n5.28). Bayesian extension of mean-variance that combines a market-equilibrium prior with investor views; cited when discussing how the questionnaire could later be folded into the advanced optimiser as an investor view.

8. **Roll, R.** (1977). A critique of the asset pricing theory's tests Part I: On past and potential testability of the theory. *Journal of Financial Economics*. DOI: [10.1016/0304-405X(77)90009-5](https://doi.org/10.1016/0304-405X(77)90009-5). The Roll critique on the unobservability of the true market portfolio. Cited in the limitations section.

9. **DeMiguel, V., Garlappi, L., & Uppal, R.** (2007). Optimal Versus Naive Diversification: How Inefficient is the 1/N Portfolio Strategy? *Review of Financial Studies*. DOI: [10.1093/rfs/hhm075](https://doi.org/10.1093/rfs/hhm075). Empirical finding that equal-weighting often beats sample-mean-variance optimisation out of sample. Sets the bar that any model-based recommender must clear.

9a. **Pedersen, L. H.** (2015). *Efficiently Inefficient: How Smart Money Invests and Market Prices Are Determined*. Princeton University Press. DOI: [10.2307/j.ctt1287knh](https://doi.org/10.2307/j.ctt1287knh). Practitioner-academic synthesis of how informed-investor activity shapes price formation; cited in the Introduction to motivate the gap between zero-commission retail tooling and defensible portfolio construction.

9b. **Beketov, M., Lehmann, K., & Wittke, M.** (2018). Robo Advisors: quantitative methods inside the robots. *Journal of Asset Management*. DOI: [10.1057/s41260-018-0092-9](https://doi.org/10.1057/s41260-018-0092-9). Industry-survey paper documenting the bucket-based onboarding pattern of incumbent robo-advisors; cited in the Introduction as the failure mode the present recommender targets.

10. **Avellaneda, M., & Lee, J.-H.** (2010). Statistical arbitrage in the US equities market. *Quantitative Finance*. DOI: [10.1080/14697680903124632](https://doi.org/10.1080/14697680903124632). PCA-based residual factor model of US equity returns. Foundational reference for the autoencoder embedding in the advanced model.

## Hierarchical and clustering-based portfolio construction

11. **Lopez de Prado, M.** (2016). Building Diversified Portfolios that Outperform Out of Sample. *The Journal of Portfolio Management*. DOI: [10.3905/jpm.2016.42.4.059](https://doi.org/10.3905/jpm.2016.42.4.059). Original Hierarchical Risk Parity (HRP) paper. Direct theoretical anchor for the cluster-first portfolio-construction approach used in this project.

12. **Raffinot, T.** (2018). The Hierarchical Equal Risk Contribution Portfolio. *SSRN Electronic Journal*. DOI: [10.2139/ssrn.3237540](https://doi.org/10.2139/ssrn.3237540). Extension of HRP that equalises risk contributions across hierarchical clusters; cited as the closest published analogue to the cluster-then-allocate logic used in the baseline.

13. **Jain, P., & Jain, S.** (2019). Can Machine Learning-Based Portfolios Outperform Traditional Risk-Based Portfolios? *Risks*. DOI: [10.3390/risks7030074](https://doi.org/10.3390/risks7030074). Empirical comparison of HRP, equal-risk-contribution, and ML-based portfolios on a multi-asset universe.

## Machine learning for finance

14. **Heaton, J. B., Polson, N. G., & Witte, J. H.** (2017). Deep learning for finance: deep portfolios. *Applied Stochastic Models in Business and Industry*. DOI: [10.1002/asmb.2209](https://doi.org/10.1002/asmb.2209). Autoencoder reconstruction of asset returns plus deep regression on top; methodological template for the advanced model's embedding stage.

15. **Gu, S., Kelly, B., & Xiu, D.** (2020). Empirical Asset Pricing via Machine Learning. *The Review of Financial Studies*. DOI: [10.1093/rfs/hhaa009](https://doi.org/10.1093/rfs/hhaa009). Large-scale ML benchmark on US equity returns showing tree ensembles and shallow nets outperform linear factor models on out-of-sample R-squared.

16. **Krauss, C., Do, X. A., & Huck, N.** (2017). Deep neural networks, gradient-boosted trees, random forests: Statistical arbitrage on the S&P 500. *European Journal of Operational Research*. DOI: [10.1016/j.ejor.2016.10.031](https://doi.org/10.1016/j.ejor.2016.10.031). Apples-to-apples comparison of GBM, RF, and DNNs on S&P 500 daily returns; canonical reference for GBM as a ranking model on equity panels.

17. **Fischer, T., & Krauss, C.** (2018). Deep learning with long short-term memory networks for financial market predictions. *European Journal of Operational Research*. DOI: [10.1016/j.ejor.2017.11.054](https://doi.org/10.1016/j.ejor.2017.11.054). LSTM-on-equity benchmark, with explicit transaction costs.

18. **Sezer, O. B., Gudelek, M. U., & Ozbayoglu, A. M.** (2020). Financial time series forecasting with deep learning: A systematic literature review. *Applied Soft Computing*. DOI: [10.1016/j.asoc.2020.106181](https://doi.org/10.1016/j.asoc.2020.106181). Survey of DL approaches to financial time series 2005-2019, used to position this project against the literature.

19. **Lim, B., & Zohren, S.** (2021). Time-series forecasting with deep learning: a survey. *Philosophical Transactions of the Royal Society A*. DOI: [10.1098/rsta.2020.0209](https://doi.org/10.1098/rsta.2020.0209). Broader survey of DL for time series, including financial applications and attention-based architectures.

## Stock embeddings and learning-to-rank for finance

20. **Sawhney, R., Agarwal, S., Wadhwa, A., Derr, T., & Shah, R. R.** (2021). Stock Selection via Spatiotemporal Hypergraph Attention Network: A Learning to Rank Approach. *Proceedings of the AAAI Conference on Artificial Intelligence*. DOI: [10.1609/aaai.v35i1.16127](https://doi.org/10.1609/aaai.v35i1.16127). Direct prior art for learning-to-rank applied to stock selection. The advanced model in this project follows the same pairwise-ranker pattern but with simpler temporal features.

21. **Hu, Z., Liu, W., Bian, J., Liu, X., & Liu, T.-Y.** (2017). Listening to Chaotic Whispers: A Deep Learning Framework for News-oriented Stock Trend Prediction. arXiv:[1712.02136](https://arxiv.org/abs/1712.02136). Transformer-style attention over news embeddings for stock-direction prediction. Cited as a textual-feature extension of the price-only ranker used here.

22. **Wang, J., Zhang, Y., Tang, K., & Wu, J.** (2019). AlphaStock: A Buying-Winners-and-Selling-Losers Investment Strategy using Interpretable Deep Reinforcement Attention Networks. arXiv:[1908.02646](https://arxiv.org/abs/1908.02646). End-to-end deep RL for cross-sectional stock selection; useful contrast to the supervised learning-to-rank approach of this project.

## Algorithms and tooling

23. **Breiman, L.** (2001). Random Forests. *Machine Learning*. DOI: [10.1023/A:1010933404324](https://doi.org/10.1023/A:1010933404324). The Random Forest baseline used as a sanity check ranker before LightGBM.

24. **Friedman, J. H.** (2001). Greedy Function Approximation: A Gradient Boosting Machine. *The Annals of Statistics*. DOI: [10.1214/aos/1013203451](https://doi.org/10.1214/aos/1013203451). Original gradient-boosting algorithm; ancestor of XGBoost and LightGBM, both candidates for the intra-cluster ranker.

25. **Chen, T., & Guestrin, C.** (2016). XGBoost: A Scalable Tree Boosting System. *Proceedings of the 22nd ACM SIGKDD*. DOI: [10.1145/2939672.2939785](https://doi.org/10.1145/2939672.2939785). Sparsity-aware regularised GBM; default tabular ranker in the advanced model.

26. **Burges, C. J. C., Shaked, T., Renshaw, E., Lazier, A., Deeds, M.** (2005). Learning to rank using gradient descent. *Proceedings of the 22nd International Conference on Machine Learning*. DOI: [10.1145/1102351.1102363](https://doi.org/10.1145/1102351.1102363). RankNet, the first widely deployed neural learning-to-rank algorithm. Foundational pairwise-ranking reference.

27. **Cao, Z., Qin, T., Liu, T.-Y., Tsai, M.-F., Li, H.** (2007). Learning to rank: from pairwise approach to listwise approach. *Proceedings of the 24th International Conference on Machine Learning*. DOI: [10.1145/1273496.1273513](https://doi.org/10.1145/1273496.1273513). ListNet, the listwise extension of RankNet. Cited when motivating the LightGBM lambdarank objective.

28. **Liu, T.-Y.** (2009). Learning to Rank for Information Retrieval. *Foundations and Trends in Information Retrieval*. DOI: [10.1561/1500000016](https://doi.org/10.1561/1500000016). Survey of pointwise, pairwise, and listwise learning-to-rank; standard reference when discussing the choice of ranking objective.

29. **Hochreiter, S., & Schmidhuber, J.** (1997). Long Short-Term Memory. *Neural Computation*. DOI: [10.1162/neco.1997.9.8.1735](https://doi.org/10.1162/neco.1997.9.8.1735). LSTM, the recurrent architecture that the autoencoder in the advanced model can be swapped for if temporal structure dominates.

30. **Hinton, G. E., & Salakhutdinov, R. R.** (2006). Reducing the Dimensionality of Data with Neural Networks. *Science*. DOI: [10.1126/science.1127647](https://doi.org/10.1126/science.1127647). Original deep autoencoder paper; basis for the return-surface autoencoder in the advanced model.

31. **LeCun, Y., Bengio, Y., & Hinton, G.** (2015). Deep learning. *Nature*. DOI: [10.1038/nature14539](https://doi.org/10.1038/nature14539). Canonical deep-learning review, cited when contrasting end-to-end deep architectures with the cluster-then-rank decomposition.

32. **Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L., Polosukhin, I.** (2017). Attention Is All You Need. arXiv:[1706.03762](https://arxiv.org/abs/1706.03762). Transformer architecture, cited as the upgrade path from the LSTM autoencoder if cross-asset attention is added.

33. **Mikolov, T., Chen, K., Corrado, G., & Dean, J.** (2013). Efficient Estimation of Word Representations in Vector Space. arXiv:[1301.3781](https://arxiv.org/abs/1301.3781). Skip-gram embeddings; methodological template for "stock2vec"-style embeddings learned from co-trading patterns.

34. **Engle, R. F.** (1982). Autoregressive Conditional Heteroscedasticity with Estimates of the Variance of United Kingdom Inflation. *Econometrica*. DOI: [10.2307/1912773](https://doi.org/10.2307/1912773). ARCH model, ancestor of GARCH; cited when discussing volatility-clustering features and time-varying covariance estimates for the optimiser.

35. **Bollerslev, T.** (1986). Generalized autoregressive conditional heteroskedasticity. *Journal of Econometrics*. DOI: [10.1016/0304-4076(86)90063-1](https://doi.org/10.1016/0304-4076(86)90063-1). GARCH model. The realised-volatility feature in the ranker is the simple-moving-window analogue.

36. **Martin, R. A.** (2021). PyPortfolioOpt: portfolio optimization in Python. *The Journal of Open Source Software*. DOI: [10.21105/joss.03066](https://doi.org/10.21105/joss.03066). The library used by the advanced model for the mean-variance frontier and the maximum-Sharpe portfolio.

37. **Harvey, C. R., Liu, Y., & Zhu, H.** (2015). ... and the Cross-Section of Expected Returns. *Review of Financial Studies*. DOI: [10.1093/rfs/hhv059](https://doi.org/10.1093/rfs/hhv059). Multiple-testing critique of the published equity-anomaly literature; cited in the limitations section to justify the deflated-Sharpe-style scepticism applied to the ranker's reported gains.


---

## 2024-2026 additions (post-QA literature scout)

# Additional References - Stock Portfolio Recommender (Project #21)

Independent literature scan for the cluster-then-rank stock recommender. All entries below were resolved live via the CrossRef REST API (`https://api.crossref.org/works/{doi}`) at write time; any candidate that did not resolve was omitted, not padded. Volume, issue, and pages are deliberately stripped per project rule.

The set is complementary to `reports/references.md` (37 entries, mostly seminal works 1952-2021 plus a handful of 2020 ML-for-asset-pricing benchmarks). The focus here is on 2024-2026 literature directly relevant to (a) cluster-then-rank pipelines, (b) hierarchical risk parity successors, (c) deep representations of equity returns, (d) robo-advisor design and risk profiling, and (e) backtest-overfitting diagnostics.

---

## State-of-the-art gaps in current `references.md`

After scanning the existing 37-entry reference list, five concrete SOTA gaps stand out that the current v1.0 manuscript would benefit from citing:

1. **Conditional autoencoder asset-pricing models.** The current advanced model uses a vanilla return-surface autoencoder followed by K-Means; the conditional autoencoder (CAE) line of work (originally Gu, Kelly, Xiu 2021 in JFE; replicated for the Korean market by Kim et al. 2023, see entry 1 below) ties the embedding directly to forward returns through a beta-times-factor decoder. This is a strictly stronger embedding-for-pricing argument than reconstruction-only autoencoders and is the natural bridge between the embedding step and the LightGBM ranker.
2. **Backtest-overfitting diagnostics in the ML era.** The manuscript invokes Harvey, Liu, Zhu 2015 once. Arian, Norouzi Mobarekeh, Seco 2024 (entry 16 below, *Knowledge-Based Systems*) operationalise probabilistic backtest-overfitting diagnostics for ML pipelines specifically; this is the right modern citation for the limitations section, not just the 2015 multiple-testing paper.
3. **Deep-learning Black-Litterman views.** The manuscript mentions Black-Litterman as future work but does not cite any modern operationalisation. Ko and Lee 2025 (entry 9) and Su, Lu, Yen 2026 in *Expert Systems with Applications* (entry 10) both publish deep-learning-derived view vectors for Black-Litterman, exactly the upgrade path described in section 5.5.
4. **Multi-objective ESG-aware portfolio optimisation.** The brief excludes crypto and fixed income but does not address ESG. Feng et al. 2024 in *European Financial Management* (entry 12) and Müller, Joubrel 2025 in *Finance Research Letters* (entry 13) are the current references for sustainable mean-variance with ML-derived views, and the Streamlit demo in v1.0 should at minimum acknowledge them.
5. **Risk-tolerance profiling from text.** The questionnaire-to-risk-score mapping in section 3.1 is a hand-crafted weighted sum. Xing 2024 in *Information Processing and Management* (entry 22) shows how to derive a continuous risk-tolerance score from free-text user responses, which is a credible v1.0 extension that the current manuscript does not cite.

---

## Architectures and embeddings for equity returns (2024-2026)

1. Kim E, Cho T, Koo B, Kang H. Conditional autoencoder asset pricing models for the Korean stock market. PLOS ONE. 2023. DOI:10.1371/journal.pone.0281783

2. Zhu T. Latent factor model in asset pricing: A deep learning approach in the Chinese stock market. Finance Research Letters. 2025. DOI:10.1016/j.frl.2025.108519

3. Yañez C, Kristjanpoller W, Minutolo M. Stock market index prediction using transformer neural network models and frequency decomposition. Neural Computing and Applications. 2024. DOI:10.1007/s00521-024-09931-4

4. Yang J, Li P, Cui Y, Han X, Zhou M. Multi-Sensor Temporal Fusion Transformer for Stock Performance Prediction: An Adaptive Sharpe Ratio Approach. Sensors. 2025. DOI:10.3390/s25030976

5. Lynch S, Derakhshan P, Lynch S. A Novel Hybrid Temporal Fusion Transformer Graph Neural Network Model for Stock Market Prediction. AppliedMath. 2025. DOI:10.3390/appliedmath5040176

6. Kim H. Enhancing Stock2vec: Company Embedding Method Using Search Volume Intensity Exposures. SSRN Electronic Journal. 2024. DOI:10.2139/ssrn.4840419

7. Bi D, Chang L, Yang Y. Iterative Complement-clustering PCA: Uncovering latent industry structures in stock returns. Economics Letters. 2025. DOI:10.1016/j.econlet.2025.112611

## Hierarchical risk parity and clustering-based portfolio construction (2024-2026)

8. Rayfield B. Text based hierarchical risk parity (TBHRP). Financial Markets and Portfolio Management. 2025. DOI:10.1007/s11408-025-00491-8

9. Salas-Molina F, Nin J. Fast hierarchical risk parity methods for portfolio selection. Annals of Operations Research. 2026. DOI:10.1007/s10479-026-07149-2

10. Wattanasin P, Chomtohsuwan T, Kraiwanit T. Dynamic Risk Parity Portfolio Optimization: A Comparative Study with Markowitz and Static Risk Parity. Journal of Risk and Financial Management. 2026. DOI:10.3390/jrfm19020135

11. Palit D, Prybutok V. A Study of Hierarchical Risk Parity in Portfolio Construction. Journal of Economic Analysis. 2024. DOI:10.58567/jea03030006

12. Palit D, Prybutok V. Comparative Study of the Equal-Weight Method and Hierarchical Risk Parity in Portfolio Construction. Finance and Economics Review. 2024. DOI:10.38157/fer.v6i1.609

13. Renzi-Ricci G, Harvey O, Baynes L. From Risk Parity to Outcome Risk Parity: A Review and Extension of the Risk Parity Portfolio with Return Predictability. The Journal of Portfolio Management. 2024. DOI:10.3905/jpm.2024.50.5.073

## Mean-variance, Black-Litterman, and modern optimisation (2024-2026)

14. Ko H, Lee J. Portfolio Management Transformed: An Enhanced Black-Litterman Approach Integrating Asset Pricing Theory and Machine Learning. Computational Economics. 2025. DOI:10.1007/s10614-024-10760-9

15. Su X, Lu K, Yen J. Objective Black-Litterman views through deep learning: A novel hybrid model for enhanced portfolio returns. Expert Systems with Applications. 2026. DOI:10.1016/j.eswa.2025.128868

16. Oriol B, Miot A. Ledoit-Wolf linear shrinkage with unknown mean. Journal of Multivariate Analysis. 2025. DOI:10.1016/j.jmva.2025.105429

17. Yanagi T, Yasumoto Y, Takano Y. Mean-Variance Portfolio Optimization with Shrinkage Estimation for Recommender Systems. Proceedings of the International Conference on Operations Research and Enterprise Systems. 2026. DOI:10.5220/0014285500004055

18. Huang M, Dang S, Bhuiyan M. Multi-objective portfolio optimization for stock return prediction using machine learning. Expert Systems with Applications. 2026. DOI:10.1016/j.eswa.2025.129672

## Cluster-and-rank, deep learning for stock selection (2024-2026)

19. Ashrafzadeh M, Sadrani M, Zolfani S. Deep learning and machine learning models for portfolio optimization: Enhancing return prediction with stock clustering. Results in Engineering. 2025. DOI:10.1016/j.rineng.2025.106263

20. Alzaman C. Deep learning in stock portfolio selection and predictions. Expert Systems with Applications. 2024. DOI:10.1016/j.eswa.2023.121404

21. Alzaman C. Optimizing Portfolio Selection Through Stock Ranking and Matching: A Reinforcement Learning Approach. SSRN Electronic Journal. 2024. DOI:10.2139/ssrn.4742704

22. Manogna R, Kulkarni N. Portfolio Optimization Model for Stock Price Prediction Using Machine Learning. Journal of Statistical Theory and Applications. 2025. DOI:10.1007/s44199-025-00140-z

23. Liu Z. Multivariate machine learning algorithm for stock return prediction modeling. Proceedings of the 2024 4th International Conference on Big Data and Computing. 2024. DOI:10.1145/3718751.3718943

24. Xu Y. Deep reinforcement learning-driven intelligent portfolio management with green computing: Sustainable portfolio optimisation. Sustainable Computing: Informatics and Systems. 2025. DOI:10.1016/j.suscom.2025.101125

## Robo-advisor design, user profiling, and behavioural inputs (2024-2026)

25. Ahmad U, Van Keulen M, Briassouli A, Saad M. Cognitive biases, Robo advisor and investment decision psychology: An investor's perspective from New York stock exchange. Acta Psychologica. 2025. DOI:10.1016/j.actpsy.2025.105048

26. Xing F. Financial risk tolerance profiling from text. Information Processing and Management. 2024. DOI:10.1016/j.ipm.2024.103704

27. Hornuf L, Merkle C, Zeisberger S. Nudging Investors towards Sustainability: A Field Experiment with a Robo-Advisor. SSRN Electronic Journal. 2025. DOI:10.2139/ssrn.5051082

## Sustainable / ESG-aware portfolio construction (2024-2026)

28. Feng X, von Mettenheim H, Sermpinis G, Stasinakis C. Sustainable Portfolio Construction via Machine Learning: ESG, SDG and Sentiment. European Financial Management. 2024. DOI:10.1111/eufm.12531

29. Müller L, Joubrel M. A novel approach to sustainable mean-variance portfolio optimization: Accounting for ESG-related uncertainty. Finance Research Letters. 2025. DOI:10.1016/j.frl.2025.108056

30. Truyols-Pont M, Bilbao-Terol A, Arenas-Parra M. Machine Learning for Sustainable Portfolio Optimization Applied to a Water Market. Mathematics. 2024. DOI:10.3390/math12243975

31. Gaurav A, Baishnab K, Singh P. Intelligent ESG portfolio optimization: A multi-objective AI-driven framework for sustainable investments in the Indian stock market. Sustainable Futures. 2025. DOI:10.1016/j.sftr.2025.100832

## Backtest overfitting and validation diagnostics (2024-2026)

32. Arian H, Norouzi Mobarekeh D, Seco L. Backtest overfitting in the machine learning era: A comparison of out-of-sample testing methods in a synthetic controlled environment. Knowledge-Based Systems. 2024. DOI:10.1016/j.knosys.2024.112477

