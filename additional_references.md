# Additional References - Stock Portfolio Recommender (Project #21)

Independent literature scan for the cluster-then-rank stock recommender. All entries below were resolved live via the CrossRef REST API (`https://api.crossref.org/works/{doi}`) at write time; any candidate that did not resolve was omitted, not padded. Volume, issue, and pages are deliberately stripped per project rule.

The set is complementary to `reports/references.md` (37 entries, mostly seminal works 1952-2021 plus a handful of 2020 ML-for-asset-pricing benchmarks). The focus here is on 2024-2026 literature directly relevant to (a) cluster-then-rank pipelines, (b) hierarchical risk parity successors, (c) deep representations of equity returns, (d) robo-advisor design and risk profiling, and (e) backtest-overfitting diagnostics.

---

## State-of-the-art gaps in current `references.md`

After scanning the existing 37-entry reference list, five concrete SOTA gaps stand out that the current Phase 1 manuscript would benefit from citing:

1. **Conditional autoencoder asset-pricing models.** The current advanced model uses a vanilla return-surface autoencoder followed by K-Means; the conditional autoencoder (CAE) line of work (originally Gu, Kelly, Xiu 2021 in JFE; replicated for the Korean market by Kim et al. 2023, see entry 1 below) ties the embedding directly to forward returns through a beta-times-factor decoder. This is a strictly stronger embedding-for-pricing argument than reconstruction-only autoencoders and is the natural bridge between the embedding step and the LightGBM ranker.
2. **Backtest-overfitting diagnostics in the ML era.** The manuscript invokes Harvey, Liu, Zhu 2015 once. Arian, Norouzi Mobarekeh, Seco 2024 (entry 16 below, *Knowledge-Based Systems*) operationalise probabilistic backtest-overfitting diagnostics for ML pipelines specifically; this is the right modern citation for the limitations section, not just the 2015 multiple-testing paper.
3. **Deep-learning Black-Litterman views.** The manuscript mentions Black-Litterman as future work but does not cite any modern operationalisation. Ko and Lee 2025 (entry 9) and Su, Lu, Yen 2026 in *Expert Systems with Applications* (entry 10) both publish deep-learning-derived view vectors for Black-Litterman, exactly the upgrade path described in section 5.5.
4. **Multi-objective ESG-aware portfolio optimisation.** The brief excludes crypto and fixed income but does not address ESG. Feng et al. 2024 in *European Financial Management* (entry 12) and Müller, Joubrel 2025 in *Finance Research Letters* (entry 13) are the current references for sustainable mean-variance with ML-derived views, and the Streamlit demo in Phase 2 should at minimum acknowledge them.
5. **Risk-tolerance profiling from text.** The questionnaire-to-risk-score mapping in section 3.1 is a hand-crafted weighted sum. Xing 2024 in *Information Processing and Management* (entry 22) shows how to derive a continuous risk-tolerance score from free-text user responses, which is a credible Phase 2 extension that the current manuscript does not cite.

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
