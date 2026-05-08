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
