# Validation Report - Project 21 (Stock Portfolio Recommender)

**Role:** A / VALIDATOR
**Date:** 2026-05-08
**Overall status:** PASS-WITH-WARNINGS

## Compact summary

Project 21 is a Phase-1 scaffold-only deliverable. Notebook JSON is valid, both Python scripts parse cleanly, manuscript word count (4398) sits inside the 4000-5000 target band, the presentation HTML has zero external `http(s)` resources, all 8 IMRaD sections are present, every concrete method named in Methods has a matching implementation in `src/model_baseline.py` or `src/model_advanced.py`, and 5 of 5 randomly sampled references resolve via CrossRef HTTP 200 with title-match. Em-dash count is 0 across all artefacts and there are no AI-tell phrases. Two issues warrant a WARN: two inline citations (`Beketov et al. 2018`, `Pedersen 2015`) in the manuscript do not appear in `reports/references.md` (orphan citations), and the scaffold contains no executed model artefacts in `deliverables/` (expected since project is scaffold-only, not in #1-#8 executed range).

## Detailed findings

1. **Notebook validity**
   - [PASS] `notebooks/01_EDA.ipynb` parses as JSON (json.load OK).

2. **Python script syntax**
   - [PASS] `src/model_baseline.py` ast.parse OK.
   - [PASS] `src/model_advanced.py` ast.parse OK.

3. **Manuscript word count**
   - [PASS] `wc -w manuscripts/manuscript.md` = 4398 (target 4000-5000).

4. **Self-contained HTML**
   - [PASS] `grep -E 'href="http|src="http' deliverables/presentation.html` returns only DOI hyperlinks inside the references list (5 hits, all `https://doi.org/...`). No external CSS, JS, fonts, or images. The DOI links are in-prose citations, not loaded resources, so the presentation renders fully offline. Treating as PASS for self-containment.

5. **IMRaD completeness**
   - [PASS] All eight expected sections present: Title, Abstract, Introduction, Methods, Results, Discussion, Conclusion, References.

6. **Method drift (Methods section vs code)**
   - [PASS] StandardScaler, KMeans, silhouette, Sharpe, autoencoder, LightGBM lambdarank, PyPortfolioOpt + EfficientFrontier + max_sharpe, NDCG (`ndcg_score`), Adam, MSE, mean-variance optimisation - all present in baseline and/or advanced scripts.
   - [PASS] No methods named in manuscript that are absent from code.

7. **Citation drift (inline `[Author Year]` vs `reports/references.md`)**
   - 36 unique inline citations parsed from the manuscript.
   - [PASS] 34 of 36 map to a reference entry by surname + year proximity.
   - [WARN] `Beketov et al. 2018` is cited in section 1 (Introduction, robo-advisor bucket critique) but no Beketov entry exists in `references.md`. Either add the reference (likely Beketov, Lehmann, Wittke 2018, *Journal of Asset Management*, "Robo Advisors: quantitative methods inside the robots") or remove the inline cite.
   - [WARN] `Pedersen 2015` is cited in section 1 (Introduction, retail-investor framing) but no Pedersen entry exists in `references.md`. Likely Lasse Heje Pedersen 2015, *Efficiently Inefficient* (Princeton). Add the reference or remove the cite.

8. **Live re-verification of 5 random references via CrossRef**
   - [PASS] Markowitz 1952 (10.2307/2975974) HTTP 200, title="Portfolio Selection" - match.
   - [PASS] Lopez de Prado 2016 (10.3905/jpm.2016.42.4.059) HTTP 200, title="Building Diversified Portfolios that Outperform Out of Sample" - match.
   - [PASS] Gu Kelly Xiu 2020 (10.1093/rfs/hhaa009) HTTP 200, title="Empirical Asset Pricing via Machine Learning" - match.
   - [PASS] Sawhney 2021 (10.1609/aaai.v35i1.16127) HTTP 200, title="Stock Selection via Spatiotemporal Hypergraph Attention Network: A Learning to Rank Approach" - match.
   - [PASS] Martin 2021 (10.21105/joss.03066) HTTP 200, title="PyPortfolioOpt: portfolio optimization in Python" - match.
   - 5 of 5 resolve and titles match the manuscript usage.

9. **Em-dash scan**
   - [PASS] 0 occurrences of U+2014 across `brief.md`, `notebooks/01_EDA.ipynb`, `reports/references.md`, both `src/*.py`, `manuscripts/manuscript.md`, and `deliverables/presentation.html`.

10. **AI-tell scan**
    - [PASS] `grep -riE 'verified by [0-9]+ agents|AI-verified|cross-checked by Claude'` returns 0 hits across the project folder.

11. **Checkpoint schema**
    - [PASS] `checkpoint.json` keys = `[project_number, title, methodology, phase, status, needs_main_session_execution, blockers]`. All four required fields (`project_number, title, methodology, status`) present; `phase`, `needs_main_session_execution`, `blockers` are valid extras.

12. **Executed-artefact note (scaffold-only project)**
    - [WARN] `deliverables/` contains only `presentation.html`; no `baseline_kmeans.joblib`, `advanced_autoencoder.pt`, `advanced_ranker.txt`, or `*_metrics.json`. Expected for a Phase-1 scaffold (project #21 is outside the executed #1-#8 range), so this is a WARN not a FAIL. Manuscript explicitly carries `<TBD after model run>` placeholders, which is internally consistent.

## Recommended fixes (none are blockers)

- Add bibliography entries for `Beketov et al. 2018` and `Pedersen 2015` to `reports/references.md`, or strike both inline citations from the Introduction. (Easy 5-minute fix; closes the only substantive gap.)
- Optional: when the user runs the pipeline in Phase 2, the `<TBD>` cells in Results section 4 should be backfilled from the `*_metrics.json` and CSV files the scripts emit, never hand-typed.

## Output file

`/root/AI/liora_projects/21_stock_portfolio/validation_report.md`

Role A complete.
