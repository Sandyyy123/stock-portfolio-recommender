# CHANGES - Project 21 Stock Portfolio Recommender

## 2026-05-08: Survivorship-bias caveat added across deliverables

**Trigger:** IMPROVER report (`improvements.md`) flagged that the universe is constructed from the current Wikipedia S&P 500 and DAX 40 constituent lists, then backtested on stocks that survived to today. This is fatal to any client-facing performance claim and must be addressed before Phase 2.

### What changed

1. **`manuscripts/manuscript.md`**
   - Abstract: added a bold "Important caveat on backtest validity" paragraph stating that the Phase 1 universe is survivorship-biased, that empirical inflation is roughly 1 to 2 percent per year on US equities (Brown, Goetzmann, Ross 1995), and that no Phase 1 number can be quoted as out-of-sample edge.
   - Keywords: added "survivorship bias" and "walk-forward backtest" so the limitation is searchable.
   - Section 5.3 (Limitations): demoted the old one-sentence survivorship mention into the broader limitations list and added a dedicated subsection 5.3.1 ("Survivorship bias in the backtest universe (critical)"). The new subsection covers (a) how the universe is constructed from current snapshots, (b) why this inflates returns asymmetrically (winners stay, losers vanish), (c) the concrete point-in-time fix using CRSP, iShares IVV holdings history, or Deutsche Borse historical DAX records, and (d) an explicit "Phase 2 must implement a walk-forward backtest with point-in-time constituents and explicit transaction costs" mandate naming the four required ingredients (expanding-window harness with quarterly re-fit, point-in-time universe, 5 to 10 bps transaction cost plus turnover penalty, side-by-side reporting against a 1/N benchmark drawn from the same point-in-time universe).

2. **`src/model_baseline.py`**
   - Added a multi-line WARNING banner at the very top of the file (above the existing module docstring) flagging the survivorship-bias issue and the Phase 2 to-do.

3. **`src/model_advanced.py`**
   - Added the same WARNING banner at the top.

4. **`data/README.md`**
   - Added a "WARNING: survivorship bias in the universe (read before running any backtest)" section immediately after the file's opening paragraph, before the universe-construction instructions. The section names the bias source, quantifies the empirical inflation, lists the point-in-time data sources for Phase 2, and tells future readers not to remove the banner until the universe is rebuilt.

### Why this matters

The IMPROVER report named survivorship bias as a "HIGH" weakness and the cure (a walk-forward backtest with point-in-time constituents and transaction costs) as the single TOP recommendation. The bias is a textbook one (Brown, Goetzmann, Ross 1995) and is the difference between a defensible recommender and a finance-101 mistake. Phase 1 of this project is a methodology specification only; the caveat now appears at every entry point a reader can hit (manuscript abstract, manuscript limitations, both model scripts, data README) so that no downstream consumer of the artefacts can mistake the Phase 1 deliverable for an empirical performance claim.

### What was NOT changed

- Algorithm code paths in `src/model_baseline.py` and `src/model_advanced.py` are untouched. The banners are documentation only; the universe-construction logic still pulls from Wikipedia. Replacing it with point-in-time data is a Phase 2 task.
- Numerical results in the manuscript Results section remain `<TBD>` placeholders. The survivorship-bias caveat is independent of whether those cells are filled in, and applies to any number that lands in them.
- The reference list in `reports/references.md` is unchanged; Brown, Goetzmann, Ross 1995 was already in scope for the Limitations section.

### Next action for Phase 2

Implement `src/eval_backtest.py` with:
- An expanding-window walk-forward harness, quarterly re-fit, 12-month holding period.
- A point-in-time historical-constituent file keyed by rebalance date (CRSP or iShares IVV holdings export for S&P 500; Deutsche Borse historical DAX list for DAX 40).
- A per-trade cost of 5 to 10 basis points plus a turnover penalty in the optimiser objective.
- Side-by-side reporting of Sharpe ratio, maximum drawdown, hit rate, and turnover for the baseline, the advanced model, and a 1/N benchmark drawn from the same point-in-time universe.

Only after `eval_backtest.py` is in place and runs cleanly may any performance number from this project be shown to a client, an investment committee, or a defense audience.
