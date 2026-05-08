"""Baseline recommender for Project 21: Stock Portfolio Recommender.

Pipeline (matches the brief):
  1. Load engineered features from data/cache/features.parquet (built by notebooks/01_EDA.ipynb).
  2. K-Means cluster the universe on (ann_return, ann_vol, sharpe, beta, max_drawdown), k chosen by silhouette over k in {3..10}.
  3. Within each cluster, rank stocks by Sharpe ratio.
  4. Map a user's questionnaire to a target risk profile, pick clusters, return top-N stocks subject to the
     "no more than 30% per sector" constraint.

Run from the project root:
    python src/model_baseline.py

This script is code-only for Phase 1; the user runs it manually after the EDA notebook has produced the feature table.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "cache"
DELIVERABLES = PROJECT_ROOT / "deliverables"
FEATURE_COLS = ["ann_return", "ann_vol", "sharpe", "beta", "max_drawdown"]
SECTOR_CAP = 0.30  # no more than 30% of the recommended portfolio in any single sector


# ----------------------------- user model ------------------------------------ #


@dataclass
class UserProfile:
    """Output of the questionnaire (the brief's five inputs)."""

    years_experience: int          # 0 to 30+
    horizon_years: int             # planned holding period
    drawdown_tolerance: float      # max % drawdown the user can sit through, e.g. 0.20 for 20%
    monthly_budget_eur: float      # informational only at this stage; affects N_stocks downstream
    sector_preferences: list[str]  # GICS sectors the user wants to overweight; empty = no preference

    def risk_score(self) -> float:
        """Map the questionnaire to a 0..1 risk score.

        0 = lowest risk (all-defensive bias), 1 = highest risk (growth/cyclical bias).
        Heuristic from the brief; tuned downstream against backtests.
        """
        s = 0.0
        s += min(self.years_experience, 20) / 20 * 0.25
        s += min(self.horizon_years, 20) / 20 * 0.25
        s += min(self.drawdown_tolerance, 0.50) / 0.50 * 0.40
        s += 0.10 if self.monthly_budget_eur >= 500 else 0.05
        return float(np.clip(s, 0.0, 1.0))


# ----------------------------- clustering ------------------------------------ #


def fit_kmeans(features: pd.DataFrame) -> tuple[KMeans, StandardScaler, int]:
    """Fit K-Means on the standardised feature matrix; choose k by silhouette over {3..10}."""
    X = features[FEATURE_COLS].dropna()
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)

    best_k, best_score, best_model = None, -np.inf, None
    for k in range(3, 11):
        km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(Xs)
        score = silhouette_score(Xs, km.labels_)
        if score > best_score:
            best_k, best_score, best_model = k, score, km

    print(f"[baseline] best k = {best_k}, silhouette = {best_score:.4f}")
    return best_model, scaler, best_k


def annotate_clusters(features: pd.DataFrame, km: KMeans, scaler: StandardScaler) -> pd.DataFrame:
    out = features.dropna(subset=FEATURE_COLS).copy()
    out["cluster"] = km.predict(scaler.transform(out[FEATURE_COLS]))
    centroids = pd.DataFrame(scaler.inverse_transform(km.cluster_centers_), columns=FEATURE_COLS)
    centroids["cluster"] = range(len(centroids))
    print("[baseline] cluster centroids (original units):")
    print(centroids.round(3).to_string(index=False))
    return out, centroids


def label_clusters(centroids: pd.DataFrame) -> dict[int, str]:
    """Heuristic, human-readable cluster labels.

    Pure labelling for the demo; the actual ranking does not depend on these strings.
    Rules: high beta + high vol -> 'growth/cyclical'; low beta + low vol -> 'defensive';
    high Sharpe at moderate vol -> 'quality compounders'; deep negative max_drawdown -> 'high-risk tail'.
    """
    labels: dict[int, str] = {}
    for _, row in centroids.iterrows():
        c = int(row["cluster"])
        if row["beta"] > 1.2 and row["ann_vol"] > 0.30:
            labels[c] = "growth-cyclical"
        elif row["beta"] < 0.8 and row["ann_vol"] < 0.22:
            labels[c] = "defensive-low-beta"
        elif row["sharpe"] > 0.8 and row["ann_vol"] < 0.30:
            labels[c] = "quality-compounders"
        elif row["max_drawdown"] < -0.50:
            labels[c] = "high-risk-tail"
        else:
            labels[c] = f"mixed-{c}"
    return labels


# ----------------------------- ranking and selection ------------------------- #


def pick_clusters_for_user(profile: UserProfile, centroids: pd.DataFrame, labels: dict[int, str]) -> list[int]:
    """Choose 1 to 3 clusters that best match the user's risk score."""
    score = profile.risk_score()
    centroids = centroids.copy()
    # Composite cluster risk: scaled vol + |beta-1| - sharpe (lower is calmer).
    centroids["risk_idx"] = (
        centroids["ann_vol"].rank(pct=True)
        + (centroids["beta"] - 1).abs().rank(pct=True)
        - centroids["sharpe"].rank(pct=True)
    )
    centroids = centroids.sort_values("risk_idx")
    n_clusters = len(centroids)

    if score < 0.33:
        picks = centroids.head(1)["cluster"].tolist()                # calmest cluster only
    elif score < 0.66:
        picks = centroids.iloc[: max(2, n_clusters // 2)]["cluster"].tolist()
    else:
        picks = centroids.tail(max(2, n_clusters // 2))["cluster"].tolist()
    print(f"[baseline] user risk score = {score:.2f} -> picked clusters = {[labels[c] for c in picks]}")
    return picks


def select_topn(features: pd.DataFrame, picks: list[int], profile: UserProfile, n_total: int = 8) -> pd.DataFrame:
    """Within the picked clusters, rank by Sharpe, apply 30% per-sector cap, return n_total stocks."""
    pool = features[features["cluster"].isin(picks)].copy()
    pool = pool.sort_values("sharpe", ascending=False)

    # Soft preference for the user's preferred sectors: bump those rows to the top.
    if profile.sector_preferences:
        prefer_mask = pool["sector"].isin(profile.sector_preferences)
        pool = pd.concat([pool[prefer_mask], pool[~prefer_mask]])

    selected, sector_count, max_per_sector = [], {}, max(1, int(np.floor(n_total * SECTOR_CAP)))
    for ticker, row in pool.iterrows():
        sec = row["sector"]
        if sector_count.get(sec, 0) >= max_per_sector:
            continue
        selected.append(ticker)
        sector_count[sec] = sector_count.get(sec, 0) + 1
        if len(selected) == n_total:
            break

    chosen = pool.loc[selected, ["sector", "ann_return", "ann_vol", "sharpe", "beta", "max_drawdown", "cluster"]]
    return chosen


# ----------------------------- entry point ----------------------------------- #


def main() -> None:
    feat_path = DATA_DIR / "features.parquet"
    if not feat_path.exists():
        raise FileNotFoundError(
            f"{feat_path} not found. Run notebooks/01_EDA.ipynb first to build the feature table."
        )

    features = pd.read_parquet(feat_path)
    print(f"[baseline] features: {features.shape[0]} tickers x {features.shape[1]} cols")

    km, scaler, best_k = fit_kmeans(features)
    features_with_cluster, centroids = annotate_clusters(features, km, scaler)
    labels = label_clusters(centroids)

    # Demo profile: a moderate user
    demo_profile = UserProfile(
        years_experience=3,
        horizon_years=10,
        drawdown_tolerance=0.25,
        monthly_budget_eur=500,
        sector_preferences=["Information Technology", "Health Care"],
    )
    picks = pick_clusters_for_user(demo_profile, centroids, labels)
    portfolio = select_topn(features_with_cluster, picks, demo_profile, n_total=8)

    print("\n[baseline] recommended portfolio for demo profile:")
    print(portfolio.round(3).to_string())

    DELIVERABLES.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"kmeans": km, "scaler": scaler, "k": best_k, "feature_cols": FEATURE_COLS, "labels": labels},
        DELIVERABLES / "baseline_kmeans.joblib",
    )
    portfolio.to_csv(DELIVERABLES / "baseline_demo_portfolio.csv")
    metrics = {
        "k": best_k,
        "silhouette": float(silhouette_score(scaler.transform(features[FEATURE_COLS].dropna()), km.labels_)),
        "n_tickers": int(len(features_with_cluster)),
        "demo_portfolio_size": int(len(portfolio)),
        "demo_user_risk_score": demo_profile.risk_score(),
    }
    (DELIVERABLES / "baseline_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"\n[baseline] saved -> {DELIVERABLES / 'baseline_kmeans.joblib'}")
    print(f"[baseline] saved -> {DELIVERABLES / 'baseline_demo_portfolio.csv'}")
    print(f"[baseline] saved -> {DELIVERABLES / 'baseline_metrics.json'}")


if __name__ == "__main__":
    main()
