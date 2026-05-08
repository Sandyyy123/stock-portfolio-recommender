"""Advanced recommender for Project 21: Stock Portfolio Recommender.

Pipeline:
  1. Build a per-ticker return-surface tensor (52 weekly returns per year x 4 years), feed into a small
     PyTorch autoencoder, take the bottleneck as a 16-dim stock embedding.
  2. Cluster on (embedding, sharpe, beta, max_drawdown) with K-Means (k chosen by silhouette).
  3. Use PyPortfolioOpt to compute a max-Sharpe mean-variance portfolio per cluster (cluster-level optimisation,
     not asset-level), giving each cluster a "risk profile" weight vector.
  4. Train a LightGBM lambdarank reranker that scores each ticker per cluster, conditioned on the user's
     questionnaire (risk score and horizon), with the target = forward 12-month return rank.
  5. At inference, pick clusters by user risk profile, take the top-N reranked tickers, apply the 30%
     per-sector cap, return the recommended portfolio.

This script is code-only for Phase 1; runs after notebooks/01_EDA.ipynb has produced features.parquet
and a weekly-returns matrix at data/cache/returns_weekly.parquet (the EDA notebook's pivot of `ret_d`
resampled to weekly is sufficient).

Run from the project root:
    python src/model_advanced.py
"""
from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, ndcg_score
from sklearn.preprocessing import StandardScaler

import lightgbm as lgb

try:
    from pypfopt import EfficientFrontier, expected_returns, risk_models
except ImportError:  # pragma: no cover - documented in data/README.md
    EfficientFrontier = None
    expected_returns = None
    risk_models = None


warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "cache"
DELIVERABLES = PROJECT_ROOT / "deliverables"
EMBED_DIM = 16
WINDOW_WEEKS = 208  # 4 years of weekly returns
SECTOR_CAP = 0.30


# ----------------------------- user model ------------------------------------ #


@dataclass
class UserProfile:
    years_experience: int
    horizon_years: int
    drawdown_tolerance: float
    monthly_budget_eur: float
    sector_preferences: list[str]

    def risk_score(self) -> float:
        s = 0.0
        s += min(self.years_experience, 20) / 20 * 0.25
        s += min(self.horizon_years, 20) / 20 * 0.25
        s += min(self.drawdown_tolerance, 0.50) / 0.50 * 0.40
        s += 0.10 if self.monthly_budget_eur >= 500 else 0.05
        return float(np.clip(s, 0.0, 1.0))


# ----------------------------- autoencoder ----------------------------------- #


class ReturnAutoencoder(nn.Module):
    """Compress a length-WINDOW_WEEKS weekly-return surface into EMBED_DIM."""

    def __init__(self, input_dim: int = WINDOW_WEEKS, embed_dim: int = EMBED_DIM):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, embed_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, 64), nn.ReLU(),
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, input_dim),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return z, x_hat


def train_autoencoder(weekly_returns: pd.DataFrame, n_epochs: int = 80, lr: float = 1e-3, device: str = "cpu") -> tuple[nn.Module, np.ndarray, list[str]]:
    """Train the autoencoder; return the model, the embedding matrix, and the row-aligned ticker order."""
    rec = weekly_returns.tail(WINDOW_WEEKS).T.dropna()  # rows = tickers, cols = weeks
    tickers = rec.index.tolist()
    X = rec.values.astype(np.float32)
    Xs = StandardScaler().fit_transform(X)

    model = ReturnAutoencoder(input_dim=Xs.shape[1], embed_dim=EMBED_DIM).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    ds = TensorDataset(torch.from_numpy(Xs).float())
    dl = DataLoader(ds, batch_size=64, shuffle=True)
    for epoch in range(n_epochs):
        running = 0.0
        for (batch,) in dl:
            batch = batch.to(device)
            z, x_hat = model(batch)
            loss = loss_fn(x_hat, batch)
            opt.zero_grad(); loss.backward(); opt.step()
            running += loss.item()
        if (epoch + 1) % 10 == 0:
            print(f"[advanced] AE epoch {epoch + 1}/{n_epochs} loss = {running / max(1, len(dl)):.4f}")

    model.eval()
    with torch.no_grad():
        z_all, _ = model(torch.from_numpy(Xs).float().to(device))
    return model, z_all.cpu().numpy(), tickers


# ----------------------------- mean-variance per cluster --------------------- #


def cluster_max_sharpe_weights(returns_for_cluster: pd.DataFrame) -> dict[str, float]:
    """Use PyPortfolioOpt to build a max-Sharpe portfolio for one cluster's tickers."""
    if EfficientFrontier is None:
        # Equal-weight fallback if PyPortfolioOpt is not installed.
        n = returns_for_cluster.shape[1]
        return dict(zip(returns_for_cluster.columns.tolist(), [1 / n] * n))
    mu = expected_returns.mean_historical_return(returns_for_cluster, returns_data=True, frequency=52)
    S = risk_models.sample_cov(returns_for_cluster, returns_data=True, frequency=52)
    ef = EfficientFrontier(mu, S, weight_bounds=(0, 0.30))
    try:
        ef.max_sharpe(risk_free_rate=0.04)
        return ef.clean_weights()
    except Exception as exc:
        print(f"[advanced] max-Sharpe failed ({exc}); falling back to min-vol")
        ef.min_volatility()
        return ef.clean_weights()


# ----------------------------- LightGBM lambdarank --------------------------- #


def build_ranker_dataset(features: pd.DataFrame, user_risk: float) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """One row per ticker; relevance label = quintile rank of fwd_return_12m within the cluster.

    The user's risk score is broadcast onto every row (the same ranker is reused across users at inference;
    the risk score acts as a context feature, in line with personalised-LTR practice).
    """
    df = features.dropna(subset=["fwd_return_12m"]).copy()
    df["user_risk"] = user_risk
    df["rel"] = (
        df.groupby("cluster")["fwd_return_12m"].rank(pct=True).mul(5).round().clip(0, 5).astype(int)
    )
    feat_cols = [
        "ann_return", "ann_vol", "sharpe", "beta", "max_drawdown",
        "user_risk",
    ]
    if "momentum_6m" in df.columns:
        feat_cols.append("momentum_6m")
    if "realized_vol_60d" in df.columns:
        feat_cols.append("realized_vol_60d")
    df = df.sort_values("cluster")
    group = df.groupby("cluster").size().to_numpy()
    return df[feat_cols + ["rel", "cluster"]], df["rel"].to_numpy(), group


def fit_ranker(df: pd.DataFrame, group: np.ndarray) -> lgb.LGBMRanker:
    feat_cols = [c for c in df.columns if c not in ("rel", "cluster")]
    model = lgb.LGBMRanker(
        objective="lambdarank",
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=5,
        random_state=42,
    )
    model.fit(df[feat_cols], df["rel"], group=group)
    return model


def evaluate_ranker(model: lgb.LGBMRanker, df: pd.DataFrame) -> dict:
    feat_cols = [c for c in df.columns if c not in ("rel", "cluster")]
    df = df.copy()
    df["score"] = model.predict(df[feat_cols])
    ndcgs = []
    for c, sub in df.groupby("cluster"):
        if len(sub) < 3:
            continue
        ndcgs.append(ndcg_score([sub["rel"].to_numpy()], [sub["score"].to_numpy()], k=min(5, len(sub))))
    return {"ndcg_at_5_mean": float(np.mean(ndcgs)) if ndcgs else None, "n_clusters": int(df["cluster"].nunique())}


# ----------------------------- recommend ------------------------------------- #


def recommend(
    features_with_cluster: pd.DataFrame,
    cluster_weights: dict[int, dict[str, float]],
    ranker: lgb.LGBMRanker,
    profile: UserProfile,
    n_total: int = 8,
) -> pd.DataFrame:
    feat_cols = [c for c in features_with_cluster.columns if c not in ("sector", "exchange", "cluster", "name", "fwd_return_12m")]
    feat_cols = [c for c in feat_cols if features_with_cluster[c].dtype != object]
    df = features_with_cluster.copy()
    df["user_risk"] = profile.risk_score()
    df["score"] = ranker.predict(df[[*feat_cols, "user_risk"]])

    # Pick clusters by user risk: simple low/mid/high mapping using cluster-level avg vol
    avg_vol_per_cluster = df.groupby("cluster")["ann_vol"].mean().sort_values()
    if profile.risk_score() < 0.33:
        picks = avg_vol_per_cluster.head(1).index.tolist()
    elif profile.risk_score() < 0.66:
        picks = avg_vol_per_cluster.head(max(2, len(avg_vol_per_cluster) // 2)).index.tolist()
    else:
        picks = avg_vol_per_cluster.tail(max(2, len(avg_vol_per_cluster) // 2)).index.tolist()

    pool = df[df["cluster"].isin(picks)].sort_values("score", ascending=False)
    if profile.sector_preferences:
        m = pool["sector"].isin(profile.sector_preferences)
        pool = pd.concat([pool[m], pool[~m]])

    selected, sector_count, max_per_sector = [], {}, max(1, int(np.floor(n_total * SECTOR_CAP)))
    for ticker, row in pool.iterrows():
        if sector_count.get(row["sector"], 0) >= max_per_sector:
            continue
        selected.append(ticker)
        sector_count[row["sector"]] = sector_count.get(row["sector"], 0) + 1
        if len(selected) == n_total:
            break

    return pool.loc[selected, ["sector", "cluster", "score", "ann_return", "ann_vol", "sharpe", "beta", "max_drawdown"]]


# ----------------------------- entry point ----------------------------------- #


def main() -> None:
    feat_path = DATA_DIR / "features.parquet"
    weekly_path = DATA_DIR / "returns_weekly.parquet"
    if not feat_path.exists() or not weekly_path.exists():
        raise FileNotFoundError(
            "Run notebooks/01_EDA.ipynb first; it produces features.parquet and returns_weekly.parquet."
        )

    features = pd.read_parquet(feat_path).dropna(subset=["fwd_return_12m"])
    weekly = pd.read_parquet(weekly_path)
    print(f"[advanced] features: {features.shape[0]} tickers, weekly: {weekly.shape}")

    # 1. Autoencoder embeddings
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ae, embeddings, ae_tickers = train_autoencoder(weekly, device=device)
    emb = pd.DataFrame(embeddings, index=ae_tickers, columns=[f"emb_{i}" for i in range(EMBED_DIM)])
    features = features.join(emb, how="inner")

    # 2. Cluster on embeddings + risk axes
    cluster_features = [c for c in features.columns if c.startswith("emb_")] + ["sharpe", "beta", "max_drawdown"]
    X = StandardScaler().fit_transform(features[cluster_features])
    best_k, best_score, best_km = None, -np.inf, None
    for k in range(3, 11):
        km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
        s = silhouette_score(X, km.labels_)
        if s > best_score:
            best_k, best_score, best_km = k, s, km
    features["cluster"] = best_km.labels_
    print(f"[advanced] best k = {best_k} silhouette = {best_score:.4f}")

    # 3. Per-cluster max-Sharpe weights
    cluster_weights: dict[int, dict[str, float]] = {}
    for c, group in features.groupby("cluster"):
        sub = weekly.loc[:, [t for t in group.index if t in weekly.columns]].dropna(how="all", axis=1)
        if sub.shape[1] < 2:
            continue
        cluster_weights[int(c)] = cluster_max_sharpe_weights(sub)

    # 4. Train the lambdarank reranker
    profile = UserProfile(
        years_experience=3, horizon_years=10, drawdown_tolerance=0.25,
        monthly_budget_eur=500, sector_preferences=["Information Technology", "Health Care"],
    )
    rd, y, group = build_ranker_dataset(features, profile.risk_score())
    ranker = fit_ranker(rd, group)
    metrics = evaluate_ranker(ranker, rd)
    print("[advanced] ranker metrics:", metrics)

    # 5. Recommend
    portfolio = recommend(features, cluster_weights, ranker, profile, n_total=8)
    print("\n[advanced] recommended portfolio for demo profile:")
    print(portfolio.round(3).to_string())

    DELIVERABLES.mkdir(parents=True, exist_ok=True)
    torch.save(ae.state_dict(), DELIVERABLES / "advanced_autoencoder.pt")
    joblib.dump({"kmeans": best_km, "k": best_k, "cluster_weights": cluster_weights}, DELIVERABLES / "advanced_cluster.joblib")
    ranker.booster_.save_model(str(DELIVERABLES / "advanced_ranker.txt"))
    portfolio.to_csv(DELIVERABLES / "advanced_demo_portfolio.csv")
    out_metrics = {
        "k": best_k,
        "silhouette": float(best_score),
        "embed_dim": EMBED_DIM,
        "ranker_ndcg_at_5": metrics["ndcg_at_5_mean"],
        "n_tickers": int(features.shape[0]),
        "demo_user_risk_score": profile.risk_score(),
    }
    (DELIVERABLES / "advanced_metrics.json").write_text(json.dumps(out_metrics, indent=2))
    print(f"\n[advanced] saved -> {DELIVERABLES / 'advanced_autoencoder.pt'}")
    print(f"[advanced] saved -> {DELIVERABLES / 'advanced_cluster.joblib'}")
    print(f"[advanced] saved -> {DELIVERABLES / 'advanced_ranker.txt'}")
    print(f"[advanced] saved -> {DELIVERABLES / 'advanced_metrics.json'}")


if __name__ == "__main__":
    main()
