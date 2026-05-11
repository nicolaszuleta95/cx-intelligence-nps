"""
Customer segmentation using K-Means on NLP-derived features.

Features used: FinBERT sentiment score, dominant LDA topic (encoded),
simulated NPS score, and product category (label-encoded).

Cluster naming follows CX best practices — each segment receives a
human-readable label that guides action rather than just describing data.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import LabelEncoder, StandardScaler


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

KMEANS_RANDOM_STATE = 42
KMEANS_N_INIT = 10
FEATURE_COLS = ["finbert_score", "dominant_topic", "nps_score", "product_encoded"]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_feature_matrix(df: pd.DataFrame) -> tuple[np.ndarray, LabelEncoder]:
    """Construct and scale the feature matrix for K-Means clustering.

    Features:
    - finbert_score: FinBERT confidence score (continuous, 0–1)
    - dominant_topic: LDA topic index (will be treated as ordinal)
    - nps_score: Simulated NPS score (0–10)
    - product_encoded: Label-encoded banking product category

    Args:
        df: DataFrame containing the four feature columns (or a 'product' column
            that will be encoded on the fly).

    Returns:
        Tuple of (scaled feature matrix as np.ndarray, fitted LabelEncoder for product).
    """
    df = df.copy()

    # Encode product if not already done
    le = LabelEncoder()
    if "product_encoded" not in df.columns:
        df["product_encoded"] = le.fit_transform(df["product"].astype(str))
    else:
        le.fit(df["product"].astype(str))

    # Select and scale features
    X = df[FEATURE_COLS].fillna(0).values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, le


def find_optimal_k(
    X: np.ndarray, k_range: range = range(2, 9)
) -> dict[str, dict]:
    """Evaluate K-Means across a range of k values.

    Args:
        X: Scaled feature matrix from build_feature_matrix().
        k_range: Range of k values to evaluate.

    Returns:
        Dict with two keys:
        - 'elbow': {k → inertia (WCSS)}
        - 'silhouette': {k → silhouette score}
    """
    elbow: dict[int, float] = {}
    silhouette: dict[int, float] = {}

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=KMEANS_RANDOM_STATE, n_init=KMEANS_N_INIT)
        labels = km.fit_predict(X)
        elbow[k] = round(km.inertia_, 2)
        if k > 1:
            silhouette[k] = round(silhouette_score(X, labels), 4)

    return {"elbow": elbow, "silhouette": silhouette}


def train_kmeans(X: np.ndarray, n_clusters: int) -> KMeans:
    """Fit a K-Means model with the given number of clusters.

    Args:
        X: Scaled feature matrix.
        n_clusters: Number of clusters (k).

    Returns:
        Fitted KMeans instance.
    """
    km = KMeans(
        n_clusters=n_clusters,
        random_state=KMEANS_RANDOM_STATE,
        n_init=KMEANS_N_INIT,
    )
    km.fit(X)
    return km


def assign_clusters(df: pd.DataFrame, kmeans_model: KMeans, X: np.ndarray) -> pd.DataFrame:
    """Add a 'cluster' column with cluster assignments.

    Args:
        df: Original DataFrame (aligned with X).
        kmeans_model: Fitted KMeans model.
        X: Scaled feature matrix used for prediction.

    Returns:
        DataFrame with a new 'cluster' integer column.
    """
    df = df.copy()
    df["cluster"] = kmeans_model.predict(X)
    return df


def build_cluster_profiles(df: pd.DataFrame) -> pd.DataFrame:
    """Compute aggregate statistics for each cluster.

    Args:
        df: DataFrame with 'cluster', 'nps_score', 'finbert_label',
            'finbert_score', and 'product' columns.

    Returns:
        DataFrame with one row per cluster and columns:
        cluster, n_complaints, avg_nps, dominant_sentiment,
        avg_sentiment_score, top_product, pct_detractors.
    """
    records = []
    for cluster_id, gdf in df.groupby("cluster"):
        avg_nps = round(gdf["nps_score"].mean(), 2)
        dominant_sentiment = gdf["finbert_label"].mode().iloc[0] if len(gdf) > 0 else "unknown"
        avg_score = round(gdf["finbert_score"].mean(), 4)
        top_product = gdf["product"].mode().iloc[0] if len(gdf) > 0 else "unknown"
        pct_detractors = round(
            (gdf["nps_segment"] == "Detractor").sum() / len(gdf) * 100, 1
        )
        records.append(
            {
                "cluster": cluster_id,
                "n_complaints": len(gdf),
                "avg_nps": avg_nps,
                "dominant_sentiment": dominant_sentiment,
                "avg_sentiment_score": avg_score,
                "top_product": top_product,
                "pct_detractors": pct_detractors,
            }
        )
    return pd.DataFrame(records).sort_values("avg_nps")


def name_clusters(profiles_df: pd.DataFrame) -> dict[int, str]:
    """Assign CX-actionable names to clusters based on their profiles.

    Naming heuristic:
    - Very low NPS + high detractor pct → "Critical Risk"
    - Low NPS + negative dominant sentiment → "Silent Dissatisfied"
    - Medium NPS → "Neutral Observers"
    - High NPS + positive sentiment → "Promoter Candidates"
    - Highest NPS → "Active Promoters"

    Args:
        profiles_df: Output from build_cluster_profiles().

    Returns:
        Dict mapping cluster integer → CX segment name string.
    """
    sorted_df = profiles_df.sort_values("avg_nps").reset_index(drop=True)
    n = len(sorted_df)

    cx_names = {
        0: "Critical Risk",
        1: "Silent Dissatisfied",
        2: "Neutral Observers",
        3: "Promoter Candidates",
        4: "Active Promoters",
    }
    # Fallback for unexpected cluster counts
    fallback = [
        "Critical Risk", "Dissatisfied", "Neutral", "Satisfied", "Promoter"
    ]

    cluster_to_name: dict[int, str] = {}
    for rank, row in sorted_df.iterrows():
        name = cx_names.get(rank, fallback[min(rank, len(fallback) - 1)])
        cluster_to_name[int(row["cluster"])] = name

    return cluster_to_name


def save_kmeans_model(kmeans_model: KMeans, path: str | Path) -> None:
    """Persist the fitted K-Means model with joblib.

    Args:
        kmeans_model: Fitted KMeans instance.
        path: Output path (e.g. models/kmeans_model.joblib).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(kmeans_model, path)
    print(f"K-Means model saved → {path}")


def load_kmeans_model(path: str | Path) -> KMeans:
    """Load a joblib-serialized K-Means model.

    Args:
        path: Path to the .joblib file.

    Returns:
        Fitted KMeans instance.
    """
    return joblib.load(Path(path))
