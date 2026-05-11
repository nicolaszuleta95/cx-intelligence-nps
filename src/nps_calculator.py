"""
NPS simulation, classification, and aggregation for CFPB complaints data.

Because the CFPB dataset does not include satisfaction scores, NPS is
constructed as a proxy using response timeliness and complaint resolution
outcome — a deliberate methodological choice documented in the README and
Model Card.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NPS_NOISE_STD = 1.0        # Gaussian noise magnitude added for realism
PROMOTER_THRESHOLD = 9     # NPS score >= this → Promoter
PASSIVE_THRESHOLD = 7      # score 7–8 → Passive; below → Detractor

# Company response values that indicate a positive resolution
POSITIVE_RESOLUTIONS = {
    "Closed with monetary relief",
    "Closed with non-monetary relief",
    "Closed with relief",
}

EXPLANATION_RESOLUTIONS = {
    "Closed with explanation",
    "Closed",
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def simulate_nps_score(row: pd.Series) -> int:
    """Compute a synthetic NPS score (0–10) from CFPB resolution metadata.

    Logic:
    - Timely response + positive resolution  → Promoter range (8–10)
    - Timely response + explanation only     → Passive range  (6–8)
    - No timely response OR no resolution    → Detractor range (0–6)

    Gaussian noise (σ=1) is added for realism, then the result is
    clipped to [0, 10] and rounded to the nearest integer.

    Args:
        row: A single row from the processed complaints DataFrame.
             Must contain 'timely_response' and 'company_response_to_consumer'.

    Returns:
        Integer NPS score in the range [0, 10].
    """
    rng = np.random.default_rng()

    timely = str(row.get("timely_response", "")).strip().lower() == "yes"
    resolution = str(row.get("company_response_to_consumer", "")).strip()

    if timely and resolution in POSITIVE_RESOLUTIONS:
        base_score = 9.0  # Promoter territory
    elif timely and resolution in EXPLANATION_RESOLUTIONS:
        base_score = 7.0  # Passive territory
    else:
        base_score = 3.5  # Detractor territory

    noisy_score = base_score + rng.normal(0, NPS_NOISE_STD)
    return int(np.clip(round(noisy_score), 0, 10))


def classify_nps(score: int) -> str:
    """Map a 0–10 NPS score to its segment label.

    Args:
        score: Integer NPS score in [0, 10].

    Returns:
        'Promoter' if score >= 9, 'Passive' if 7–8, 'Detractor' if <= 6.
    """
    if score >= PROMOTER_THRESHOLD:
        return "Promoter"
    if score >= PASSIVE_THRESHOLD:
        return "Passive"
    return "Detractor"


def calculate_nps(df: pd.DataFrame) -> float:
    """Compute the Net Promoter Score for the full dataset.

    Formula: NPS = (n_promoters / total − n_detractors / total) × 100

    Args:
        df: DataFrame that must contain a 'nps_segment' column with values
            'Promoter', 'Passive', 'Detractor'.

    Returns:
        NPS as a float in the range [−100, 100].
    """
    total = len(df)
    if total == 0:
        return 0.0

    n_promoters = (df["nps_segment"] == "Promoter").sum()
    n_detractors = (df["nps_segment"] == "Detractor").sum()
    return round((n_promoters / total - n_detractors / total) * 100, 2)


def nps_by_group(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """Calculate NPS broken down by a categorical column.

    Args:
        df: DataFrame with 'nps_segment' and the specified group column.
        group_col: Column to group by (e.g. 'product', 'state').

    Returns:
        DataFrame indexed by group_col with columns:
        n_total, n_promoters, n_detractors, pct_promoters,
        pct_detractors, nps.
    """
    records = []
    for group, gdf in df.groupby(group_col):
        n_total = len(gdf)
        n_promoters = (gdf["nps_segment"] == "Promoter").sum()
        n_detractors = (gdf["nps_segment"] == "Detractor").sum()
        nps_val = round(
            (n_promoters / n_total - n_detractors / n_total) * 100, 2
        )
        records.append(
            {
                group_col: group,
                "n_total": n_total,
                "n_promoters": int(n_promoters),
                "n_detractors": int(n_detractors),
                "pct_promoters": round(n_promoters / n_total * 100, 1),
                "pct_detractors": round(n_detractors / n_total * 100, 1),
                "nps": nps_val,
            }
        )
    return pd.DataFrame(records).sort_values("nps", ascending=False)


def nps_trend(
    df: pd.DataFrame,
    date_col: str = "date_received",
    freq: str = "M",
) -> pd.DataFrame:
    """Compute NPS over time, grouped by calendar period.

    Args:
        df: DataFrame with 'nps_segment' and a date column.
        date_col: Name of the date column to group by.
        freq: Pandas offset alias — 'M' (monthly), 'Q' (quarterly), etc.

    Returns:
        DataFrame with columns: period, n_total, nps.
        Sorted chronologically.
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])
    df["period"] = df[date_col].dt.to_period(freq)

    records = []
    for period, gdf in df.groupby("period"):
        n_total = len(gdf)
        n_promoters = (gdf["nps_segment"] == "Promoter").sum()
        n_detractors = (gdf["nps_segment"] == "Detractor").sum()
        nps_val = round(
            (n_promoters / n_total - n_detractors / n_total) * 100, 2
        )
        records.append({"period": str(period), "n_total": n_total, "nps": nps_val})

    return pd.DataFrame(records)
