"""
Data loading, filtering, and cleaning for CFPB Consumer Complaints dataset.

Handles the full pipeline from raw CSV to processed banking-specific subset
ready for NLP analysis.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BANKING_PRODUCTS = [
    "Checking or savings account",
    "Credit card or prepaid card",
    "Mortgage",
    "Personal loan",
    "Student loan",
]

PRODUCT_LABEL_MAP = {
    "Bank account or service": "Checking or savings account",
    "Checking or savings account": "Checking or savings account",
    "Credit card": "Credit card or prepaid card",
    "Prepaid card": "Credit card or prepaid card",
    "Credit card or prepaid card": "Credit card or prepaid card",
    "Mortgage": "Mortgage",
    "Consumer Loan": "Personal loan",
    "Personal loan": "Personal loan",
    "Student loan": "Student loan",
}

RAW_FILENAME = "consumer_complaints.csv"
PROCESSED_FILENAME = "banking_complaints.csv"

# Maps CFPB company_response_to_consumer → 3-class resolution sentiment label.
# Used by the supervised fine-tuning approach in notebook 02.
RESOLUTION_LABEL_MAP: dict[str, str | None] = {
    "Closed with monetary relief":     "positive",  # company refunded / compensated
    "Closed with non-monetary relief": "neutral",   # company fixed issue, no financial comp
    "Closed with explanation":         "negative",  # company explained only — no action
    "Untimely response":               "negative",  # company failed to respond on time
    "Closed without relief":           "negative",  # explicit denial
    "Closed":                          "negative",  # generic close — no stated action
    "In progress":                     None,        # exclude — outcome unknown
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def map_resolution_label(response: str) -> str | None:
    """Map a CFPB company response string to a resolution sentiment label.

    Args:
        response: Value of the 'company_response_to_consumer' column.

    Returns:
        'positive', 'neutral', or 'negative'. None for unknown / in-progress
        entries that should be excluded from supervised training.
    """
    return RESOLUTION_LABEL_MAP.get(str(response).strip(), None)


def load_raw_data(path: str | Path) -> pd.DataFrame:
    """Load the raw CFPB complaints CSV.

    Args:
        path: Path to the raw CSV file (consumer_complaints.csv).

    Returns:
        DataFrame with all columns preserved as-is.

    Raises:
        FileNotFoundError: If the CSV file does not exist at the given path.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Raw data not found at '{path}'. "
            "Download from https://www.kaggle.com/datasets/cfpb/us-consumer-finance-complaints "
            "and place it in data/raw/."
        )
    df = pd.read_csv(path, low_memory=False)
    df.columns = [c.strip().lower().replace(" ", "_").replace("?", "") for c in df.columns]
    return df


def filter_banking_products(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only complaints for the five banking products in scope.

    Args:
        df: Raw complaints DataFrame (must contain a 'product' column).

    Returns:
        Filtered DataFrame with ~40,000–60,000 rows.
    """
    df = df.copy()
    df["product_raw"] = df["product"]
    df["product"] = df["product"].map(PRODUCT_LABEL_MAP).fillna(df["product"])
    mask = df["product"].isin(BANKING_PRODUCTS)
    return df[mask].copy()


def clean_text(text: str) -> str:
    """Normalize a single complaint narrative string.

    Steps applied:
    - Lowercase
    - Remove CFPB redaction tokens (e.g. "XXXX", "XX/XX/XXXX")
    - Strip non-alphabetic characters except spaces
    - Collapse multiple spaces

    Args:
        text: Raw complaint narrative string.

    Returns:
        Cleaned, lowercase string. Empty string if input is null/empty.
    """
    if not isinstance(text, str) or not text.strip():
        return ""
    text = text.lower()
    # Remove CFPB redaction placeholders
    text = re.sub(r"\bxx+\b", " ", text)
    text = re.sub(r"\bxx/xx(?:/xxxx)?\b", " ", text)
    # Remove non-alphabetic characters except spaces
    text = re.sub(r"[^a-z\s]", " ", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def remove_nulls(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows where the complaint narrative is null or empty.

    Args:
        df: DataFrame that must contain a 'consumer_complaint_narrative' column.

    Returns:
        DataFrame with null-narrative rows removed.
    """
    df = df.dropna(subset=["consumer_complaint_narrative"]).copy()
    df = df[df["consumer_complaint_narrative"].str.strip() != ""].copy()
    return df


def save_processed(df: pd.DataFrame, path: str | Path) -> None:
    """Persist the processed DataFrame to CSV.

    Args:
        df: Processed complaints DataFrame.
        path: Destination file path (e.g. data/processed/banking_complaints.csv).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Saved {len(df):,} rows → {path}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def run_pipeline(
    raw_path: str | Path | None = None,
    output_path: str | Path | None = None,
) -> pd.DataFrame:
    """Execute the full data-processing pipeline.

    Args:
        raw_path: Path to the raw CSV. Defaults to data/raw/consumer_complaints.csv
                  relative to this file's project root.
        output_path: Destination for processed CSV. Defaults to
                     data/processed/banking_complaints.csv.

    Returns:
        Processed DataFrame ready for NPS simulation and NLP analysis.
    """
    root = Path(__file__).resolve().parents[1]
    raw_path = Path(raw_path) if raw_path else root / "data" / "raw" / RAW_FILENAME
    output_path = (
        Path(output_path) if output_path else root / "data" / "processed" / PROCESSED_FILENAME
    )

    print("Loading raw data…")
    df = load_raw_data(raw_path)
    print(f"  Loaded {len(df):,} total complaints")

    print("Filtering to banking products…")
    df = filter_banking_products(df)
    print(f"  Retained {len(df):,} banking complaints")

    print("Removing nulls in complaint narrative…")
    df = remove_nulls(df)
    print(f"  After null removal: {len(df):,} rows")

    print("Cleaning text…")
    df["text_clean"] = df["consumer_complaint_narrative"].apply(clean_text)
    # Drop rows where cleaning produced an empty string
    df = df[df["text_clean"] != ""].copy()
    print(f"  After text cleaning: {len(df):,} rows")

    save_processed(df, output_path)
    return df


if __name__ == "__main__":
    run_pipeline()
