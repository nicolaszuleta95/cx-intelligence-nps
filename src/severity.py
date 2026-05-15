"""
Complaint Severity Predictor — structured feature engineering and multi-class classification.

Replaces sentiment.py which was discarded after empirical validation:
6 sentiment models tested (VADER, FinBERT, RoBERTa, Zero-Shot DeBERTa, DistilBERT
v1 and v2) all achieved 33–41% accuracy (near random chance for 3 classes).

Root cause: classifying positive/negative/neutral in a corpus of bank complaints
has no semantic validity — all complaints are inherently negative by definition.

This module defines complaint severity (LOW / MEDIUM / HIGH) from structured CFPB
variables — a target that naturally exists in the data and directly maps to CX
action priorities.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder

try:
    from xgboost import XGBClassifier  # noqa: F401
    _XGBOOST_AVAILABLE = True
except ImportError:
    _XGBOOST_AVAILABLE = False

try:
    from lightgbm import LGBMClassifier  # noqa: F401
    _LIGHTGBM_AVAILABLE = True
except ImportError:
    _LIGHTGBM_AVAILABLE = False


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Company response types — ordered by resolution favorability
RESPONSE_FAVORABILITY: dict[str, int] = {
    "Closed with monetary relief": 5,
    "Closed with non-monetary relief": 4,
    "Closed with explanation": 3,
    "Closed without relief": 2,
    "In progress": 1,
    "Unk": 0,
}

# Product severity weight — more complex products generate higher-severity complaints
PRODUCT_SEVERITY_WEIGHT: dict[str, int] = {
    "Mortgage": 4,
    "Student loan": 3,
    "Credit card or prepaid card": 2,
    "Checking or savings account": 2,
    "Personal loan": 1,
}

SEVERITY_LABELS = ["LOW", "MEDIUM", "HIGH"]

# CFPB column name constants
NARRATIVE_COL = "consumer_complaint_narrative"
RESPONSE_COL = "company_response_to_consumer"
PRODUCT_COL = "product"
TIMELY_COL = "timely_response"
CHANNEL_COL = "submitted_via"
DATE_RECEIVED_COL = "date_received"
DATE_SENT_COL = "date_sent_to_company"
CONSUMER_ID_COL = "complaint_id"

# Feature columns used by SeverityPredictor
FEATURE_COLS = [
    "timely_response_binary",
    "response_type_encoded",
    "product_encoded",
    "complaint_length",
    "has_narrative",
    "submission_channel_encoded",
    "days_to_resolution",
    "multi_complaint_flag",
]


# ---------------------------------------------------------------------------
# Public API — Feature Engineering
# ---------------------------------------------------------------------------


def build_severity_features(df: pd.DataFrame) -> pd.DataFrame:
    """Construct structured features for complaint severity prediction.

    Operates on the CFPB processed DataFrame and adds the following columns:

    - timely_response_binary: 1 if 'Timely response?' == 'Yes', else 0.
    - response_type_encoded: Ordinal encoding of company resolution favorability
        (5 = monetary relief → 0 = unknown/in-progress).
    - product_encoded: Ordinal encoding of banking product by typical complaint
        severity weight (Mortgage=4 → Personal loan=1).
    - complaint_length: Word count of consumer_complaint_narrative (0 if absent).
    - has_narrative: 1 if narrative is non-empty, else 0.
    - submission_channel_encoded: Label-encoded 'Submitted via' channel.
    - days_to_resolution: Days between date_received and date_sent_to_company
        (NaN if either date is missing or unparseable).
    - multi_complaint_flag: 1 if the same complaint_id appears more than once
        (proxy for repeated contact / chronic dissatisfier).

    Args:
        df: Cleaned CFPB banking complaints DataFrame.

    Returns:
        DataFrame with all original columns plus the engineered feature columns.
    """
    df = df.copy()

    # --- timely_response_binary ---
    if TIMELY_COL in df.columns:
        df["timely_response_binary"] = (
            df[TIMELY_COL].astype(str).str.strip().str.lower() == "yes"
        ).astype(int)
    else:
        df["timely_response_binary"] = 0

    # --- response_type_encoded ---
    if RESPONSE_COL in df.columns:
        df["response_type_encoded"] = (
            df[RESPONSE_COL]
            .astype(str)
            .str.strip()
            .map(lambda x: RESPONSE_FAVORABILITY.get(x, 0))
        )
    else:
        df["response_type_encoded"] = 0

    # --- product_encoded ---
    if PRODUCT_COL in df.columns:
        df["product_encoded"] = (
            df[PRODUCT_COL]
            .astype(str)
            .str.strip()
            .map(lambda x: PRODUCT_SEVERITY_WEIGHT.get(x, 1))
        )
    else:
        df["product_encoded"] = 1

    # --- complaint_length + has_narrative ---
    if NARRATIVE_COL in df.columns:
        narrative = df[NARRATIVE_COL].fillna("").astype(str)
        df["complaint_length"] = narrative.str.split().str.len()
        df["has_narrative"] = (df["complaint_length"] > 0).astype(int)
    else:
        df["complaint_length"] = 0
        df["has_narrative"] = 0

    # --- submission_channel_encoded ---
    if CHANNEL_COL in df.columns:
        le = LabelEncoder()
        df["submission_channel_encoded"] = le.fit_transform(
            df[CHANNEL_COL].fillna("Unknown").astype(str)
        )
    else:
        df["submission_channel_encoded"] = 0

    # --- days_to_resolution ---
    df["days_to_resolution"] = np.nan
    if DATE_RECEIVED_COL in df.columns and DATE_SENT_COL in df.columns:
        try:
            received = pd.to_datetime(df[DATE_RECEIVED_COL], errors="coerce")
            sent = pd.to_datetime(df[DATE_SENT_COL], errors="coerce")
            df["days_to_resolution"] = (sent - received).dt.days
        except Exception:
            pass

    # --- multi_complaint_flag ---
    df["multi_complaint_flag"] = 0
    if CONSUMER_ID_COL in df.columns:
        complaint_counts = df[CONSUMER_ID_COL].value_counts()
        df["multi_complaint_flag"] = (
            df[CONSUMER_ID_COL].map(complaint_counts) > 1
        ).astype(int)

    return df


def define_severity_label(df: pd.DataFrame) -> pd.DataFrame:
    """Define the complaint severity target variable (LOW / MEDIUM / HIGH).

    Severity is determined by a rule-based combination of structured CFPB
    variables, reflecting the urgency of CX action required:

    Severity rules (applied in order — first match wins):
    ─────────────────────────────────────────────────────────────────────────────
    HIGH (requires immediate CX escalation):
        • timely_response_binary == 0 (response NOT timely)
        • OR response_type_encoded <= 1 (no resolution or in-progress)
        • AND complaint_length > 150 words (detailed, protracted complaint)

    LOW (batch processing / monitor):
        • timely_response_binary == 1 (response was timely)
        • AND response_type_encoded >= 4 (monetary or non-monetary relief)
        • AND complaint_length <= 150 words

    MEDIUM (follow-up within 24h):
        • Everything else — partial resolution, explanation only, or medium length

    NPS-based refinement (if nps_score column is available):
        • nps_score <= 4 (Detractor) can upgrade MEDIUM → HIGH
        • nps_score >= 8 (Promoter) can downgrade MEDIUM → LOW

    The resulting column 'severity' contains: 'LOW', 'MEDIUM', or 'HIGH'.
    'severity_encoded' maps these to integers: 0=LOW, 1=MEDIUM, 2=HIGH.

    Args:
        df: DataFrame with feature columns from build_severity_features().
            Must contain: timely_response_binary, response_type_encoded,
            complaint_length. Optionally: nps_score.

    Returns:
        DataFrame with new 'severity' (str) and 'severity_encoded' (int) columns.
    """
    df = df.copy()

    required = ["timely_response_binary", "response_type_encoded", "complaint_length"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing columns — run build_severity_features() first: {missing}"
        )

    # Vectorized rule masks
    is_untimely = df["timely_response_binary"] == 0
    is_unresolved = df["response_type_encoded"] <= 1
    is_long = df["complaint_length"] > 150
    is_favorable = df["response_type_encoded"] >= 4
    is_timely = df["timely_response_binary"] == 1

    high_mask = (is_untimely | is_unresolved) & is_long
    low_mask = is_timely & is_favorable & ~is_long

    severity = pd.Series("MEDIUM", index=df.index)
    severity[high_mask] = "HIGH"
    severity[low_mask] = "LOW"

    # NPS-based refinement (optional)
    if "nps_score" in df.columns:
        nps = df["nps_score"]
        severity[(severity == "MEDIUM") & (nps <= 4)] = "HIGH"
        severity[(severity == "MEDIUM") & (nps >= 8)] = "LOW"

    df["severity"] = severity

    label_map = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
    df["severity_encoded"] = df["severity"].map(label_map)

    return df


# ---------------------------------------------------------------------------
# SeverityPredictor Class
# ---------------------------------------------------------------------------


class SeverityPredictor:
    """Predict complaint severity (LOW / MEDIUM / HIGH) from structured CFPB features.

    Three models available for comparison:
    - 'logistic': Logistic Regression (interpretable baseline)
    - 'xgboost': XGBoost (primary model — consistent with Project 1: banking-churn-prediction)
    - 'lightgbm': LightGBM (alternative gradient boosting)

    The production model is XGBoost — aligns with existing portfolio and performs
    well on structured tabular banking data.

    Usage:
        predictor = SeverityPredictor(model_type='xgboost')
        predictor.fit(X_train, y_train)
        metrics = predictor.evaluate(X_test, y_test)
        predictor.save(ROOT / "models" / "severity_model.joblib")
    """

    VALID_MODELS = ("logistic", "xgboost", "lightgbm")

    def __init__(self, model_type: str = "xgboost") -> None:
        if model_type not in self.VALID_MODELS:
            raise ValueError(
                f"model_type must be one of {self.VALID_MODELS}, got '{model_type}'"
            )
        self.model_type = model_type
        self._model = self._build_model(model_type)
        self._is_fitted = False
        self._feature_names: list[str] = []

    def _build_model(self, model_type: str):
        if model_type == "logistic":
            return LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=42,
                multi_class="multinomial",
            )
        elif model_type == "xgboost":
            if not _XGBOOST_AVAILABLE:
                raise ImportError("xgboost not installed. Run: pip install xgboost")
            from xgboost import XGBClassifier
            return XGBClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric="mlogloss",
                random_state=42,
                verbosity=0,
            )
        elif model_type == "lightgbm":
            if not _LIGHTGBM_AVAILABLE:
                raise ImportError("lightgbm not installed. Run: pip install lightgbm")
            from lightgbm import LGBMClassifier
            return LGBMClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                class_weight="balanced",
                random_state=42,
                verbosity=-1,
            )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Train the model.

        Args:
            X: Feature DataFrame (from build_severity_features()).
            y: Target Series with severity labels ('LOW', 'MEDIUM', 'HIGH')
               or encoded integers (0, 1, 2).
        """
        self._feature_names = list(X.columns)
        X_arr = X.fillna(0).values
        y_enc = self._encode_labels(y)
        self._model.fit(X_arr, y_enc)
        self._is_fitted = True

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict severity as integer labels (0=LOW, 1=MEDIUM, 2=HIGH).

        Args:
            X: Feature DataFrame with the same columns as training data.

        Returns:
            Integer array of predictions.
        """
        self._check_fitted()
        return self._model.predict(X.fillna(0).values)

    def predict_labels(self, X: pd.DataFrame) -> np.ndarray:
        """Predict severity as string labels (LOW / MEDIUM / HIGH).

        Args:
            X: Feature DataFrame.

        Returns:
            String array of predictions.
        """
        preds = self.predict(X)
        inv_map = {0: "LOW", 1: "MEDIUM", 2: "HIGH"}
        return np.array([inv_map[p] for p in preds])

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities.

        Args:
            X: Feature DataFrame.

        Returns:
            Array of shape (n_samples, 3): columns are [P(LOW), P(MEDIUM), P(HIGH)].
        """
        self._check_fitted()
        return self._model.predict_proba(X.fillna(0).values)

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> dict:
        """Compute evaluation metrics on a held-out set.

        Args:
            X: Feature DataFrame.
            y: True labels (integer or string).

        Returns:
            Dict with: accuracy, f1_weighted, classification_report, confusion_matrix.
        """
        self._check_fitted()
        y_pred = self.predict(X)
        y_true = self._encode_labels(y)

        return {
            "accuracy": round(accuracy_score(y_true, y_pred), 4),
            "f1_weighted": round(
                f1_score(y_true, y_pred, average="weighted", zero_division=0), 4
            ),
            "classification_report": classification_report(
                y_true,
                y_pred,
                target_names=SEVERITY_LABELS,
                zero_division=0,
            ),
            "confusion_matrix": confusion_matrix(y_true, y_pred),
        }

    def cross_validate(
        self, X: pd.DataFrame, y: pd.Series, cv: int = 5
    ) -> dict[str, float]:
        """Run stratified k-fold cross-validation.

        Args:
            X: Feature DataFrame.
            y: Target Series.
            cv: Number of folds (default 5).

        Returns:
            Dict with mean and std of accuracy and f1_weighted across folds.
        """
        X_arr = X.fillna(0).values
        y_enc = self._encode_labels(y)

        acc_scores = cross_val_score(
            self._model, X_arr, y_enc, cv=cv, scoring="accuracy"
        )
        f1_scores = cross_val_score(
            self._model, X_arr, y_enc, cv=cv, scoring="f1_weighted"
        )
        return {
            "cv_accuracy_mean": round(acc_scores.mean(), 4),
            "cv_accuracy_std": round(acc_scores.std(), 4),
            "cv_f1_mean": round(f1_scores.mean(), 4),
            "cv_f1_std": round(f1_scores.std(), 4),
        }

    def get_feature_importance(self) -> pd.DataFrame:
        """Return feature importances as a sorted DataFrame.

        For tree-based models (XGBoost, LightGBM), uses the built-in
        feature_importances_ attribute. For Logistic Regression, uses the
        mean absolute coefficient across classes.

        Returns:
            DataFrame with columns: feature_name, importance, importance_pct
            — sorted by importance descending.
        """
        self._check_fitted()

        if hasattr(self._model, "feature_importances_"):
            importances = self._model.feature_importances_
        elif hasattr(self._model, "coef_"):
            importances = np.abs(self._model.coef_).mean(axis=0)
        else:
            raise RuntimeError(
                f"Model type '{self.model_type}' does not expose feature importances."
            )

        feature_names = self._feature_names if self._feature_names else FEATURE_COLS
        df = pd.DataFrame(
            {"feature_name": feature_names, "importance": importances}
        ).sort_values("importance", ascending=False)

        total = df["importance"].sum()
        df["importance_pct"] = (df["importance"] / total * 100).round(2) if total > 0 else 0.0
        return df.reset_index(drop=True)

    def save(self, path: str | Path) -> None:
        """Serialize the fitted model to disk with joblib.

        Args:
            path: Output path — typically models/severity_model.joblib.
        """
        self._check_fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"model": self._model, "feature_names": self._feature_names}
        joblib.dump(payload, path)
        print(f"SeverityPredictor saved → {path}")

    def load(self, path: str | Path) -> None:
        """Load a previously saved SeverityPredictor from disk.

        Args:
            path: Path to the .joblib file created by save().
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model not found at '{path}'")
        payload = joblib.load(path)
        self._model = payload["model"]
        self._feature_names = payload["feature_names"]
        self._is_fitted = True
        print(f"SeverityPredictor loaded ← {path}")

    def save_feature_list(self, path: str | Path) -> None:
        """Export the ordered feature list to JSON for inference reproducibility.

        The feature list ensures that downstream inference always passes columns
        in the correct order.

        Args:
            path: Output path — typically models/severity_features.json.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self._feature_names, f, indent=2)
        print(f"Feature list saved → {path}")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _encode_labels(self, y: pd.Series) -> pd.Series:
        """Encode string severity labels to integers if needed."""
        if y.dtype == object:
            label_map = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
            return y.map(label_map)
        return y

    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError(
                "Model has not been fitted yet. Call fit() first."
            )
