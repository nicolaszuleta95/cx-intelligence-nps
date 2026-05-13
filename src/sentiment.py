"""
Sentiment analysis pipeline: VADER, FinBERT, RoBERTa, and Zero-Shot NLI.

All four models share a unified interface via SentimentPipeline so downstream
code — notebooks and Streamlit — never needs to know which model is active.

Model summary:
- vader      : Rule-based lexicon (social-media tuned). Fast, no GPU.
- finbert    : ProsusAI/finbert — BERT pre-trained on financial NEWS text.
- roberta    : cardiffnlp/twitter-roberta-base-sentiment-latest — RoBERTa
               fine-tuned on 124M tweets for 3-class sentiment.
- zeroshot   : cross-encoder/nli-deberta-v3-small — NLI model used for
               zero-shot classification with domain-adapted CX labels.
               Labels are phrased to match the ground-truth definition
               (resolution satisfaction) rather than generic sentiment,
               directly addressing the label-mismatch problem.

Key design decisions:
- FinBERT is NOT serialised locally — loaded from HuggingFace Hub at runtime.
- truncation=True, max_length=512 always set (FinBERT / RoBERTa hard limit).
- Batching (batch_size=32) for all transformer models.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

try:
    import torch  # noqa: F401 — optional; speeds up inference when available
except Exception:  # pragma: no cover
    torch = None

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FINBERT_MODEL_ID = "ProsusAI/finbert"
ROBERTA_MODEL_ID = "cardiffnlp/twitter-roberta-base-sentiment-latest"
ZEROSHOT_MODEL_ID = "cross-encoder/nli-deberta-v3-small"

# Fine-tuned DistilBERT trained on CFPB resolution labels (notebook 02, §7)
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
FINETUNED_MODEL_PATH    = _PROJECT_ROOT / "models" / "distilbert_cx"
# Second-stage fine-tuning on GT labels (notebook 02, §12)
FINETUNED_MODEL_V2_PATH = _PROJECT_ROOT / "models" / "distilbert_cx_v2"

FINBERT_BATCH_SIZE = 32
ROBERTA_BATCH_SIZE = 32
ZEROSHOT_BATCH_SIZE = 16   # NLI inference is heavier per-sample
FINETUNED_BATCH_SIZE = 32

# VADER thresholds (standard practice)
VADER_POSITIVE_THRESHOLD = 0.05
VADER_NEGATIVE_THRESHOLD = -0.05

# Zero-shot candidate labels — phrased in terms of CX resolution satisfaction
# so the NLI model's semantic understanding aligns with the ground-truth definition.
ZEROSHOT_LABELS = {
    "positive": "The complaint was resolved to the customer's satisfaction with compensation or credit",
    "negative": "The complaint describes an unresolved problem or an unsatisfactory response from the company",
    "neutral":  "The complaint provides factual information without a clear indication of resolution or satisfaction",
}


class SentimentPipeline:
    """Unified interface for VADER, FinBERT, RoBERTa, and Zero-Shot sentiment.

    Usage:
        pipe = SentimentPipeline(method='roberta')
        result = pipe.analyze("The bank charged me incorrectly.")
        # {'label': 'negative', 'score': 0.91}

    Args:
        method: One of 'vader', 'finbert', 'roberta', 'zeroshot'.
    """

    VALID_METHODS = ("vader", "finbert", "roberta", "zeroshot", "finetuned", "finetuned_v2")

    def __init__(self, method: str = "finbert") -> None:
        if method not in self.VALID_METHODS:
            raise ValueError(f"method must be one of {self.VALID_METHODS}, got '{method}'")

        self.method = method

        if method == "vader":
            self._vader = SentimentIntensityAnalyzer()

        elif method == "finbert":
            from transformers import pipeline as hf_pipeline
            self._model = hf_pipeline(
                "text-classification",
                model=FINBERT_MODEL_ID,
                truncation=True,
                max_length=512,
            )

        elif method == "roberta":
            from transformers import pipeline as hf_pipeline
            self._model = hf_pipeline(
                "text-classification",
                model=ROBERTA_MODEL_ID,
                truncation=True,
                max_length=512,
            )

        elif method == "zeroshot":
            from transformers import pipeline as hf_pipeline
            self._model = hf_pipeline(
                "zero-shot-classification",
                model=ZEROSHOT_MODEL_ID,
            )
            # Candidate labels in insertion order → maps to positive/negative/neutral
            self._zeroshot_candidates = list(ZEROSHOT_LABELS.values())
            # Reverse map: label phrase → short label
            self._zeroshot_label_map = {v: k for k, v in ZEROSHOT_LABELS.items()}

        elif method == "finetuned":
            if not FINETUNED_MODEL_PATH.exists():
                raise FileNotFoundError(
                    f"Fine-tuned model not found at '{FINETUNED_MODEL_PATH}'. "
                    "Run notebook 02 Section 7 to train and save the model first."
                )
            from transformers import pipeline as hf_pipeline
            self._model = hf_pipeline(
                "text-classification",
                model=str(FINETUNED_MODEL_PATH),
                truncation=True,
                max_length=256,
            )

        elif method == "finetuned_v2":
            if not FINETUNED_MODEL_V2_PATH.exists():
                raise FileNotFoundError(
                    f"Fine-tuned v2 model not found at '{FINETUNED_MODEL_V2_PATH}'. "
                    "Run notebook 02 Section 12 to train and save the model first."
                )
            from transformers import pipeline as hf_pipeline
            self._model = hf_pipeline(
                "text-classification",
                model=str(FINETUNED_MODEL_V2_PATH),
                truncation=True,
                max_length=256,
            )

    # ------------------------------------------------------------------
    # Single-text inference
    # ------------------------------------------------------------------

    def analyze(self, text: str) -> dict[str, str | float]:
        """Classify the sentiment of a single text.

        Args:
            text: Raw or pre-cleaned customer comment.

        Returns:
            Dict with keys:
            - 'label': 'positive', 'negative', or 'neutral'
            - 'score': Confidence/compound score (float)
        """
        if not isinstance(text, str) or not text.strip():
            return {"label": "neutral", "score": 0.0}

        if self.method == "vader":
            return self._analyze_vader(text)
        if self.method == "zeroshot":
            return self._analyze_zeroshot_single(text)
        return self._analyze_transformer_single(text)

    def _analyze_vader(self, text: str) -> dict[str, str | float]:
        scores = self._vader.polarity_scores(text)
        compound = scores["compound"]
        if compound >= VADER_POSITIVE_THRESHOLD:
            label = "positive"
        elif compound <= VADER_NEGATIVE_THRESHOLD:
            label = "negative"
        else:
            label = "neutral"
        return {"label": label, "score": round(compound, 4)}

    def _analyze_transformer_single(self, text: str) -> dict[str, str | float]:
        """Single-text inference for FinBERT and RoBERTa."""
        result = self._model(text)[0]
        label = result["label"].lower()
        score = round(result["score"], 4)
        return {"label": label, "score": score}

    def _analyze_zeroshot_single(self, text: str) -> dict[str, str | float]:
        result = self._model(text, candidate_labels=self._zeroshot_candidates)
        # Highest-scored candidate wins
        top_phrase = result["labels"][0]
        top_score = result["scores"][0]
        label = self._zeroshot_label_map[top_phrase]
        return {"label": label, "score": round(top_score, 4)}

    # ------------------------------------------------------------------
    # Batch inference
    # ------------------------------------------------------------------

    def analyze_batch(self, texts: list[str]) -> pd.DataFrame:
        """Classify sentiment for a list of texts.

        Args:
            texts: List of customer comment strings.

        Returns:
            DataFrame with columns: text, label, score.
        """
        if self.method == "vader":
            results = [self._analyze_vader(t) for t in texts]
        elif self.method == "zeroshot":
            results = self._batch_zeroshot(texts)
        else:  # finbert, roberta, finetuned
            results = self._batch_transformer(texts)

        df = pd.DataFrame(results)
        df.insert(0, "text", texts)
        return df

    def _batch_transformer(self, texts: list[str]) -> list[dict]:
        """Batch inference for FinBERT, RoBERTa, and fine-tuned DistilBERT."""
        if self.method == "roberta":
            batch_size = ROBERTA_BATCH_SIZE
        elif self.method in ("finetuned", "finetuned_v2"):
            batch_size = FINETUNED_BATCH_SIZE
        else:
            batch_size = FINBERT_BATCH_SIZE
        all_results: list[dict] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            batch_results = self._model(batch)
            for r in batch_results:
                all_results.append(
                    {"label": r["label"].lower(), "score": round(r["score"], 4)}
                )
        return all_results

    def _batch_zeroshot(self, texts: list[str]) -> list[dict]:
        """Batch zero-shot classification."""
        all_results: list[dict] = []
        for i in range(0, len(texts), ZEROSHOT_BATCH_SIZE):
            batch = texts[i : i + ZEROSHOT_BATCH_SIZE]
            batch_results = self._model(batch, candidate_labels=self._zeroshot_candidates)
            for r in batch_results:
                top_phrase = r["labels"][0]
                top_score = r["scores"][0]
                label = self._zeroshot_label_map[top_phrase]
                all_results.append({"label": label, "score": round(top_score, 4)})
        return all_results

    # ------------------------------------------------------------------
    # Evaluation — all four models at once
    # ------------------------------------------------------------------

    def compare(
        self, texts: list[str], true_labels: list[str]
    ) -> dict[str, dict]:
        """Evaluate all four models against manually labeled ground truth.

        Args:
            texts: List of complaint narratives (aligned with true_labels).
            true_labels: Ground-truth labels ('positive', 'negative', 'neutral').

        Returns:
            Dict keyed by method name, each containing:
            - 'f1': weighted F1 score
            - 'accuracy': accuracy score
            - 'cm': confusion matrix (numpy array)
            - 'predictions': list of predicted labels
        """
        label_order = ["positive", "negative", "neutral"]
        results: dict[str, dict] = {}

        for method in self.VALID_METHODS:
            if method == self.method:
                pipe = self
            else:
                pipe = SentimentPipeline(method=method)

            preds = [pipe.analyze(t)["label"] for t in texts]
            results[method] = {
                "f1": round(f1_score(true_labels, preds, labels=label_order,
                                     average="weighted", zero_division=0), 4),
                "accuracy": round(accuracy_score(true_labels, preds), 4),
                "cm": confusion_matrix(true_labels, preds, labels=label_order),
                "predictions": preds,
            }

        return results
