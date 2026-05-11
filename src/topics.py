"""
Topic modeling for banking complaint narratives.

Pipeline: spaCy lematization → TF-IDF vectorization → LDA topic model.
Coherence scoring (via gensim) guides the selection of n_topics.

Banking-domain stopwords are added to the standard English set to prevent
generic financial terms from polluting topic keywords.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BANKING_STOPWORDS = {
    "bank", "account", "loan", "credit", "card", "payment", "company",
    "consumer", "financial", "service", "customer", "complaint", "report",
    "statement", "information", "contact", "letter", "time", "day", "month",
    "year", "dollar", "amount", "charge", "fee", "transaction", "number",
    "request", "respond", "receive", "send", "tell", "know", "try", "call",
    "say", "get", "go", "make", "come", "take", "use", "give", "want",
    "need", "ask", "think", "let", "place", "way", "state", "states", "us",
    "would", "could", "should", "also", "even", "still", "back",
}

LDA_MAX_ITER = 20
LDA_RANDOM_STATE = 42
TFIDF_MIN_DF = 5
TFIDF_MAX_DF = 0.90
TFIDF_MAX_FEATURES = 5_000


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def preprocess_for_lda(texts: list[str], nlp_model: Any) -> list[str]:
    """Lemmatize and filter texts using a spaCy language model.

    Retains only alphabetic tokens that are not stopwords, not punctuation,
    and have length > 2. Adds banking-domain stopwords on top of spaCy's
    built-in stopword list.

    Args:
        texts: List of pre-cleaned complaint narratives.
        nlp_model: A loaded spaCy language model (e.g. spacy.load('en_core_web_sm')).

    Returns:
        List of whitespace-joined lemma strings, one per input text.
    """
    extended_stops = nlp_model.Defaults.stop_words | BANKING_STOPWORDS
    processed: list[str] = []

    # Use nlp.pipe for efficiency (batch processing)
    for doc in nlp_model.pipe(texts, batch_size=256, disable=["ner", "parser"]):
        tokens = [
            token.lemma_.lower()
            for token in doc
            if token.is_alpha
            and not token.is_stop
            and token.lemma_.lower() not in extended_stops
            and len(token.lemma_) > 2
        ]
        processed.append(" ".join(tokens))

    return processed


def build_tfidf_matrix(texts: list[str]) -> tuple[Any, TfidfVectorizer]:
    """Vectorize pre-processed texts with TF-IDF.

    Args:
        texts: List of lemmatized, stopword-filtered strings.

    Returns:
        Tuple of (sparse TF-IDF matrix, fitted TfidfVectorizer).
    """
    vectorizer = TfidfVectorizer(
        min_df=TFIDF_MIN_DF,
        max_df=TFIDF_MAX_DF,
        max_features=TFIDF_MAX_FEATURES,
        ngram_range=(1, 2),
    )
    matrix = vectorizer.fit_transform(texts)
    return matrix, vectorizer


def train_lda(matrix: Any, n_topics: int) -> LatentDirichletAllocation:
    """Fit a Latent Dirichlet Allocation model.

    Args:
        matrix: TF-IDF sparse matrix from build_tfidf_matrix().
        n_topics: Number of latent topics to extract.

    Returns:
        Fitted LDA model.
    """
    lda = LatentDirichletAllocation(
        n_components=n_topics,
        max_iter=LDA_MAX_ITER,
        learning_method="online",
        random_state=LDA_RANDOM_STATE,
    )
    lda.fit(matrix)
    return lda


def get_topic_keywords(
    lda_model: LatentDirichletAllocation,
    vectorizer: TfidfVectorizer,
    n_words: int = 10,
) -> dict[int, list[str]]:
    """Extract the top-N keywords for each LDA topic.

    Args:
        lda_model: Fitted LDA model.
        vectorizer: Fitted TfidfVectorizer used to build the matrix.
        n_words: Number of top keywords to return per topic.

    Returns:
        Dict mapping topic index → list of top keyword strings.
    """
    feature_names = vectorizer.get_feature_names_out()
    keywords: dict[int, list[str]] = {}
    for topic_idx, topic_vec in enumerate(lda_model.components_):
        top_indices = topic_vec.argsort()[: -n_words - 1 : -1]
        keywords[topic_idx] = [feature_names[i] for i in top_indices]
    return keywords


def assign_topics(
    df: pd.DataFrame,
    lda_model: LatentDirichletAllocation,
    vectorizer: TfidfVectorizer,
    text_col: str = "lemmatized_text",
) -> pd.DataFrame:
    """Add a 'dominant_topic' column to the DataFrame.

    Args:
        df: DataFrame containing a column with lemmatized texts.
        lda_model: Fitted LDA model.
        vectorizer: Fitted TfidfVectorizer.
        text_col: Name of the column holding lemmatized text strings.

    Returns:
        DataFrame with two new columns:
        - 'dominant_topic': integer index of the most probable topic
        - 'topic_probability': probability of the dominant topic
    """
    df = df.copy()
    matrix = vectorizer.transform(df[text_col].fillna(""))
    topic_matrix = lda_model.transform(matrix)
    df["dominant_topic"] = np.argmax(topic_matrix, axis=1)
    df["topic_probability"] = np.max(topic_matrix, axis=1)
    return df


def calculate_coherence(
    texts: list[str], range_topics: range
) -> list[dict[str, float | int]]:
    """Compute gensim coherence scores to guide n_topics selection.

    Uses the C_V coherence measure, which correlates well with human
    judgements of topic quality.

    Args:
        texts: List of tokenized texts (whitespace-separated strings).
        range_topics: Range of n_topics values to evaluate (e.g. range(3, 11)).

    Returns:
        List of dicts with keys 'n_topics' and 'coherence', sorted by n_topics.
    """
    from gensim import corpora
    from gensim.models import CoherenceModel, LdaModel

    tokenized = [t.split() for t in texts]
    dictionary = corpora.Dictionary(tokenized)
    dictionary.filter_extremes(no_below=5, no_above=0.9)
    corpus = [dictionary.doc2bow(doc) for doc in tokenized]

    results: list[dict] = []
    for n in range_topics:
        lda_g = LdaModel(
            corpus=corpus,
            id2word=dictionary,
            num_topics=n,
            random_state=LDA_RANDOM_STATE,
            passes=5,
        )
        cm = CoherenceModel(
            model=lda_g,
            texts=tokenized,
            dictionary=dictionary,
            coherence="c_v",
        )
        results.append({"n_topics": n, "coherence": round(cm.get_coherence(), 4)})

    return results


def save_lda_model(
    lda_model: LatentDirichletAllocation,
    vectorizer: TfidfVectorizer,
    path: str | Path,
) -> None:
    """Serialize the fitted LDA model and vectorizer together.

    Args:
        lda_model: Fitted LDA model.
        vectorizer: Fitted TfidfVectorizer.
        path: Output path (e.g. models/lda_model.pkl).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump({"lda": lda_model, "vectorizer": vectorizer}, f)
    print(f"LDA model saved → {path}")


def load_lda_model(path: str | Path) -> tuple[LatentDirichletAllocation, TfidfVectorizer]:
    """Load a previously serialized LDA model bundle.

    Args:
        path: Path to the pickle file produced by save_lda_model().

    Returns:
        Tuple of (lda_model, vectorizer).
    """
    with open(Path(path), "rb") as f:
        bundle = pickle.load(f)
    return bundle["lda"], bundle["vectorizer"]
