"""
Hand-engineered structural/stylistic features for headline text, combined
with TF-IDF word features in the clickbait classifier (see train_model.py /
retrain.py). Pure bag-of-words only tells the model *what words* appear;
these features tell it *how the headline is written* — length, punctuation,
capitalization, listicle framing, etc. — which generalizes far better to
unseen vocabulary (e.g. new entities/names the TF-IDF model has never seen).

This module must stay import-stable (class path `features.StructuralFeatures`)
since fitted instances are pickled inside the joblib model artifact and
need to be unpickled the same way at inference time in main.py.
"""
import re

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "will", "would",
    "could", "should", "of", "in", "on", "at", "to", "for", "and", "or", "but",
    "with", "by", "from", "as", "this", "that", "these", "those", "it", "its",
    "his", "her", "their", "your", "our", "my", "has", "have", "had", "not",
}

_QUESTION_WORDS = {"why", "how", "what", "who", "when", "where", "which"}

_LISTICLE_RE = re.compile(
    r"\b\d+\s+(?:reasons?|ways?|things?|facts?|tips?|signs?|foods?|mistakes?|"
    r"hacks?|secrets?|tricks?|times?)\b",
    re.IGNORECASE,
)


def _word_tokens(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9']+", text)


class StructuralFeatures(BaseEstimator, TransformerMixin):
    """
    Transforms a list of raw headline strings into a dense numeric feature
    matrix capturing surface structure (length, punctuation, capitalization,
    listicle/question framing) rather than specific vocabulary.
    """

    #: Keep in sync with the number of values returned by `_row`.
    N_FEATURES = 13

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.array([self._row(text) for text in X], dtype=np.float64)

    def get_feature_names_out(self, input_features=None):
        return np.array([
            "n_words", "n_chars", "avg_word_len", "caps_ratio", "stopword_ratio",
            "digit_token_count", "starts_with_number", "starts_with_question_word",
            "exclaim_count", "question_count", "ends_with_exclaim",
            "ends_with_question", "is_listicle",
        ])

    def _row(self, text: str) -> list[float]:
        text = text or ""
        words = _word_tokens(text)
        n_words = len(words) or 1
        n_chars = len(text)
        alpha = [c for c in text if c.isalpha()]
        caps_ratio = (sum(1 for c in alpha if c.isupper()) / len(alpha)) if alpha else 0.0
        avg_word_len = sum(len(w) for w in words) / n_words
        stopword_ratio = sum(1 for w in words if w.lower() in _STOPWORDS) / n_words
        digit_token_count = sum(1 for w in words if w.isdigit())
        starts_with_number = 1.0 if words and words[0].isdigit() else 0.0
        starts_with_question_word = 1.0 if words and words[0].lower() in _QUESTION_WORDS else 0.0
        exclaim_count = float(text.count("!"))
        question_count = float(text.count("?"))
        stripped = text.rstrip()
        ends_with_exclaim = 1.0 if stripped.endswith("!") else 0.0
        ends_with_question = 1.0 if stripped.endswith("?") else 0.0
        is_listicle = 1.0 if _LISTICLE_RE.search(text.lower()) else 0.0

        return [
            float(n_words), float(n_chars), avg_word_len, caps_ratio, stopword_ratio,
            float(digit_token_count), starts_with_number, starts_with_question_word,
            exclaim_count, question_count, ends_with_exclaim, ends_with_question,
            is_listicle,
        ]
