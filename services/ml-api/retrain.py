"""
Retrains the clickbait TF-IDF + Logistic Regression model by combining the
original 32k-headline labeled dataset with real user feedback collected in
Postgres. This is what makes the model *actually* learn over time — earlier,
feedback only overrode the stored verdict for one exact headline; this
module folds every majority-agreed correction back into the model's training
data so it generalizes to new, unseen headlines too.
"""
import os
import time
import logging
from collections import defaultdict

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler

from features import StructuralFeatures
from dataset_sources import build_capped_dataset

logger = logging.getLogger(__name__)

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "clickbait_model.joblib")

# Keep in sync with train_model.py — see the comment there for why the real
# class is capped rather than using the full ~3.9M-row archive uncapped.
REAL_CLASS_CAP = 300_000
CLASS_WEIGHT = {0: 1, 1: 3}


def _load_base_dataset() -> tuple[list[str], list[int]]:
    return build_capped_dataset(real_cap=REAL_CLASS_CAP)


def _load_teacher_examples(database_url: str, provider: str) -> tuple[list[str], list[int], int]:
    """
    Pull every analysis that was labeled by a given "teacher" API
    (model_used starts with '<provider>:', e.g. 'gemini:' or 'groq:') and
    turn it into a labeled training example -- this is how the local model
    distills Gemini's and Groq's judgment over time so that, eventually,
    the local model alone is accurate enough and neither API key is needed
    for day-to-day classification. Low-confidence calls are skipped since
    they're less reliable teacher signals.
    Returns (texts, labels, raw_row_count).
    """
    import psycopg2  # imported lazily so the service still runs without it
    from urllib.parse import urlparse, unquote

    MIN_CONFIDENCE = float(os.environ.get("GEMINI_TRAINING_MIN_CONFIDENCE", "60"))

    texts: list[str] = []
    labels: list[int] = []
    try:
        parsed = urlparse(database_url)
        conn = psycopg2.connect(
            dbname=parsed.path.lstrip("/"),
            user=unquote(parsed.username) if parsed.username else None,
            password=unquote(parsed.password) if parsed.password else None,
            host=parsed.hostname,
            port=parsed.port or 5432,
        )
        cur = conn.cursor()
        cur.execute(
            """
            SELECT headline, body, verdict, confidence
            FROM analyses
            WHERE model_used LIKE %s
              AND verdict IN (\'REAL\', \'CLICKBAIT\')
              AND confidence >= %s
            """,
            (f"{provider}:%", MIN_CONFIDENCE),
        )
        rows = cur.fetchall()
        cur.close()
        conn.close()
    except Exception:
        logger.exception("Could not load %s-labeled analyses from database for retraining", provider)
        return texts, labels, 0

    for headline, body, verdict, _confidence in rows:
        text = headline if not body else f"{headline} {body}"
        texts.append(text)
        labels.append(1 if verdict == "CLICKBAIT" else 0)

    return texts, labels, len(rows)

def _load_feedback_examples(database_url: str) -> tuple[list[str], list[int], int]:
    """
    Pull every piece of user feedback, resolve conflicting votes on the same
    headline via majority vote (same rule already used for live overrides in
    the API server), and turn each confidently-agreed correction into a new
    labeled training example.
    Returns (texts, labels, raw_feedback_row_count).
    """
    import psycopg2  # imported lazily so the service still runs without it
    from urllib.parse import urlparse, unquote

    texts: list[str] = []
    labels: list[int] = []
    raw_count = 0
    try:
        # Parse manually rather than passing the DSN straight through — a
        # literal "@" in the password (common when reusing a personal
        # password) otherwise gets misread as the user@host separator.
        parsed = urlparse(database_url)
        conn = psycopg2.connect(
            dbname=parsed.path.lstrip("/"),
            user=unquote(parsed.username) if parsed.username else None,
            password=unquote(parsed.password) if parsed.password else None,
            host=parsed.hostname,
            port=parsed.port or 5432,
        )
        cur = conn.cursor()
        cur.execute(
            """
            SELECT a.headline, a.body, f.correct_label
            FROM feedback f
            JOIN analyses a ON f.analysis_id = a.id
            """
        )
        rows = cur.fetchall()
        cur.close()
        conn.close()
    except Exception:
        logger.exception("Could not load feedback from database for retraining")
        return texts, labels, raw_count

    raw_count = len(rows)
    votes: dict[str, dict] = defaultdict(lambda: {"REAL": 0, "CLICKBAIT": 0, "headline": "", "body": None})
    for headline, body, label in rows:
        if label not in ("REAL", "CLICKBAIT"):
            continue
        key = headline.strip().lower()
        votes[key][label] += 1
        votes[key]["headline"] = headline
        votes[key]["body"] = body

    for v in votes.values():
        real_v, cb_v = v["REAL"], v["CLICKBAIT"]
        if real_v == cb_v:
            continue  # tied vote — no confident consensus, don't train on it
        label = 0 if real_v > cb_v else 1
        text = v["headline"] if not v["body"] else f'{v["headline"]} {v["body"]}'
        texts.append(text)
        labels.append(label)

    return texts, labels, raw_count


def retrain(database_url: str | None = None) -> dict:
    """Retrain the model from base dataset + feedback + teacher-labeled
    (Gemini and Groq) analyses, and save it to disk. Every analysis either
    API classified is folded back in as a labeled example (knowledge
    distillation), so the local model gradually learns to reproduce their
    judgment without needing either API key.
    """
    base_texts, base_labels = _load_base_dataset()
    fb_texts, fb_labels, raw_feedback_count = ([], [], 0)
    gm_texts, gm_labels, raw_gemini_count = ([], [], 0)
    gq_texts, gq_labels, raw_groq_count = ([], [], 0)
    if database_url:
        fb_texts, fb_labels, raw_feedback_count = _load_feedback_examples(database_url)
        gm_texts, gm_labels, raw_gemini_count = _load_teacher_examples(database_url, "gemini")
        gq_texts, gq_labels, raw_groq_count = _load_teacher_examples(database_url, "groq")

    texts = base_texts + fb_texts + gm_texts + gq_texts
    labels = np.array(base_labels + fb_labels + gm_labels + gq_labels)
    # Track provenance per example so we can measure, after the split, how
    # well the model specifically reproduces each teacher's labels — that
    # combined number is what determines whether the local model is ready
    # to stand alone without either API.
    sources = (
        ["base"] * len(base_texts)
        + ["feedback"] * len(fb_texts)
        + ["gemini"] * len(gm_texts)
        + ["groq"] * len(gq_texts)
    )

    x_train, x_test, y_train, y_test, src_train, src_test = train_test_split(
        texts, labels, sources, test_size=0.15, random_state=42, stratify=labels,
    )

    vectorizer = FeatureUnion([
        ("tfidf", TfidfVectorizer(
            lowercase=True, ngram_range=(1, 2), max_features=30000,
            sublinear_tf=True, min_df=2,
        )),
        ("structural", Pipeline([
            ("extract", StructuralFeatures()),
            ("scale", StandardScaler()),
        ])),
    ])
    x_train_vec = vectorizer.fit_transform(x_train)
    x_test_vec = vectorizer.transform(x_test)

    clf = LogisticRegression(max_iter=1000, C=5.0, class_weight=CLASS_WEIGHT, solver="liblinear")
    clf.fit(x_train_vec, y_train)

    preds = clf.predict(x_test_vec)
    acc = accuracy_score(y_test, preds)

    def _teacher_accuracy(provider: str) -> float | None:
        idx = [i for i, s in enumerate(src_test) if s == provider]
        if not idx:
            return None
        return float(accuracy_score(np.array(y_test)[idx], preds[idx]))

    gemini_accuracy = _teacher_accuracy("gemini")
    groq_accuracy = _teacher_accuracy("groq")

    teacher_test_idx = [i for i, s in enumerate(src_test) if s in ("gemini", "groq")]
    teacher_accuracy = (
        float(accuracy_score(np.array(y_test)[teacher_test_idx], preds[teacher_test_idx]))
        if teacher_test_idx
        else None
    )

    artifact = {
        "vectorizer": vectorizer,
        "classifier": clf,
        "accuracy": acc,
        "model_name": "tfidf-structural-logreg-v3-fulldata",
        "feedback_examples_used": len(fb_texts),
        "raw_feedback_rows": raw_feedback_count,
        "gemini_examples_used": len(gm_texts),
        "raw_gemini_rows": raw_gemini_count,
        "gemini_accuracy": gemini_accuracy,
        "groq_examples_used": len(gq_texts),
        "raw_groq_rows": raw_groq_count,
        "groq_accuracy": groq_accuracy,
        "teacher_examples_used": len(gm_texts) + len(gq_texts),
        "teacher_accuracy": teacher_accuracy,
        "trained_at": time.time(),
    }
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(artifact, MODEL_PATH)
    logger.info(
        "Retrained model: base_examples=%d feedback_examples=%d (from %d raw votes) "
        "gemini_examples=%d groq_examples=%d test_acc=%.4f gemini_acc=%s groq_acc=%s teacher_acc=%s",
        len(base_texts), len(fb_texts), raw_feedback_count, len(gm_texts), len(gq_texts), acc,
        f"{gemini_accuracy:.4f}" if gemini_accuracy is not None else "n/a",
        f"{groq_accuracy:.4f}" if groq_accuracy is not None else "n/a",
        f"{teacher_accuracy:.4f}" if teacher_accuracy is not None else "n/a",
    )
    return artifact


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    retrain(os.environ.get("DATABASE_URL"))
