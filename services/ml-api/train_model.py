"""
Trains a TF-IDF + Logistic Regression clickbait classifier on the labeled
headline dataset in Clickbait-Detection-main/Dataset and saves the fitted
vectorizer + model to services/ml-api/models/clickbait_model.joblib.

Run with:
    python train_model.py
"""
import os
import logging

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler

from features import StructuralFeatures
from dataset_sources import build_capped_dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "clickbait_model.joblib")

# Real-class row budget: the full archive corpus is ~3.9M rows vs. ~28k
# fake/clickbait rows (138:1). Training on all of it — even with
# class_weight="balanced" — over-corrects and collapses clickbait-class
# precision (measured: 0.34). Capping the real class to a more moderate
# ratio (~10:1) keeps the Indian-headline vocabulary/domain diversity gains
# while preserving usable precision/recall (measured: 0.68/0.80 @ cap
# 300k + class_weight {0: 1, 1: 3}, vs. 0.34/0.92 uncapped/balanced).
REAL_CLASS_CAP = 300_000
CLASS_WEIGHT = {0: 1, 1: 3}


def main() -> None:
    # Pulls in the curated clickbait corpus plus every bulk Indian-domain
    # source (archive headlines, fact-checked claims, labeled articles) so
    # the model sees real-world vocabulary/entities, not just the original
    # Buzzfeed/NYT-style dataset.
    texts, label_list = build_capped_dataset(real_cap=REAL_CLASS_CAP)
    labels = np.array(label_list)

    x_train, x_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.15, random_state=42, stratify=labels,
    )

    # Combine bag-of-words (TF-IDF) with hand-engineered structural features
    # (length, punctuation, caps ratio, listicle framing, etc.) so the model
    # also learns *how* a headline is written, not just which words it
    # contains — this generalizes much better to unseen vocabulary/entities.
    vectorizer = FeatureUnion([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            max_features=30000,
            sublinear_tf=True,
            min_df=2,
        )),
        ("structural", Pipeline([
            ("extract", StructuralFeatures()),
            ("scale", StandardScaler()),
        ])),
    ])
    x_train_vec = vectorizer.fit_transform(x_train)
    x_test_vec = vectorizer.transform(x_test)

    # class_weight is moderately tilted toward the minority (clickbait)
    # class — true inverse-frequency "balanced" weighting over-corrects at
    # this scale and collapses precision (see note above).
    clf = LogisticRegression(max_iter=1000, C=5.0, class_weight=CLASS_WEIGHT, solver="liblinear")
    clf.fit(x_train_vec, y_train)

    y_pred = clf.predict(x_test_vec)
    acc = accuracy_score(y_test, y_pred)
    logger.info("Test accuracy: %.4f", acc)
    logger.info("\n%s", classification_report(y_test, y_pred, target_names=["real", "clickbait"]))

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(
        {
            "vectorizer": vectorizer,
            "classifier": clf,
            "accuracy": acc,
            "model_name": "tfidf-structural-logreg-v3-fulldata",
        },
        MODEL_PATH,
    )
    logger.info("Saved model to %s", MODEL_PATH)


if __name__ == "__main__":
    main()
