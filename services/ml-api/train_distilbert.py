"""
Fine-tunes DistilBERT (contextual embeddings) as an upgrade path beyond the
TF-IDF + structural-features Logistic Regression model. Headlines are very
short, so traditional bag-of-words vectorization throws away word order and
semantic context; a pre-trained Transformer actually understands sentence
meaning, which should generalize much better — especially to phrasing it
has never seen combined with unfamiliar vocabulary/entities.

This trains on a stratified/capped merge of every labeled source available
(curated clickbait_data.txt/non_clickbait_data.txt with Indian-entity
additions, the Bharat Fake News Kosh fact-checked claims, news_dataset.csv
articles, and a large random sample of the archive Indian headlines corpus
— see dataset_sources.build_capped_dataset) and saves the fine-tuned model +
tokenizer to services/ml-api/models/distilbert_clickbait/. main.py will
automatically pick it up at the next restart and blend its prediction in
alongside the existing TF-IDF/rule-based score (see `ml_predict_proba` /
`transformer_predict_proba` in main.py).

NOTE: this runs on CPU on this machine (no CUDA available). The real-class
row count is capped (REAL_CLASS_CAP) so the run finishes in a few hours
instead of days. Progress + per-epoch checkpoints are logged/saved so
partial progress is never silently lost.

Run with:
    python train_distilbert.py
"""
import logging
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
)

from dataset_sources import build_capped_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, "models", "distilbert_clickbait")

BASE_MODEL = "distilbert-base-uncased"
MAX_LENGTH = 48
BATCH_SIZE = 32
EPOCHS = 2
LEARNING_RATE = 2e-5
# Keep end-of-epoch eval fast — a few thousand held-out examples is plenty
# to track whether the model is learning, without doubling training time.
EVAL_SUBSET_SIZE = 3000
# Real-class row budget for the merged dataset (curated + bulk Indian
# sources). Keeps CPU-only fine-tuning in the hours range instead of days,
# while still bringing in massive domain-diversity gains over the original
# ~32k-headline corpus.
REAL_CLASS_CAP = 120_000


class HeadlineDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def _evaluate(model, loader, device) -> tuple[float, str]:
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            labels = batch.pop("labels")
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch).logits
            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.numpy().tolist())
    acc = accuracy_score(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=["real", "clickbait"])
    return acc, report


def main() -> None:
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Use every available CPU core for matmul — this is the single biggest
    # lever for training speed when no GPU is present.
    n_threads = os.cpu_count() or 4
    torch.set_num_threads(n_threads)
    logger.info("Training on device=%s with %d CPU threads", device, n_threads)

    text_list, label_list = build_capped_dataset(real_cap=REAL_CLASS_CAP)
    texts = text_list
    labels = np.array(label_list)
    logger.info(
        "Loaded %d total examples (%d clickbait/fake, %d real)",
        len(texts), int(labels.sum()), int((labels == 0).sum()),
    )

    x_train, x_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.15, random_state=42, stratify=labels,
    )
    # Trim eval set for speed — still a representative, stratified sample.
    if len(x_test) > EVAL_SUBSET_SIZE:
        x_test_eval, _, y_test_eval, _ = train_test_split(
            x_test, y_test, train_size=EVAL_SUBSET_SIZE, random_state=42, stratify=y_test,
        )
    else:
        x_test_eval, y_test_eval = x_test, y_test

    logger.info("Downloading/loading tokenizer + base model: %s", BASE_MODEL)
    tokenizer = DistilBertTokenizerFast.from_pretrained(BASE_MODEL)
    model = DistilBertForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=2)
    model.to(device)

    logger.info("Tokenizing %d train / %d eval examples", len(x_train), len(x_test_eval))
    train_enc = tokenizer(x_train, truncation=True, padding="max_length", max_length=MAX_LENGTH, return_tensors="pt")
    eval_enc = tokenizer(x_test_eval, truncation=True, padding="max_length", max_length=MAX_LENGTH, return_tensors="pt")

    train_ds = HeadlineDataset(train_enc, y_train.tolist())
    eval_ds = HeadlineDataset(eval_enc, list(y_test_eval))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    eval_loader = DataLoader(eval_ds, batch_size=BATCH_SIZE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    total_steps = len(train_loader) * EPOCHS
    logger.info("Starting training: %d epochs, %d steps/epoch, %d total steps", EPOCHS, len(train_loader), total_steps)

    # Mirror the TF-IDF model's class_weight="balanced": the real class
    # still outnumbers fake/clickbait roughly 4:1 even after capping, so
    # weight the loss inversely to class frequency.
    class_counts = np.bincount(y_train, minlength=2).astype(np.float64)
    class_weights = torch.tensor(
        len(y_train) / (2.0 * np.clip(class_counts, 1, None)), dtype=torch.float32,
    ).to(device)
    logger.info("Class weights (real, clickbait): %s", class_weights.tolist())
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    os.makedirs(MODEL_DIR, exist_ok=True)
    step = 0
    start_time = time.time()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        epoch_start = time.time()
        for batch in train_loader:
            labels_batch = batch.pop("labels").to(device)
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            outputs = model(**batch)
            loss = loss_fn(outputs.logits, labels_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            step += 1
            if step % 50 == 0:
                elapsed = time.time() - start_time
                rate = step / elapsed
                remaining = (total_steps - step) / rate if rate > 0 else float("nan")
                logger.info(
                    "epoch %d/%d step %d/%d loss=%.4f avg_loss=%.4f elapsed=%.0fs eta=%.0fs",
                    epoch, EPOCHS, step, total_steps, loss.item(),
                    running_loss / (step % len(train_loader) or len(train_loader)),
                    elapsed, remaining,
                )

        acc, report = _evaluate(model, eval_loader, device)
        logger.info("Epoch %d done in %.0fs — eval accuracy=%.4f\n%s", epoch, time.time() - epoch_start, acc, report)

        # Save a checkpoint after every epoch so progress is never lost if
        # the process gets interrupted partway through a long CPU run.
        model.save_pretrained(MODEL_DIR)
        tokenizer.save_pretrained(MODEL_DIR)
        with open(os.path.join(MODEL_DIR, "metrics.txt"), "w", encoding="utf-8") as f:
            f.write(f"epoch={epoch}\naccuracy={acc:.4f}\ntrained_at={time.time()}\n")
        logger.info("Saved checkpoint to %s after epoch %d", MODEL_DIR, epoch)

    logger.info("Training complete in %.0fs", time.time() - start_time)


if __name__ == "__main__":
    main()
