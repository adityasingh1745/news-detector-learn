"""
Loads and merges all labeled headline/claim/article sources used to train
the clickbait/fake-news models:

  1. Clickbait-Detection-main/Dataset/{clickbait,non_clickbait}_data.txt
     - original curated clickbait headline corpus (now includes hand-added
       Indian-entity headlines for domain diversity).
  2. archive/india-news-headlines.csv
     - ~3.87M real Times-of-India headlines (2001-2021). All "real" (0).
  3. archive (1)/bharatfakenewskosh (3).xlsx
     - ~26k Indian fact-checked viral claims with English translations.
       Label True -> verified/real (0), Label False -> debunked/fake (1).
  4. archive (2)/news_dataset.csv
     - ~3.7k REAL/FAKE labeled news articles.

`build_full_dataset()` returns every row from every source (used for the
cheap TF-IDF + structural-features model, which can comfortably fit ~4M
short documents in a few minutes).

`build_capped_dataset()` returns a stratified/capped sample (used for
DistilBERT fine-tuning, where CPU-only training time scales linearly with
row count) — it keeps every fake/clickbait example (the minority class is
scarce) and randomly subsamples the much larger real-headline pool down to
`real_cap` rows.
"""
import csv
import logging
import os
import random
import re

logger = logging.getLogger(__name__)

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))

DATASET_DIR = os.path.join(REPO_ROOT, "Clickbait-Detection-main", "Dataset")
CLICKBAIT_TXT = os.path.join(DATASET_DIR, "clickbait_data.txt")
NON_CLICKBAIT_TXT = os.path.join(DATASET_DIR, "non_clickbait_data.txt")

ARCHIVE_HEADLINES_CSV = os.path.join(REPO_ROOT, "archive", "india-news-headlines.csv")
BHARAT_FAKE_NEWS_XLSX = os.path.join(REPO_ROOT, "archive (1)", "bharatfakenewskosh (3).xlsx")
NEWS_DATASET_CSV = os.path.join(REPO_ROOT, "archive (2)", "news_dataset.csv")

# Raised from the default 128KB limit because news_dataset.csv contains
# multi-paragraph article bodies embedded as quoted CSV fields.
csv.field_size_limit(10_000_000)


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def load_curated_pairs() -> tuple[list[str], list[int]]:
    """Original hand-curated + Indian-entity-augmented clickbait corpus."""
    with open(CLICKBAIT_TXT, encoding="utf-8") as f:
        clickbait = [line.strip() for line in f if line.strip()]
    with open(NON_CLICKBAIT_TXT, encoding="utf-8") as f:
        non_clickbait = [line.strip() for line in f if line.strip()]
    texts = clickbait + non_clickbait
    labels = [1] * len(clickbait) + [0] * len(non_clickbait)
    logger.info("Curated corpus: %d clickbait / %d non-clickbait", len(clickbait), len(non_clickbait))
    return texts, labels


def load_archive_real_headlines(sample_size: int | None = None, seed: int = 42) -> list[str]:
    """Real Indian news headlines from the Times-of-India archive.

    If `sample_size` is None, every row is returned (used for the TF-IDF
    model). Otherwise a reservoir sample of `sample_size` rows is returned,
    which keeps the sample spread uniformly across the whole file (all
    years / categories) rather than just the first N rows.
    """
    if not os.path.exists(ARCHIVE_HEADLINES_CSV):
        logger.warning("Archive headlines CSV not found at %s, skipping", ARCHIVE_HEADLINES_CSV)
        return []

    rng = random.Random(seed)
    reservoir: list[str] = []
    total = 0
    with open(ARCHIVE_HEADLINES_CSV, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = _clean(row.get("headline_text", ""))
            if not text:
                continue
            total += 1
            if sample_size is None:
                reservoir.append(text)
                continue
            if len(reservoir) < sample_size:
                reservoir.append(text)
            else:
                j = rng.randint(0, total - 1)
                if j < sample_size:
                    reservoir[j] = text
    logger.info(
        "Archive real headlines: %d available, %d loaded", total, len(reservoir),
    )
    return reservoir


def load_bharat_fake_news_kosh() -> tuple[list[str], list[int]]:
    """Fact-checked Indian viral claims (English translations), with Label
    True = verified/real, False = debunked/fake.
    """
    if not os.path.exists(BHARAT_FAKE_NEWS_XLSX):
        logger.warning("Bharat Fake News Kosh xlsx not found at %s, skipping", BHARAT_FAKE_NEWS_XLSX)
        return [], []

    import openpyxl

    wb = openpyxl.load_workbook(BHARAT_FAKE_NEWS_XLSX, read_only=True)
    ws = wb["A"]

    texts: list[str] = []
    labels: list[int] = []
    for row in ws.iter_rows(min_row=2, values_only=True):
        if len(row) <= 18:
            continue
        eng_statement = row[5]
        label = row[18]
        if label not in (True, False) or not eng_statement:
            continue
        text = _clean(str(eng_statement))
        if not text:
            continue
        texts.append(text)
        labels.append(0 if label is True else 1)
    logger.info("Bharat Fake News Kosh: %d real / %d fake claims", labels.count(0), labels.count(1))
    return texts, labels


def load_news_dataset_csv() -> tuple[list[str], list[int]]:
    """REAL/FAKE labeled news articles."""
    if not os.path.exists(NEWS_DATASET_CSV):
        logger.warning("news_dataset.csv not found at %s, skipping", NEWS_DATASET_CSV)
        return [], []

    texts: list[str] = []
    labels: list[int] = []
    with open(NEWS_DATASET_CSV, encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = (row.get("label") or "").strip().upper()
            text = _clean(row.get("text", ""))
            if label not in ("REAL", "FAKE") or not text:
                continue
            texts.append(text)
            labels.append(0 if label == "REAL" else 1)
    logger.info("news_dataset.csv: %d real / %d fake articles", labels.count(0), labels.count(1))
    return texts, labels


def build_full_dataset() -> tuple[list[str], list[int]]:
    """Every labeled row from every source. Used for the TF-IDF +
    structural-features model, which trains in minutes even on ~4M rows.
    """
    texts, labels = load_curated_pairs()

    real_headlines = load_archive_real_headlines(sample_size=None)
    texts.extend(real_headlines)
    labels.extend([0] * len(real_headlines))

    bfnk_texts, bfnk_labels = load_bharat_fake_news_kosh()
    texts.extend(bfnk_texts)
    labels.extend(bfnk_labels)

    nd_texts, nd_labels = load_news_dataset_csv()
    texts.extend(nd_texts)
    labels.extend(nd_labels)

    logger.info(
        "Full merged dataset: %d total (%d real / %d fake)",
        len(texts), labels.count(0), labels.count(1),
    )
    return texts, labels


def build_capped_dataset(real_cap: int = 120_000, seed: int = 42) -> tuple[list[str], list[int]]:
    """Stratified/capped sample for DistilBERT fine-tuning: keeps every
    fake/clickbait example, and subsamples the much larger real-class pool
    down to roughly `real_cap` rows so that CPU-only training finishes in a
    practical amount of time while still gaining massive domain diversity
    from the archive headlines.
    """
    curated_texts, curated_labels = load_curated_pairs()
    bfnk_texts, bfnk_labels = load_bharat_fake_news_kosh()
    nd_texts, nd_labels = load_news_dataset_csv()

    all_texts = curated_texts + bfnk_texts + nd_texts
    all_labels = curated_labels + bfnk_labels + nd_labels

    fake_texts = [t for t, l in zip(all_texts, all_labels) if l == 1]
    real_texts = [t for t, l in zip(all_texts, all_labels) if l == 0]

    remaining_real_budget = max(real_cap - len(real_texts), 0)
    archive_sample = load_archive_real_headlines(sample_size=remaining_real_budget, seed=seed)
    real_texts.extend(archive_sample)

    texts = fake_texts + real_texts
    labels = [1] * len(fake_texts) + [0] * len(real_texts)

    rng = random.Random(seed)
    combined = list(zip(texts, labels))
    rng.shuffle(combined)
    texts, labels = [t for t, _ in combined], [l for _, l in combined]

    logger.info(
        "Capped dataset for DistilBERT: %d total (%d real / %d fake)",
        len(texts), labels.count(0), labels.count(1),
    )
    return texts, list(labels)
