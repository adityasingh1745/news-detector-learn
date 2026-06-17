"""
Fake News & Clickbait Detection ML Service
Uses HuggingFace transformers with rule-based fallback.
"""
import os
import re
import logging
from contextlib import asynccontextmanager
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
classifier = None
model_name = "loading"
model_ready = False

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
class PredictRequest(BaseModel):
    headline: str
    body: Optional[str] = None

class KeywordMatch(BaseModel):
    word: str
    score: int  # 0-100 severity for this specific signal

class PredictResponse(BaseModel):
    verdict: str
    confidence: float
    scores: dict
    indicators: list[str]
    keywords: list[KeywordMatch]
    model_used: str

class StatusResponse(BaseModel):
    ready: bool
    model_name: str
    feedback_count: int
    last_retrained: Optional[str]

# ---------------------------------------------------------------------------
# Lifespan — load model on startup
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global classifier, model_name, model_ready
    # Try models in order from smallest to largest
    CANDIDATE_MODELS = [
        "GonzaloA/fake-news-bert-base-uncased",
        "hamzab/roberta-fake-news-classification",
        "jy46604790/Fake-News-Bert-Detect",
    ]
    for candidate in CANDIDATE_MODELS:
        try:
            from transformers import pipeline
            logger.info(f"Attempting to load model: {candidate}")
            classifier = pipeline(
                "text-classification",
                model=candidate,
                device=-1,
                truncation=True,
                max_length=512,
            )
            model_name = candidate
            model_ready = True
            logger.info(f"Model loaded successfully: {model_name}")
            break
        except Exception as exc:
            logger.warning(f"Could not load {candidate}: {exc}")
            classifier = None

    if not model_ready:
        logger.warning("All transformer models failed; using enhanced rule-based classifier")
        model_name = "enhanced-rule-based-v1"
        model_ready = True

    yield
    # Cleanup
    classifier = None

app = FastAPI(title="Fake News ML Service", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Rule-based feature extraction
# ---------------------------------------------------------------------------
# (pattern, display_label, severity_score)
CB_SCORED: list[tuple[str, str, int]] = [
    (r"you won'?t believe",        "you won't believe",    92),
    (r"will blow your mind",        "will blow your mind",  95),
    (r"what happens? next",         "what happens next",    72),
    (r"must.?(?:see|read|watch)",   "must-see/read/watch",  75),
    (r"how to\b.{0,40}(?:instantly|immediately|overnight|fast)", "how to ... instantly", 72),
    (r"\d+\s+(?:reasons|ways|things|facts|secrets|tips)", None, 80),  # None = use matched text
    (r"shock(?:ing|ed|s)?",         None,                   78),
    (r"unbelievable|incredible",    None,                   74),
    (r"amazing|stunning",           None,                   65),
    (r"\bOMG\b",                    "OMG",                  85),
    (r"\bWOW\b",                    "WOW",                  80),
    (r"\bexclusive\b",              "exclusive",            60),
    (r"\bviral\b",                  "viral",                65),
    (r"secret(?:ly|s)?",            None,                   55),
]

FAKE_SCORED: list[tuple[str, str, int]] = [
    (r"they don'?t want you to know", "they don't want you to know", 95),
    (r"stolen\s+election",          "stolen election",      95),
    (r"deep state",                 "deep state",           92),
    (r"\bhoax\b",                   "hoax",                 90),
    (r"secret agenda",              "secret agenda",        88),
    (r"\bconspiracy\b",             "conspiracy",           85),
    (r"\bscam\b",                   "scam",                 82),
    (r"\bsuppressed\b",             "suppressed",           82),
    (r"\bpropaganda\b",             "propaganda",           80),
    (r"\bfraud\b",                  "fraud",                78),
    (r"\bmisinformation\b",         "misinformation",       75),
    (r"whistleblow\w*",             None,                   65),
    (r"\bwake up\b",                "wake up",              62),
    (r"mainstream media\b.{0,30}(?:lie|hid|cover)",  "mainstream media lies", 85),
    (r"\bpedogate\b",               "pedogate",             95),
]

CREDIBLE_PHRASES = [
    r"\baccording to\b",
    r"\bstudies show\b",
    r"\bresearch (shows|finds|indicates|suggests)\b",
    r"\bexperts say\b",
    r"\bofficial(s|ly)?\b",
    r"\bstatement\b",
    r"\bconference\b",
    r"\buniversity\b",
    r"\bgovernment\b",
    r"\breport(s|ed)?\b",
    r"\bsource(s)?\b",
    r"\bdata (shows|indicates|suggests)\b",
]

CAPS_THRESHOLD = 0.3


def count_pattern_hits(text: str, patterns: list[str]) -> int:
    text_l = text.lower()
    return sum(1 for p in patterns if re.search(p, text_l))


def caps_ratio(text: str) -> float:
    alpha = [c for c in text if c.isalpha()]
    if not alpha:
        return 0.0
    return sum(1 for c in alpha if c.isupper()) / len(alpha)


def rule_based_scores(headline: str, body: Optional[str]) -> tuple[float, float, float]:
    """Return (p_real, p_clickbait, p_fake) using weighted keyword scoring for more variance."""
    full_text = headline + (" " + body[:600] if body else "")
    headline_lower = headline.lower()
    full_lower = full_text.lower()

    # Weighted sums — vary naturally based on which/how-many signals fire
    cb_weight = sum(score for p, _, score in CB_SCORED if re.search(p, headline_lower))
    fake_weight = sum(score for p, _, score in FAKE_SCORED if re.search(p, full_lower))

    # Structural signals
    cr = caps_ratio(headline)
    if cr > CAPS_THRESHOLD:
        cb_weight += 60
    if re.search(r"!!+", headline):
        cb_weight += 55
    elif re.search(r"!$", headline):
        cb_weight += 30
    if re.search(r"\?$", headline):
        cb_weight += 20

    credible_hits = count_pattern_hits(full_text, CREDIBLE_PHRASES)

    # Convert weights to 0-1 (normalise against realistic max weights)
    p_cb_raw = min(cb_weight / 260.0, 1.0)
    p_fake_raw = min(fake_weight / 260.0, 1.0)

    credibility_discount = min(credible_hits / 3.0, 1.0) * 0.35
    p_fake_raw = max(0.0, p_fake_raw - credibility_discount)

    p_cb = p_cb_raw * 0.85
    p_fake = p_fake_raw * 0.80

    credibility_boost = min(credible_hits / 4.0, 0.6)
    p_real_raw = max(0.0, 1.0 - (p_cb + p_fake) * 0.5) * (0.5 + credibility_boost)

    total = p_real_raw + p_cb + p_fake
    if total < 0.01:
        return 0.6, 0.2, 0.2
    return p_real_raw / total, p_cb / total, p_fake / total


def extract_keywords(headline: str, body: Optional[str]) -> list[KeywordMatch]:
    """Return matched keywords with per-pattern severity scores, sorted highest first."""
    full_text = headline + (" " + body[:600] if body else "")
    headline_lower = headline.lower()
    full_lower = full_text.lower()

    found: list[KeywordMatch] = []
    seen: set[str] = set()

    for pattern, label, score in CB_SCORED:
        m = re.search(pattern, headline_lower)
        if m:
            word = label if label else m.group(0).strip()
            if word.lower() not in seen:
                seen.add(word.lower())
                found.append(KeywordMatch(word=word, score=score))

    for pattern, label, score in FAKE_SCORED:
        m = re.search(pattern, full_lower)
        if m:
            word = label if label else m.group(0).strip()
            if word.lower() not in seen:
                seen.add(word.lower())
                found.append(KeywordMatch(word=word, score=score))

    # Structural meta-signals
    if re.search(r"!!+", headline):
        found.append(KeywordMatch(word="!!!", score=70))
    elif re.search(r"!$", headline):
        found.append(KeywordMatch(word="!", score=45))
    if re.search(r"\?$", headline):
        found.append(KeywordMatch(word="?", score=40))
    if caps_ratio(headline) > CAPS_THRESHOLD:
        found.append(KeywordMatch(word="ALL CAPS", score=75))

    found.sort(key=lambda k: k.score, reverse=True)
    return found[:12]


def extract_indicators(
    headline: str, body: Optional[str], p_cb: float, p_fake: float, model_label: Optional[str]
) -> list[str]:
    indicators: list[str] = []
    h_lower = headline.lower()
    if p_cb > 0.4:
        indicators.append("Sensational or emotionally-charged language detected")
    if re.search(r"!{1,}$", headline):
        indicators.append("Headline ends with exclamation mark(s)")
    if re.search(r"\b\d+\s+(reasons|ways|things|facts|tips)\b", h_lower):
        indicators.append("Listicle-style headline (numbered bait)")
    if caps_ratio(headline) > CAPS_THRESHOLD:
        indicators.append("Excessive capital letters")
    if any(re.search(p, headline.lower()) for p, _, _ in FAKE_SCORED):
        indicators.append("Conspiracy-related vocabulary detected")
    if body and len(body.strip()) < 100:
        indicators.append("Very short article body — limited factual context")
    if count_pattern_hits(headline + (body or ""), CREDIBLE_PHRASES) >= 2:
        indicators.append("Multiple credible-source references found")
    if model_label == "FAKE":
        indicators.append("BERT classifier: language matches known misinformation patterns")
    if model_label == "REAL":
        indicators.append("BERT classifier: language consistent with factual reporting")
    return indicators if indicators else ["No strong signals detected — borderline content"]


# ---------------------------------------------------------------------------
# Label normalisation for HuggingFace model outputs
# ---------------------------------------------------------------------------
def normalise_label(raw_label: str) -> str:
    """Map any HuggingFace label to FAKE|REAL."""
    u = raw_label.upper()
    if u in ("FAKE", "LABEL_0", "LABEL-0", "0"):
        return "FAKE"
    if u in ("REAL", "LABEL_1", "LABEL-1", "1", "TRUE"):
        return "REAL"
    # hamzab/roberta-fake-news uses 0=FAKE 1=REAL too
    return "REAL"


# ---------------------------------------------------------------------------
# Core prediction
# ---------------------------------------------------------------------------
def predict(headline: str, body: Optional[str]) -> PredictResponse:
    combined = headline + (" " + body[:400] if body else "")

    model_label: Optional[str] = None
    model_conf: float = 0.5

    if classifier is not None:
        try:
            result = classifier(combined[:512])[0]
            model_label = normalise_label(result["label"])
            model_conf = float(result["score"])
        except Exception as exc:
            logger.warning(f"Inference error: {exc}")

    # Rule-based scores (always computed)
    p_real_rb, p_cb_rb, p_fake_rb = rule_based_scores(headline, body)

    if model_label is not None:
        if model_label == "FAKE":
            p_fake_ml = model_conf
            p_real_ml = 1.0 - model_conf
        else:
            p_real_ml = model_conf
            p_fake_ml = 1.0 - model_conf

        # Dynamic blend: more keyword evidence → more weight on rule-based scoring
        # This creates genuine variance across headlines
        combined_lower = combined.lower()
        kw_weight = (
            sum(score for p, _, score in CB_SCORED if re.search(p, combined_lower)) +
            sum(score for p, _, score in FAKE_SCORED if re.search(p, combined_lower))
        )
        rb_ratio = min(0.30 + kw_weight / 700.0, 0.65)  # 0.30 (weak) → 0.65 (strong signals)
        ml_ratio = 1.0 - rb_ratio

        p_real = ml_ratio * p_real_ml + rb_ratio * p_real_rb
        p_fake = ml_ratio * p_fake_ml + rb_ratio * p_fake_rb
        p_cb = rb_ratio * p_cb_rb

        total = p_real + p_fake + p_cb
        if total > 0:
            p_real /= total
            p_fake /= total
            p_cb /= total
    else:
        p_real, p_cb, p_fake = p_real_rb, p_cb_rb, p_fake_rb

    scores_map = {"REAL": p_real, "CLICKBAIT": p_cb, "FAKE": p_fake}
    verdict = max(scores_map, key=lambda k: scores_map[k])
    confidence = round(scores_map[verdict] * 100, 1)

    indicators = extract_indicators(headline, body, p_cb, p_fake, model_label)
    keywords = extract_keywords(headline, body)

    return PredictResponse(
        verdict=verdict,
        confidence=confidence,
        scores={
            "real": round(p_real * 100, 1),
            "clickbait": round(p_cb * 100, 1),
            "fake": round(p_fake * 100, 1),
        },
        indicators=indicators,
        keywords=keywords,
        model_used=model_name,
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.post("/predict", response_model=PredictResponse)
async def predict_route(req: PredictRequest):
    if not model_ready:
        raise HTTPException(status_code=503, detail="Model is still initialising, please retry shortly")
    return predict(req.headline, req.body)


@app.get("/status", response_model=StatusResponse)
async def status_route():
    return StatusResponse(
        ready=model_ready,
        model_name=model_name,
        feedback_count=0,
        last_retrained=None,
    )


@app.get("/healthz")
async def health():
    return {"status": "ok", "model_ready": model_ready}


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("ML_PORT", "8001"))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
