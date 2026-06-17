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

class PredictResponse(BaseModel):
    verdict: str
    confidence: float
    scores: dict
    indicators: list[str]
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
CLICKBAIT_PHRASES = [
    r"\byou won'?t believe\b",
    r"\bshock(ing|ed|s)?\b",
    r"\bwhat happen(s|ed) next\b",
    r"\bwill blow your mind\b",
    r"\bthis is why\b",
    r"\bsecret(ly|s)?\b",
    r"\b(amazing|incredible|unbelievable|stunning)\b",
    r"\bexclusive\b",
    r"\b(must.?see|must.?read|must.?watch)\b",
    r"\bviral\b",
    r"\b\d+\s+(reasons|ways|things|facts|secrets|tips)\b",
    r"\bhow to\b.*\b(instantly|immediately|overnight|fast)\b",
    r"!{1,}$",
    r"!!+",
    r"\?$",
    r"\bOMG\b",
    r"\bWOW\b",
]

FAKE_NEWS_PHRASES = [
    r"\bsecret agenda\b",
    r"\bdeep state\b",
    r"\bthey don'?t want you to know\b",
    r"\bmain ?stream media\b.*\b(lie|hiding|cover.?up)\b",
    r"\bwake up\b",
    r"\bsheep\b",
    r"\bhoax\b",
    r"\bscam\b",
    r"\bconspiracy\b",
    r"\bpedogate\b",
    r"\bfake\b",
    r"\bpropaganda\b",
    r"\billegit(imate)?\b",
    r"\bfraud\b",
    r"\bstolen\b.*\belection\b",
    r"\bmisinformation\b",
    r"\bsuppressed\b",
    r"\bwhistleblow\b",
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
    """Return (p_real, p_clickbait, p_fake) as 0-1 probabilities."""
    full_text = headline + (" " + body[:600] if body else "")

    cb_hits = count_pattern_hits(headline, CLICKBAIT_PHRASES)
    fake_hits = count_pattern_hits(full_text, FAKE_NEWS_PHRASES)
    credible_hits = count_pattern_hits(full_text, CREDIBLE_PHRASES)
    cr = caps_ratio(headline)
    has_excl = bool(re.search(r"!{1,}$", headline))
    has_question = bool(re.search(r"\?$", headline))
    listicle = bool(re.search(r"\b\d+\s+(reasons|ways|things|facts|tips)\b", headline.lower()))
    if has_excl:
        cb_hits += 1
    if listicle:
        cb_hits += 1
    if cr > CAPS_THRESHOLD:
        cb_hits += 2

    # Score clickbait
    p_cb = min(cb_hits / 5.0, 1.0) * 0.85

    # Score fake
    fake_raw = min(fake_hits / 4.0, 1.0)
    credibility_discount = min(credible_hits / 3.0, 1.0) * 0.4
    p_fake = max(0.0, fake_raw - credibility_discount) * 0.80

    # Real score — inversely proportional to cb+fake, boosted by credible phrases
    credibility_boost = min(credible_hits / 4.0, 0.6)
    p_real_raw = max(0.0, 1.0 - (p_cb + p_fake) * 0.5) * (0.5 + credibility_boost)

    # Normalise
    total = p_real_raw + p_cb + p_fake
    if total < 0.01:
        return 0.6, 0.2, 0.2
    return p_real_raw / total, p_cb / total, p_fake / total


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
    if count_pattern_hits(headline, FAKE_NEWS_PHRASES) > 0:
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
        # Blend: 60% ML model, 40% rule-based
        if model_label == "FAKE":
            p_fake_ml = model_conf
            p_real_ml = 1.0 - model_conf
        else:
            p_real_ml = model_conf
            p_fake_ml = 1.0 - model_conf

        # Distribute residual ML probability between fake and real proportionally
        p_real = 0.6 * p_real_ml + 0.4 * p_real_rb
        p_fake = 0.6 * p_fake_ml + 0.4 * p_fake_rb
        p_cb = 0.4 * p_cb_rb  # clickbait purely rule-based

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

    return PredictResponse(
        verdict=verdict,
        confidence=confidence,
        scores={
            "real": round(p_real * 100, 1),
            "clickbait": round(p_cb * 100, 1),
            "fake": round(p_fake * 100, 1),
        },
        indicators=indicators,
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
