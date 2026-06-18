"""
News Credibility & Clickbait Detection ML Service
Classifies headlines as REAL (credible journalism) or CLICKBAIT (sensational/bait).
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

classifier = None
model_name = "loading"
model_ready = False


class PredictRequest(BaseModel):
    headline: str
    body: Optional[str] = None


class KeywordMatch(BaseModel):
    word: str
    score: int


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


@asynccontextmanager
async def lifespan(app: FastAPI):
    global classifier, model_name, model_ready
    CANDIDATE_MODELS = [
        "hamzab/roberta-fake-news-classification",
        "GonzaloA/fake-news-bert-base-uncased",
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
        logger.warning("All transformer models failed; using rule-based classifier")
        model_name = "rule-based-v2"
        model_ready = True

    yield
    classifier = None


app = FastAPI(title="News Credibility ML Service", version="2.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ---------------------------------------------------------------------------
# Pattern lists: (pattern, display_label, severity_score 0-100)
# label=None means use matched text directly
# ---------------------------------------------------------------------------

# CLICKBAIT patterns — signal sensationalism, curiosity-gap, emotional manipulation
CB_SCORED: list[tuple[str, str | None, int]] = [
    # Curiosity gap — classic bait
    (r"you won'?t believe",                         "you won't believe",           95),
    (r"will blow your mind",                        "will blow your mind",         95),
    (r"the reason will shock you",                  "the reason will shock you",   92),
    (r"you need to see this",                       "you need to see this",        90),
    (r"they don'?t want you to know",               "they don't want you to know", 90),
    (r"this (?:one )?(?:weird|simple) trick",       "this one weird trick",        92),
    (r"doctors? (?:hate|won'?t tell) (?:him|her|you|this)", "doctors hate this",   90),
    (r"what (?:happens?|happened) next",            "what happens next",           88),
    (r"what nobody (?:tells?|told) you",            "what nobody tells you",       88),
    (r"(?:can|could) you believe",                  "can you believe",             82),
    (r"nobody (?:expected|saw|believes?) this",     "nobody expected this",        85),
    (r"you (?:won'?t|wouldn'?t) guess",             "you wouldn't guess",          85),
    (r"find out (?:why|how|what)",                  "find out why/how",            80),
    (r"the (?:shocking|real|hidden|untold) truth",  "the hidden truth",            82),
    (r"what (?:really|actually) happened?",         "what really happened",        78),
    (r"secrets? (?:of|to|about|behind)",            "secret of/about",             75),
    (r"the (?:real )?reason (?:why|that|behind)",   "the real reason",             72),
    (r"(?:here'?s?|this is) why\b",                 "here's why",                  60),
    (r"what (?:your|the) .{0,30} (?:doesn'?t|won'?t) (?:tell|want)", "what they won't tell", 82),
    (r"(?:the|a) (?:shocking|surprising) reason",   "shocking reason",             80),
    (r"you (?:should|need to|must) (?:see|know|watch|read|hear)", "you need to know", 75),
    (r"what you (?:need to|should|must) know",      "what you need to know",       75),

    # Listicles
    (r"\b\d+\s+(?:shocking|amazing|incredible|surprising|unbelievable|mind.?blowing|crazy|insane)\b", None, 85),
    (r"\b\d+\s+(?:reasons?|ways?|things?|facts?|secrets?|tips?|signs?|hacks?|tricks?|myths?|steps?)\b", None, 78),
    (r"\b(?:top|best)\s+\d+\b",                     None,                          72),
    (r"\b\d+\s+times?\b",                            None,                          65),

    # Emotional shock language
    (r"\bshock(?:ing|ed|er)?\b",                    None,                          78),
    (r"\bunbelievable\b",                            None,                          74),
    (r"\bmind.?blow(?:ing|s)?\b",                   "mind-blowing",                80),
    (r"\bheart.?break(?:ing)?\b",                   "heartbreaking",               72),
    (r"\boutrag(?:ed|ing|eous)\b",                  None,                          70),
    (r"\bfurious\b",                                 None,                          68),
    (r"\bdisgusting\b",                              None,                          72),
    (r"\bcrazy\b",                                   None,                          55),
    (r"\binsane\b",                                  None,                          58),
    (r"\bstunning\b",                                None,                          62),
    (r"\bincredible\b",                              None,                          60),
    (r"\bamazing\b",                                 None,                          50),

    # Attack/drama verbs used as clickbait
    (r"\bslam(?:s|med|ming)?\b",                    None,                          68),
    (r"\bdestroy(?:s|ed|ing)?\b",                   None,                          65),
    (r"\bcrush(?:es|ed|ing)?\b",                    None,                          60),
    (r"\bblast(?:s|ed|ing)?\b",                     None,                          60),
    (r"\bexpose(?:s|d|ing)?\b",                     None,                          70),
    (r"\bcaught on (?:camera|video|tape)\b",         "caught on camera",            82),
    (r"\bgoes? (?:viral|off on|ballistic|crazy|nuts|wild)\b", "goes viral/off",    78),
    (r"\blose(?:s|ing)?\s+it\b",                    "loses it",                    75),
    (r"\bmelts? down\b",                             "meltdown",                    75),
    (r"\bbreaks? the internet\b",                   "breaks the internet",          88),

    # FOMO / urgency bait
    (r"^\s*BREAKING\b",                              "BREAKING",                    65),
    (r"\burgent(?:ly)?\b",                           None,                          68),
    (r"\bbefore it'?s? too late\b",                 "before it's too late",         85),
    (r"\bmust.?(?:see|watch|read|know)\b",           "must-see/watch",              78),
    (r"\bdon'?t miss\b",                             "don't miss",                  72),

    # Engagement bait
    (r"\beveryone (?:should|needs? to|must)\b",      "everyone should",             72),
    (r"\bshare (?:this|if you)\b",                   "share this",                  70),
    (r"\bOMG\b",                                     "OMG",                         85),
    (r"\bWOW\b",                                     "WOW",                         80),
    (r"\bviral\b",                                   "viral",                       62),
    (r"\bexclusive\b",                               "exclusive",                   58),

    # Vague-pronoun clickbait ("This man did X")
    (r"\bthis (?:man|woman|guy|girl|dad|mom|teacher|doctor|nurse|kid|teen|student)\b", None, 60),

    # Misinformation / pseudoscience / conspiracy — these belong in CLICKBAIT
    # since we removed FAKE as a category
    (r"\bearth is flat\b",                               "earth is flat",               95),
    (r"\bflat earth\b",                                  "flat earth",                  95),
    (r"\bearth isn'?t (?:round|a sphere|spherical)\b",   "earth isn't round",           95),
    (r"\bmoon landing.{0,20}(?:fake|faked|hoax|staged)\b", "moon landing faked",        95),
    (r"\bvaccines? (?:cause|caused|causes?) autism\b",   "vaccines cause autism",       95),
    (r"\banti.?vax\b",                                   "anti-vax",                    82),
    (r"\bclimate.{0,15}(?:hoax|fake|lie|scam|conspiracy)\b", "climate hoax",            92),
    (r"\bstolen election\b",                             "stolen election",             92),
    (r"\bdeep state\b",                                  "deep state",                  90),
    (r"\b(?:a )?hoax\b",                                 "hoax",                        88),
    (r"\bconspiracy\b",                                  "conspiracy",                  85),
    (r"\bplandemic\b",                                   "plandemic",                   95),
    (r"\bchemitrail(?:s)?\b",                            "chemtrails",                  92),
    (r"\b(?:the )?illuminati\b",                         "illuminati",                  90),
    (r"\bnew world order\b",                             "new world order",             90),
    (r"\bsheeple\b",                                     "sheeple",                     90),
    (r"\bwake up (?:people|sheeple|america)\b",          "wake up sheeple",             88),
    (r"\bglobalist(?:s)?\b",                             "globalists",                  82),
    (r"\bsuppressed\b",                                  "suppressed",                  80),
    (r"\bpropaganda\b",                                  "propaganda",                  78),
    (r"\bscam\b",                                        "scam",                        82),
    (r"\bthey'?re (?:hiding|lying|covering)\b",          "they're hiding/lying",        88),
    (r"\bcover.?up\b",                                   "cover-up",                    85),
    (r"\bsecret agenda\b",                               "secret agenda",               88),
    (r"\b(?:mind|thought) control\b",                    "mind control",                90),
    (r"\b5G.{0,20}(?:dangerous|toxic|weapon|kill|cancer|control)\b", "5G dangerous",   92),
    (r"\bsatanic\b",                                     "satanic",                     88),
    (r"\bpedogate\b",                                    "pedogate",                    95),
    (r"\bmainstream media.{0,20}(?:lie|lied|lying|hiding|covers)\b", "MSM lies",       88),
    (r"\bthey don'?t want you to know\b",               "they don't want you to know", 90),
    (r"\bbig pharma\b",                                  "big pharma",                  78),
    (r"\bpseudoscience\b",                               "pseudoscience",               75),
    (r"\bmisinformation\b",                              "misinformation",              70),
]

# REAL / CREDIBILITY patterns — signal factual, journalistic reporting
REAL_SCORED: list[tuple[str, str | None, int]] = [
    # Attribution language (strongest signals)
    (r"\baccording to\b",                           "according to",                85),
    (r"\bannounced\b",                              None,                          75),
    (r"\bconfirmed\b",                              None,                          78),
    (r"\bdenied\b",                                 None,                          68),
    (r"\bstated\b",                                 None,                          65),
    (r"\bpublished\b",                              None,                          62),
    (r"\breported(?:ly)?\b",                        None,                          65),
    (r"\bspokesp(?:erson|eople)\b",                 "spokesperson",                82),
    (r"\bstatement\b",                              None,                          70),
    (r"\bsaid\b",                                   None,                          55),
    (r"\bresponded\b",                              None,                          62),

    # Research / data language
    (r"\bstudy (?:shows?|finds?|reveals?|suggests?)\b",    "study shows",         85),
    (r"\bresearch (?:shows?|finds?|reveals?|suggests?|indicates?)\b", "research finds", 85),
    (r"\bdata (?:shows?|reveals?|suggests?|indicates?)\b", "data shows",           80),
    (r"\bsurvey\b",                                 None,                          68),
    (r"\bpoll (?:shows?|finds?)\b",                 "poll finds",                  72),
    (r"\bstatistics?\b",                            None,                          75),
    (r"\banalysis\b",                               None,                          70),
    (r"\bfindings?\b",                              None,                          68),
    (r"\bevidence\b",                               None,                          72),
    (r"\breport(?:ed)?\b",                          None,                          60),

    # Institutions / authorities
    (r"\buniversity\b",                             None,                          78),
    (r"\bresearchers?\b",                           None,                          75),
    (r"\bscientists?\b",                            None,                          72),
    (r"\bgovernment\b",                             None,                          62),
    (r"\bparliament\b",                             None,                          70),
    (r"\bcongress\b",                               None,                          68),
    (r"\bsenate\b",                                 None,                          68),
    (r"\b(?:WHO|CDC|FDA|EPA|NIH|NATO|UN|EU)\b",     None,                          82),
    (r"\bofficial(?:s|ly)?\b",                      None,                          68),
    (r"\bminister\b",                               None,                          65),
    (r"\bjudge (?:ruled|ordered|said)\b",           "judge ruled",                 78),
    (r"\bcourt\b",                                  None,                          65),
    (r"\bpolice\b",                                 None,                          58),

    # Specific numbers as facts (e.g. "45% of", "$2.3 billion")
    (r"\b\d+(?:\.\d+)?%\s+of\b",                   None,                          62),
    (r"\$\d+(?:\.\d+)?\s*(?:billion|million|trillion)\b", None,                   65),

    # Hedged / measured language
    (r"\bsuggests?\b",                              None,                          60),
    (r"\bindicates?\b",                             None,                          60),
    (r"\bappears? to\b",                            None,                          55),
    (r"\blikely\b",                                 None,                          52),
    (r"\bmay\b",                                    None,                          48),
    (r"\bcould\b",                                  None,                          46),

    # Neutral reporting verbs
    (r"\bvoted\b",                                  None,                          62),
    (r"\belected\b",                                None,                          65),
    (r"\bappointed\b",                              None,                          62),
    (r"\barrested\b",                               None,                          65),
    (r"\bcharged\b",                                None,                          62),
    (r"\bconvicted\b",                              None,                          68),
    (r"\bpassed\b",                                 None,                          55),
    (r"\bsigned\b",                                 None,                          60),
    (r"\breleased\b",                               None,                          55),
    (r"\bapproved\b",                               None,                          60),
]

CAPS_THRESHOLD = 0.35


def caps_ratio(text: str) -> float:
    alpha = [c for c in text if c.isalpha()]
    if not alpha:
        return 0.0
    return sum(1 for c in alpha if c.isupper()) / len(alpha)


def score_patterns(text: str, patterns: list[tuple[str, str | None, int]]) -> float:
    """Sum severity scores for all matched patterns in text (case-insensitive)."""
    text_l = text.lower()
    return sum(score for p, _, score in patterns if re.search(p, text_l))


def extract_keywords(headline: str, body: Optional[str]) -> list[KeywordMatch]:
    """Return matched clickbait signals sorted by severity, highest first."""
    headline_lower = headline.lower()
    found: list[KeywordMatch] = []
    seen: set[str] = set()

    for pattern, label, score in CB_SCORED:
        m = re.search(pattern, headline_lower)
        if m:
            word = label if label else m.group(0).strip()
            if word.lower() not in seen:
                seen.add(word.lower())
                found.append(KeywordMatch(word=word, score=score))

    # Structural meta-signals
    if re.search(r"!!+", headline) and "!!!" not in seen:
        found.append(KeywordMatch(word="!!!", score=72))
    elif re.search(r"!$", headline) and "!" not in seen:
        found.append(KeywordMatch(word="!", score=42))
    if re.search(r"\?$", headline) and "?" not in seen:
        found.append(KeywordMatch(word="?", score=38))
    if caps_ratio(headline) > CAPS_THRESHOLD and "ALL CAPS" not in seen:
        found.append(KeywordMatch(word="ALL CAPS", score=72))

    found.sort(key=lambda k: k.score, reverse=True)
    return found[:12]


def extract_indicators(
    headline: str,
    body: Optional[str],
    p_cb: float,
    p_real: float,
    model_label: Optional[str],
) -> list[str]:
    indicators: list[str] = []
    h_lower = headline.lower()

    if p_cb > 0.5:
        indicators.append("Sensational or emotionally-charged language detected")
    if re.search(r"!{2,}", headline):
        indicators.append("Multiple exclamation marks detected")
    elif re.search(r"!$", headline):
        indicators.append("Headline ends with exclamation mark")
    if re.search(r"\b\d+\s+(?:reasons?|ways?|things?|facts?|tips?|signs?)\b", h_lower):
        indicators.append("Listicle-style headline (numbered bait)")
    if caps_ratio(headline) > CAPS_THRESHOLD:
        indicators.append("Excessive capital letters")
    # Check for top curiosity-gap patterns
    curiosity_patterns = [p for p, _, _ in CB_SCORED[:10]]
    if any(re.search(p, h_lower) for p in curiosity_patterns):
        indicators.append("Curiosity-gap framing detected")
    if re.search(r"\baccording to\b|\bstudy (?:shows?|finds?)\b|\bresearch\b|\bconfirmed\b", h_lower):
        indicators.append("Attribution or research reference in headline")
    if body:
        body_lower = body.lower()
        if re.search(r"\baccording to\b|\bstudy\b|\bresearch\b|\bofficial\b|\bconfirmed\b|\bspokesperson\b", body_lower):
            indicators.append("Article body contains credible source citations")
        if len(body.strip()) < 80:
            indicators.append("Very short article body — limited factual context")
    if model_label == "REAL":
        indicators.append("Language model: consistent with factual reporting style")
    elif model_label == "FAKE":
        indicators.append("Language model: diverges from typical credible-reporting patterns")

    return indicators if indicators else ["No strong signals detected — content appears neutral"]


def normalise_label(raw_label: str) -> str:
    u = raw_label.upper()
    if u in ("FAKE", "LABEL_0", "LABEL-0", "0"):
        return "FAKE"
    if u in ("REAL", "LABEL_1", "LABEL-1", "1", "TRUE"):
        return "REAL"
    return "REAL"


def predict(headline: str, body: Optional[str]) -> PredictResponse:
    combined = headline + (" " + body[:600] if body else "")
    headline_lower = headline.lower()

    # --- ML model: used only as a minor tiebreaker, NOT the primary classifier ---
    # The HuggingFace model is unreliable for short/unusual inputs; cap its effect.
    model_label: Optional[str] = None
    ml_nudge: float = 0.0  # adds at most ±0.08 to the real probability

    if classifier is not None:
        try:
            result = classifier(combined[:512])[0]
            model_label = normalise_label(result["label"])
            raw_ml_conf = float(result["score"])
            # Max nudge: ±0.08 regardless of model confidence
            if model_label == "REAL":
                ml_nudge = raw_ml_conf * 0.08
            else:
                ml_nudge = -raw_ml_conf * 0.08
        except Exception as exc:
            logger.warning(f"Inference error: {exc}")

    # --- Rule-based scoring (primary classifier) ---
    # Clickbait: headline is primary, body is secondary
    cb_hl = score_patterns(headline, CB_SCORED)
    cb_body = score_patterns(body[:400] if body else "", CB_SCORED) * 0.25
    cb_raw = cb_hl + cb_body

    # Structural boosts
    if caps_ratio(headline) > CAPS_THRESHOLD:
        cb_raw += 65
    if re.search(r"!!+", headline):
        cb_raw += 62
    elif re.search(r"!$", headline):
        cb_raw += 32
    if re.search(r"\?$", headline):
        cb_raw += 25

    # Real signals: headline + body
    real_hl = score_patterns(headline, REAL_SCORED)
    real_body = score_patterns(body[:600] if body else "", REAL_SCORED) * 0.6
    real_raw = real_hl + real_body

    # Normalize to [0, 1]
    p_cb_rule = min(cb_raw / 480.0, 1.0)
    p_real_rule = min(real_raw / 420.0, 1.0)

    has_signals = (p_cb_rule + p_real_rule) > 0.05

    if has_signals:
        # Rule-based dominates; ML nudges the real score slightly
        p_real_adj = max(0.0, min(1.0, p_real_rule + ml_nudge - p_cb_rule * 0.35))
        p_cb_adj = max(0.0, p_cb_rule)
        total = p_real_adj + p_cb_adj
        if total < 0.01:
            total = 1.0
        p_real_final = p_real_adj / total
        p_cb_final = p_cb_adj / total
    else:
        # No rule signals at all (plain neutral sentence like "the sky is blue")
        # Default: REAL at moderate confidence; ML nudge applies but stays modest
        base_real = 0.62 + ml_nudge
        p_real_final = max(0.45, min(0.78, base_real))
        p_cb_final = 1.0 - p_real_final

    verdict = "CLICKBAIT" if p_cb_final > 0.5 else "REAL"
    raw_conf = p_cb_final if verdict == "CLICKBAIT" else p_real_final

    # Scale confidence: strong signals → higher confidence; no signals → capped lower
    if has_signals:
        confidence = round(max(52.0, min(97.0, raw_conf * 100)), 1)
    else:
        # No clear signals — stay humble
        confidence = round(max(52.0, min(72.0, raw_conf * 100)), 1)

    indicators = extract_indicators(headline, body, p_cb_final, p_real_final, model_label)
    keywords = extract_keywords(headline, body)

    return PredictResponse(
        verdict=verdict,
        confidence=confidence,
        scores={
            "real": round(p_real_final * 100, 1),
            "clickbait": round(p_cb_final * 100, 1),
        },
        indicators=indicators,
        keywords=keywords,
        model_used=model_name,
    )


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
