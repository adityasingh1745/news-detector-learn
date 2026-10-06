"""
News Credibility & Clickbait Detection — Hybrid ML + Rule-Based Classifier
Classifies news as REAL (credible journalism) or CLICKBAIT (sensational/bait/misinfo).
Uses a TF-IDF + Logistic Regression model trained on a 32k-headline labeled
dataset (see train_model.py), blended with pattern-based heuristics.
Falls back to pure rule-based scoring if the trained model file is missing.
"""
import asyncio
import os
import re
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Optional
from urllib.parse import urlparse
import joblib
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "models", "clickbait_model.joblib")
DATABASE_URL = os.environ.get("DATABASE_URL")
# How often to check for new user feedback and retrain the model on it.
# Raised from 10 minutes to 6 hours now that the base dataset includes the
# full ~3.9M-row archive corpus — a full retrain takes ~30-40 minutes, so a
# short interval would cause retrains to pile up / constantly thrash CPU.
RETRAIN_INTERVAL_SECONDS = int(os.environ.get("RETRAIN_INTERVAL_SECONDS", "21600"))

model_name = "rule-based-v3"
model_ready = False

# Populated at startup if a trained model is found on disk.
ml_vectorizer = None
ml_classifier = None
model_feedback_examples_used = 0
model_last_retrained: Optional[str] = None
# Distillation progress: how many teacher-labeled (Gemini + Groq) analyses
# the model has been trained on, and how accurately it reproduces their
# combined labels on a held-out split of those examples. Once both
# thresholds below are met, the local model is considered a reliable
# enough standalone replacement for both APIs.
model_gemini_examples_used = 0
model_gemini_accuracy: Optional[float] = None
model_groq_examples_used = 0
model_groq_accuracy: Optional[float] = None
model_teacher_examples_used = 0
model_teacher_accuracy: Optional[float] = None
GEMINI_GRADUATION_MIN_EXAMPLES = int(os.environ.get("GEMINI_GRADUATION_MIN_EXAMPLES", "300"))
GEMINI_GRADUATION_MIN_ACCURACY = float(os.environ.get("GEMINI_GRADUATION_MIN_ACCURACY", "0.92"))


def _is_ready_for_local_only() -> bool:
    """True once the local model has distilled enough teacher-labeled
    (Gemini + Groq combined) examples and reproduces their judgment
    accurately enough that neither API is needed for new classifications."""
    return (
        model_teacher_examples_used >= GEMINI_GRADUATION_MIN_EXAMPLES
        and model_teacher_accuracy is not None
        and model_teacher_accuracy >= GEMINI_GRADUATION_MIN_ACCURACY
    )

# Weight given to the ML model vs. the rule-based heuristics when blending
# scores (matches the architecture described in replit.md).
ML_WEIGHT = 0.6
RULE_WEIGHT = 1.0 - ML_WEIGHT

# ── Real-world fact-checking via NewsAPI.org ────────────────────────────────
# Verifies whether a claim has any actual corroborating news coverage. This
# catches confidently-worded but fabricated claims (e.g. "2036 Olympics will
# be hosted in Bangladesh") that pure style/pattern analysis cannot detect.
NEWS_API_KEY = os.environ.get("NEWS_API_KEY")
NEWS_API_URL = "https://newsapi.org/v2/everything"

# Low-quality / self-published domains that should never be surfaced as
# "evidence" even if they happen to match search terms — personal blogs,
# free blogging platforms, and known misinformation mirrors are not
# credible corroboration for a factual claim.
_LOW_QUALITY_DOMAINS = {
    "wordpress.com", "blogspot.com", "substack.com", "medium.com",
    "tumblr.com", "weebly.com", "wixsite.com", "sites.google.com",
    "blogger.com", "livejournal.com",
}


def _is_low_quality_source(url: Optional[str]) -> bool:
    if not url:
        return False
    host = urlparse(url).netloc.lower()
    host = host[4:] if host.startswith("www.") else host
    return any(host == d or host.endswith("." + d) for d in _LOW_QUALITY_DOMAINS)

_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "will", "would",
    "could", "should", "of", "in", "on", "at", "to", "for", "and", "or", "but",
    "with", "by", "from", "as", "this", "that", "these", "those", "it", "its",
    "his", "her", "their", "your", "our", "my", "has", "have", "had", "not",
    "going", "gonna", "new", "just", "now", "get", "gets", "getting",
}


def _extract_query_terms(headline: str, max_terms: int = 6) -> list[str]:
    """Pick out the most distinctive words (proper nouns, numbers) to search for."""
    words = re.findall(r"[A-Za-z0-9']+", headline)
    important = [w for w in words if w.isdigit() or (w[:1].isupper() and w.lower() not in _STOPWORDS)]
    if len(important) >= 2:
        return important[:max_terms]
    return [w for w in words if w.lower() not in _STOPWORDS][:max_terms]


def _extract_entities_and_content(headline: str) -> tuple[list[str], list[str]]:
    """Split headline words into (entities, content_words) — see _extract_search_terms."""
    words = re.findall(r"[A-Za-z0-9']+", headline)
    entities = [w for w in words if w.isdigit() or (w[:1].isupper() and w.lower() not in _STOPWORDS)]
    content_words = [
        w for w in words
        if w.lower() not in _STOPWORDS and w not in entities and len(w) > 2
    ]
    return entities, content_words


def _extract_search_terms(headline: str, max_terms: int = 6) -> list[str]:
    """
    Build the set of words used both to search NewsAPI *and* to verify
    corroboration. Deliberately includes proper nouns/numbers ("USA", "2028",
    "Rahul", "Gandhi") *and* the key non-filler content words ("olympics",
    "host", "dead", "resigns") — not just capitalized entities. Requiring the
    actual topic/claim word to be part of the match is what lets us tell
    "real event about a real entity" apart from "hoax claim riding on a real
    entity's name" (e.g. a politician's name trending without any article
    ever mentioning the claimed event).
    """
    entities, content_words = _extract_entities_and_content(headline)
    # Entities first (who/what), then the most distinctive remaining content
    # words (longer words tend to be more specific/meaningful), deduped.
    content_words = sorted(content_words, key=len, reverse=True)
    seen = {w.lower() for w in entities}
    combined = list(entities)
    for w in content_words:
        if w.lower() not in seen:
            combined.append(w)
            seen.add(w.lower())
        if len(combined) >= max_terms:
            break
    return combined


# Subjective opinion/value-judgment words. Headlines built around these
# ("X is bad for the world", "Y is the greatest country") are not factual
# claims that news coverage can corroborate or refute — bag-of-words article
# overlap on the surrounding entity/topic words can accidentally "match" a
# totally unrelated article and produce a misleading fact-check verdict.
# When detected, skip NewsAPI corroboration entirely and let the ML/rules
# model judge the phrasing/style on its own merits.
_OPINION_WORDS = {
    "bad", "terrible", "awful", "evil", "disgusting", "horrible",
    "ugly", "stupid", "corrupt", "toxic", "dangerous", "harmful",
}

# Subjective *skill/talent* praise about an individual person ("the finest
# cover drive player", "the greatest footballer ever") is unfact-checkable —
# no news database can confirm who is "the best" at something subjective.
# This is distinct from claims about institutions/measurable rankings
# (e.g. "best healthcare system"), which real surveys/reports DO cover and
# should still go through NewsAPI corroboration.
_SKILL_OPINION_PATTERN = re.compile(
    r"\b(finest|greatest|best|worst|most talented|most skilled|legendary|goat)\b"
    r".{0,35}\b(player|cricketer|batsman|batter|bowler|footballer|striker|"
    r"goalkeeper|driver|racer|singer|actor|actress|artist|chef|dancer|"
    r"athlete|golfer|boxer|wrestler|sportsman|sportswoman)\b",
    re.IGNORECASE,
)


def _is_opinion_headline(headline: str) -> bool:
    words = {w.lower() for w in re.findall(r"[A-Za-z']+", headline)}
    if words & _OPINION_WORDS:
        return True
    return bool(_SKILL_OPINION_PATTERN.search(headline))


async def fact_check_headline(headline: str) -> Optional[dict]:
    """
    Search NewsAPI.org for real coverage of this claim.
    Returns None when fact-checking is unavailable/inconclusive (no adjustment
    should be made), or a dict describing the corroboration status.
    """
    if not NEWS_API_KEY:
        return None

    if _is_opinion_headline(headline):
        return None  # subjective claim — news corroboration doesn't apply

    terms = _extract_search_terms(headline)
    if len(terms) < 2:
        return None  # not enough distinctive terms to search meaningfully

    query = " ".join(terms)
    try:
        async with httpx.AsyncClient(timeout=6.0) as client:
            resp = await client.get(
                NEWS_API_URL,
                params={
                    "q": query,
                    "language": "en",
                    "sortBy": "relevancy",
                    "pageSize": 8,
                    "apiKey": NEWS_API_KEY,
                },
            )
        if resp.status_code != 200:
            logger.warning("NewsAPI error %s: %s", resp.status_code, resp.text[:200])
            return None
        data = resp.json()
    except Exception:
        logger.exception("NewsAPI request failed")
        return None

    articles = data.get("articles", []) or []
    total_results = data.get("totalResults", 0)
    term_set = {t.lower() for t in terms}
    # Numbers (years, counts) are highly distinctive — if the headline names
    # one, a candidate article must actually mention it to be considered a
    # genuine match, not just an incidental overlap on generic words.
    required_numbers = {t.lower() for t in terms if t.isdigit()}

    best_overlap = 0.0
    best_has_required_numbers = True
    top_source = None
    top_article = None
    # Track the best candidate that actually mentions required numbers
    # separately — it should win over a higher-raw-overlap article that
    # omits a distinctive year/count, since the number is what makes the
    # claim specific in the first place.
    best_overlap_with_numbers = -1.0
    top_source_with_numbers = None
    top_article_with_numbers = None
    for a in articles:
        if _is_low_quality_source(a.get("url")):
            continue  # never surface personal blogs/self-published posts as evidence
        title = (a.get("title") or "").lower()
        desc = (a.get("description") or "").lower()
        text = f"{title} {desc}"
        hits = sum(1 for t in term_set if t in text)
        overlap = hits / len(term_set) if term_set else 0.0
        has_required_numbers = all(n in text for n in required_numbers)
        if overlap > best_overlap:
            best_overlap = overlap
            best_has_required_numbers = has_required_numbers
            top_source = (a.get("source") or {}).get("name")
            top_article = {
                "title": a.get("title"),
                "url": a.get("url"),
                "publisher": top_source,
            }
        if has_required_numbers and overlap > best_overlap_with_numbers:
            best_overlap_with_numbers = overlap
            top_source_with_numbers = (a.get("source") or {}).get("name")
            top_article_with_numbers = {
                "title": a.get("title"),
                "url": a.get("url"),
                "publisher": top_source_with_numbers,
            }

    if required_numbers and top_article_with_numbers is not None:
        # A more specific (number-matching) candidate exists — prefer it for
        # display even if a generic-word match scored marginally higher.
        top_source = top_source_with_numbers
        top_article = top_article_with_numbers
        best_has_required_numbers = True

    # Always surface whatever article we found (already filtered for
    # low-quality domains and, when relevant, preferring one that actually
    # mentions a distinctive number/year) — users asked to be able to check
    # the evidence themselves rather than have us silently hide weak
    # matches. We still flag weak matches so the label can be honest about
    # how confident this specific match is.
    displayable_article = top_article
    weak_match = best_overlap < 0.3 or (required_numbers and not best_has_required_numbers)

    if total_results == 0:
        return {"status": "no_coverage", "articles_found": 0, "top_source": None, "top_article": None}

    if best_overlap >= 0.6 and best_has_required_numbers:
        return {
            "status": "corroborated", "articles_found": total_results,
            "top_source": top_source, "top_article": displayable_article, "weak_match": weak_match,
        }

    if best_overlap >= 0.35 and total_results < 50:
        # A genuinely close (if imperfect) match exists — e.g. a headline
        # phrased slightly differently than the matching article's title.
        # This is a positive but not fully conclusive signal, so it gets a
        # mild nudge toward REAL rather than the full "corroborated" boost.
        return {
            "status": "partially_corroborated", "articles_found": total_results,
            "top_source": top_source, "top_article": displayable_article, "weak_match": weak_match,
        }

    if total_results >= 50:
        # The overall topic/entity is heavily covered, but no returned
        # article actually matches this specific combination of terms — a
        # classic sign of a fabricated/hoax claim riding on a real, famous
        # entity's name (e.g. a politician's "death" claim with no obituary
        # anywhere despite thousands of unrelated articles about them).
        return {
            "status": "unconfirmed_claim", "articles_found": total_results,
            "top_source": top_source, "top_article": displayable_article, "weak_match": weak_match,
        }

    return {
        "status": "inconclusive", "articles_found": total_results,
        "top_source": top_source, "top_article": displayable_article, "weak_match": weak_match,
    }


# ── Professional fact-check lookup via Google Fact Check Tools API ─────────
# Searches published fact-checks (PolitiFact, Snopes, Reuters Fact Check,
# etc.) for a claim matching this headline. This is a much stronger signal
# than "does related news coverage exist" (NewsAPI above) — it's an actual
# human-reviewed verdict on this specific claim, when one exists.
GOOGLE_FACTCHECK_API_KEY = os.environ.get("GOOGLE_FACTCHECK_API_KEY")
GOOGLE_FACTCHECK_URL = "https://factchecktools.googleapis.com/v1alpha1/claims:search"

_FALSE_RATING_WORDS = {
    "false", "fake", "pants on fire", "incorrect", "misleading", "hoax",
    "fabricated", "unproven", "no evidence", "unsubstantiated", "debunked",
    "not true", "mostly false", "distorts",
}
_TRUE_RATING_WORDS = {
    "true", "correct", "accurate", "confirmed", "verified", "mostly true",
    "real",
}


def _rating_to_status(rating: str) -> Optional[str]:
    r = rating.lower().strip()
    # Check false-ish phrases first since "mostly false" etc. contain "false"
    # but some ratings mix words (e.g. "half true") — order matters.
    if any(w in r for w in _FALSE_RATING_WORDS):
        return "false"
    if any(w in r for w in _TRUE_RATING_WORDS):
        return "true"
    if "half" in r or "mixture" in r or "mixed" in r or "partly" in r:
        return "mixed"
    return None


async def google_fact_check(headline: str) -> Optional[dict]:
    """
    Look up this claim in Google's Fact Check Tools index. Returns None if
    unavailable/no matching claim found, otherwise a dict with a normalized
    status ("false" | "true" | "mixed") and the reviewing publisher/rating.
    """
    if not GOOGLE_FACTCHECK_API_KEY:
        return None

    terms = _extract_search_terms(headline, max_terms=8)
    if len(terms) < 2:
        return None
    query = " ".join(terms)

    try:
        async with httpx.AsyncClient(timeout=6.0) as client:
            resp = await client.get(
                GOOGLE_FACTCHECK_URL,
                params={"query": query, "languageCode": "en", "key": GOOGLE_FACTCHECK_API_KEY},
            )
        if resp.status_code != 200:
            logger.warning("Google Fact Check API error %s: %s", resp.status_code, resp.text[:200])
            return None
        data = resp.json()
    except Exception:
        logger.exception("Google Fact Check API request failed")
        return None

    claims = data.get("claims", []) or []
    entities, content_words = _extract_entities_and_content(headline)
    term_set = {t.lower() for t in terms}
    entity_set = {e.lower() for e in entities}
    content_set = {c.lower() for c in content_words}
    best_match = None
    best_overlap = 0.0
    for c in claims:
        claim_text = (c.get("text") or "").lower()
        hits = sum(1 for t in term_set if t in claim_text)
        overlap = hits / len(term_set) if term_set else 0.0
        reviews = c.get("claimReview") or []
        if not reviews:
            continue

        # Guard against matching on shared entity mentions alone (e.g. a
        # claim that starts "Former Indian captain Mahendra Singh Dhoni..."
        # then disputes something totally unrelated, like a religious
        # conversion rumor, while our headline just states his real,
        # uncontested role). Find where the entity mentions end in the
        # claim text and require our headline's own distinctive content
        # word(s) — the actual assertion, not just who it's about — to
        # appear in the remaining "predicate" text after that point.
        if content_set:
            last_entity_end = 0
            for e in entity_set:
                idx = claim_text.rfind(e)
                if idx != -1:
                    last_entity_end = max(last_entity_end, idx + len(e))
            predicate_segment = claim_text[last_entity_end:]
            # If the segment is too short to be meaningful (e.g. an entity
            # word happens to sit right at the end of the sentence), fall
            # back to checking the full claim text rather than risk a false
            # rejection from an unlucky word order.
            search_space = predicate_segment if len(predicate_segment) >= 15 else claim_text
            if not any(w in search_space for w in content_set):
                continue  # this claim disputes a different assertion entirely

        if overlap > best_overlap:
            best_overlap = overlap
            best_match = (c, reviews[0])

    # Require the matched claim's text to substantially overlap our headline
    # so we don't apply an unrelated fact-check to this claim.
    if not best_match or best_overlap < 0.5:
        return None

    claim, review = best_match
    rating_text = review.get("textualRating", "") or ""
    status = _rating_to_status(rating_text)
    if status is None:
        return None

    return {
        "status": status,
        "rating": rating_text,
        "publisher": (review.get("publisher") or {}).get("name", "a fact-checker"),
        "url": review.get("url"),
    }


# ── Biographical grounding via Wikipedia's free REST API ───────────────────
# NewsAPI/Google Fact Check need *recent* coverage of a claim to work — but
# a huge share of fabricated headlines are about well-known people and don't
# need "breaking news" at all (is this person alive? what was their actual
# role?). Wikipedia is free, keyless, comprehensive for public figures, and
# kept current, so it's a strong, independent second opinion specifically
# for these biographical claims.
_WIKI_SUMMARY_URL = "https://en.wikipedia.org/api/rest_v1/page/summary/{}"
_WIKI_SEARCH_URL = "https://en.wikipedia.org/w/api.php"
_WIKI_HEADERS = {"User-Agent": "clickbait-detector/1.0 (contact@example.com)"}

_DEATH_CLAIM_WORDS = {"dead", "died", "death", "deceased", "passed"}
_ALIVE_CLAIM_WORDS = {"alive", "living"}
_WIKI_DEATH_INDICATORS = ("died", "death", "assassinat", "passed away", "deceased")


def _extract_proper_noun_phrase(headline: str) -> Optional[str]:
    """Find the longest run of consecutive capitalized words — the likely subject entity."""
    words = headline.split()
    best: list[str] = []
    current: list[str] = []
    for raw in words:
        w = re.sub(r"[^\w'-]", "", raw)
        is_cap = bool(w) and w[0].isupper() and w.lower() not in _STOPWORDS
        if is_cap:
            current.append(w)
            if len(current) > len(best):
                best = current[:]
        else:
            current = []
    return " ".join(best) if best else None


async def _wiki_fetch_summary(title: str) -> Optional[dict]:
    try:
        async with httpx.AsyncClient(timeout=6.0, headers=_WIKI_HEADERS) as client:
            resp = await client.get(_WIKI_SUMMARY_URL.format(title.replace(" ", "_")))
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        logger.exception("Wikipedia summary request failed")
    return None


async def _wikidata_has_death_date(qid: Optional[str]) -> Optional[bool]:
    """
    Query Wikidata's structured "date of death" property (P570) for this
    entity. This is far more reliable than text-mining the Wikipedia summary
    extract — for long, eventful biographies (heads of state, historical
    figures) the death is often mentioned well past where the short summary
    extract gets cut off, even though the structured data is always present.
    Returns True/False when the entity is a known person with this property
    resolvable, or None if it couldn't be determined either way.
    """
    if not qid:
        return None
    try:
        async with httpx.AsyncClient(timeout=6.0, headers=_WIKI_HEADERS) as client:
            resp = await client.get(
                "https://www.wikidata.org/w/api.php",
                params={
                    "action": "wbgetclaims", "entity": qid, "property": "P570",
                    "format": "json",
                },
            )
        if resp.status_code != 200:
            return None
        claims = resp.json().get("claims", {})
        return bool(claims.get("P570"))
    except Exception:
        logger.exception("Wikidata claims request failed")
        return None


async def wikipedia_check(headline: str) -> Optional[dict]:
    """
    Ground biographical claims (alive/dead status, occupation/role) against
    Wikipedia. Returns None when no relevant, confidently-matched article
    exists, otherwise a dict with status "contradicted" | "confirmed" |
    "related" plus a human-readable note and source link.
    """
    subject = _extract_proper_noun_phrase(headline)
    if not subject or len(subject) < 3:
        return None

    data = await _wiki_fetch_summary(subject)
    if data is None:
        # Try Wikipedia's own search to resolve nicknames/alternate spellings.
        try:
            async with httpx.AsyncClient(timeout=6.0, headers=_WIKI_HEADERS) as client:
                resp = await client.get(_WIKI_SEARCH_URL, params={
                    "action": "query", "list": "search", "srsearch": subject,
                    "format": "json", "srlimit": 1,
                })
            if resp.status_code == 200:
                hits = resp.json().get("query", {}).get("search", [])
                if hits:
                    data = await _wiki_fetch_summary(hits[0]["title"])
        except Exception:
            logger.exception("Wikipedia search request failed")

    if not data or data.get("type") == "disambiguation":
        return None

    title = data.get("title") or subject
    extract = (data.get("extract") or "").lower()
    description = (data.get("description") or "").lower()
    page_url = (data.get("content_urls", {}) or {}).get("desktop", {}).get("page")
    qid = data.get("wikibase_item")
    if not extract:
        return None

    # Guard against a bad search match resolving to an unrelated page — the
    # resolved title must share a meaningful word with our extracted subject.
    if not any(w in title.lower() for w in subject.lower().split() if len(w) > 2):
        return None

    headline_lower = headline.lower()
    claims_dead = any(w in headline_lower for w in _DEATH_CLAIM_WORDS)
    claims_alive = any(w in headline_lower for w in _ALIVE_CLAIM_WORDS) and not claims_dead

    if claims_dead or claims_alive:
        # Wikidata's structured "date of death" property is authoritative —
        # unlike the short summary extract, it doesn't depend on the death
        # being mentioned before the intro paragraph gets cut off.
        wikidata_dead = await _wikidata_has_death_date(qid)
        if wikidata_dead is not None:
            wiki_reports_death = wikidata_dead
        else:
            # Fall back to text-based signals if Wikidata was unreachable.
            has_death_year_range = bool(re.search(r"\(\s*\d{3,4}\s*[-–—]\s*\d{3,4}\s*\)", description))
            wiki_reports_death = has_death_year_range or any(ind in extract for ind in _WIKI_DEATH_INDICATORS)
    else:
        wiki_reports_death = any(ind in extract for ind in _WIKI_DEATH_INDICATORS)

    if claims_dead and not wiki_reports_death:
        return {
            "status": "contradicted",
            "note": f"Wikipedia's article on {title} does not report them as deceased",
            "title": title, "url": page_url,
        }
    if claims_dead and wiki_reports_death:
        return {
            "status": "confirmed",
            "note": f"Wikipedia confirms {title} is deceased",
            "title": title, "url": page_url,
        }
    if claims_alive and wiki_reports_death:
        return {
            "status": "contradicted",
            "note": f"Wikipedia reports {title} as deceased",
            "title": title, "url": page_url,
        }

    # Occupation/role corroboration — only ever a positive nudge, never a
    # contradiction (a short Wikipedia description can't list every valid
    # role/title a person has held, so absence isn't evidence of falsehood).
    _, content_words = _extract_entities_and_content(headline)
    remainder_words = {w.lower() for w in content_words if len(w) > 3}
    bio_text = f"{description} {extract}"
    if remainder_words and any(w in bio_text for w in remainder_words):
        return {
            "status": "related",
            "note": f"Wikipedia's article on {title} corroborates this description",
            "title": title, "url": page_url,
        }

    return None


class PredictRequest(BaseModel):
    headline: str
    body: Optional[str] = None


class KeywordMatch(BaseModel):
    word: str
    score: int


class SourceRef(BaseModel):
    label: str          # e.g. "Fact-check" or "News coverage"
    title: str
    url: Optional[str] = None
    publisher: Optional[str] = None


class StyleAnalysis(BaseModel):
    """
    Tier 1: writing-style analysis — rule-based heuristics + the TF-IDF/
    structural-features ML model only. Answers "does this *read* like
    clickbait?" and is completely independent of whether the claim is true.
    """
    verdict: str                 # CLICKBAIT_STYLE | NEUTRAL_STYLE | CREDIBLE_STYLE
    clickbait_score: float        # 0-100
    credible_score: float         # 0-100
    indicators: list[str]
    keywords: list[KeywordMatch]
    model_used: str


class FactCheckAnalysis(BaseModel):
    """
    Tier 2: real-world corroboration — NewsAPI coverage search, professional
    fact-checker ratings (Google Fact Check), and Wikipedia grounding.
    Answers "did this actually happen?" and never looks at headline phrasing.
    """
    status: str                   # see _FACTCHECK_STATUSES below
    note: Optional[str] = None
    sources: list[SourceRef] = []


class PredictResponse(BaseModel):
    verdict: str
    confidence: float
    scores: dict
    indicators: list[str]
    keywords: list[KeywordMatch]
    sources: list[SourceRef] = []
    model_used: str
    style_analysis: Optional[StyleAnalysis] = None
    fact_check: Optional[FactCheckAnalysis] = None


class StatusResponse(BaseModel):
    ready: bool
    model_name: str
    feedback_count: int
    last_retrained: Optional[str]
    gemini_examples_used: int = 0
    gemini_accuracy: Optional[float] = None
    groq_examples_used: int = 0
    groq_accuracy: Optional[float] = None
    teacher_examples_used: int = 0
    teacher_accuracy: Optional[float] = None
    ready_for_local_only: bool = False


def _load_model_from_disk() -> None:
    """(Re)load the trained model artifact from disk into memory."""
    global model_name, ml_vectorizer, ml_classifier, model_feedback_examples_used, model_last_retrained
    global model_gemini_examples_used, model_gemini_accuracy
    global model_groq_examples_used, model_groq_accuracy
    global model_teacher_examples_used, model_teacher_accuracy
    if not os.path.exists(MODEL_PATH):
        logger.info("No trained model found at %s — using rule-based classifier only", MODEL_PATH)
        return
    try:
        bundle = joblib.load(MODEL_PATH)
        ml_vectorizer = bundle["vectorizer"]
        ml_classifier = bundle["classifier"]
        model_name = f"{bundle.get('model_name', 'tfidf-logreg-v1')}+rules"
        model_feedback_examples_used = bundle.get("feedback_examples_used", 0)
        model_gemini_examples_used = bundle.get("gemini_examples_used", 0)
        model_gemini_accuracy = bundle.get("gemini_accuracy")
        model_groq_examples_used = bundle.get("groq_examples_used", 0)
        model_groq_accuracy = bundle.get("groq_accuracy")
        model_teacher_examples_used = bundle.get(
            "teacher_examples_used", model_gemini_examples_used + model_groq_examples_used
        )
        model_teacher_accuracy = bundle.get("teacher_accuracy")
        trained_at = bundle.get("trained_at")
        model_last_retrained = (
            datetime.fromtimestamp(trained_at, tz=timezone.utc).isoformat() if trained_at else None
        )
        logger.info(
            "Loaded model from %s (test accuracy=%.4f, feedback_examples=%d, "
            "gemini_examples=%d, gemini_accuracy=%s, groq_examples=%d, groq_accuracy=%s, teacher_accuracy=%s)",
            MODEL_PATH, bundle.get("accuracy", float("nan")), model_feedback_examples_used,
            model_gemini_examples_used,
            f"{model_gemini_accuracy:.4f}" if model_gemini_accuracy is not None else "n/a",
            model_groq_examples_used,
            f"{model_groq_accuracy:.4f}" if model_groq_accuracy is not None else "n/a",
            f"{model_teacher_accuracy:.4f}" if model_teacher_accuracy is not None else "n/a",
        )
    except Exception:
        logger.exception("Failed to load trained model, falling back to rule-based only")
        ml_vectorizer = None
        ml_classifier = None
        model_name = "rule-based-v3"


# ── Contextual embeddings (fine-tuned DistilBERT) ───────────────────────────
# Optional upgrade over pure TF-IDF: a Transformer actually understands
# sentence semantics, which generalizes better to headlines whose *phrasing*
# the TF-IDF model has seen before but whose *vocabulary* (e.g. new
# entities/names) it hasn't. Loaded only if a fine-tuned artifact exists on
# disk (see train_distilbert.py) — the service runs fine without it, falling
# back to the TF-IDF+structural+rules pipeline exactly as before.
TRANSFORMER_MODEL_DIR = os.path.join(HERE, "models", "distilbert_clickbait")
TRANSFORMER_MAX_LENGTH = 48
transformer_tokenizer = None
transformer_model = None
transformer_ready = False


def _load_transformer_model() -> None:
    """Load the fine-tuned DistilBERT model if a trained artifact exists on disk."""
    global transformer_tokenizer, transformer_model, transformer_ready, model_name
    if not os.path.isfile(os.path.join(TRANSFORMER_MODEL_DIR, "config.json")):
        logger.info("No fine-tuned DistilBERT model found at %s — skipping", TRANSFORMER_MODEL_DIR)
        return
    try:
        from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast
        transformer_tokenizer = DistilBertTokenizerFast.from_pretrained(TRANSFORMER_MODEL_DIR)
        transformer_model = DistilBertForSequenceClassification.from_pretrained(TRANSFORMER_MODEL_DIR)
        transformer_model.eval()
        transformer_ready = True
        model_name = f"{model_name}+distilbert"
        logger.info("Loaded fine-tuned DistilBERT model from %s", TRANSFORMER_MODEL_DIR)
    except Exception:
        logger.exception("Failed to load DistilBERT model — continuing without it")
        transformer_tokenizer = None
        transformer_model = None
        transformer_ready = False


def transformer_predict_proba(headline: str) -> Optional[float]:
    """Return P(clickbait) from the fine-tuned DistilBERT model, or None if unavailable."""
    if not transformer_ready or transformer_model is None or transformer_tokenizer is None:
        return None
    try:
        import torch
        with torch.no_grad():
            enc = transformer_tokenizer(
                headline, truncation=True, padding="max_length",
                max_length=TRANSFORMER_MAX_LENGTH, return_tensors="pt",
            )
            logits = transformer_model(**enc).logits
            probs = torch.softmax(logits, dim=-1)[0]
            return float(probs[1].item())
    except Exception:
        logger.exception("DistilBERT inference failed — falling back to TF-IDF model only")
        return None


async def _retrain_loop() -> None:
    """
    Background task: periodically retrains the model on the original dataset
    plus any newly accumulated, majority-agreed user feedback, then hot-swaps
    the in-memory model — real continuous learning instead of a one-time
    static artifact.
    """
    import asyncio
    from retrain import retrain as retrain_model

    while True:
        await asyncio.sleep(RETRAIN_INTERVAL_SECONDS)
        if not DATABASE_URL:
            continue
        try:
            await asyncio.to_thread(retrain_model, DATABASE_URL)
            _load_model_from_disk()
            _load_transformer_model()
        except Exception:
            logger.exception("Scheduled retraining failed — keeping previous model")


@asynccontextmanager
async def lifespan(app: FastAPI):
    import asyncio
    global model_ready
    _load_model_from_disk()
    _load_transformer_model()
    model_ready = True
    task = asyncio.create_task(_retrain_loop())
    yield
    task.cancel()


app = FastAPI(title="News Credibility ML Service", version="3.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ---------------------------------------------------------------------------
# Pattern lists — (regex, display_label, severity 0-100)
# label=None  →  use matched text as the keyword label
# ---------------------------------------------------------------------------

# ── CLICKBAIT / MISINFORMATION SIGNALS ──────────────────────────────────────
CB_SCORED: list[tuple[str, str | None, int]] = [

    # 1. Curiosity-gap bait (highest severity)
    (r"you won'?t believe",                              "you won't believe",            95),
    (r"will blow your mind",                             "will blow your mind",          95),
    (r"the reason will shock you",                       "the reason will shock you",    92),
    (r"you need to see this",                            "you need to see this",         90),
    (r"this (?:one )?(?:weird|simple|old|ancient) trick","this one weird trick",         93),
    (r"doctors? (?:hate|won'?t tell) (?:him|her|you|this)", "doctors hate this",        93),
    (r"what (?:happens?|happened) next",                 "what happens next",            88),
    (r"what nobody (?:tells?|told) you",                 "what nobody tells you",        88),
    (r"they don'?t want you to know",                    "they don't want you to know",  92),
    (r"(?:can|could) you believe",                       "can you believe",              82),
    (r"nobody (?:expected|saw|believes?) this",          "nobody expected this",         85),
    (r"find out (?:why|how|what)",                       "find out why/how",             80),
    (r"the (?:shocking|real|hidden|untold|dark) truth",  "the hidden truth",             85),
    (r"what (?:really|actually) happened?",              "what really happened",         80),
    (r"secrets? (?:of|to|about|behind)",                 "secret of/about",              75),
    (r"(?:here'?s?|this is) why\b",                      "here's why",                   60),
    (r"what (?:your|the) .{0,30} (?:doesn'?t|won'?t) (?:tell|want)", "what they won't tell", 85),
    (r"you (?:should|need to|must) (?:see|know|watch|read|hear)", "you need to know",    78),
    (r"what you (?:need to|should|must) know",           "what you need to know",        78),
    (r"you (?:won'?t|wouldn'?t) guess",                  "you wouldn't guess",           85),
    (r"I (?:can'?t believe|never expected)",             "I can't believe",              80),

    # 2. Listicles
    (r"\b\d+\s+(?:shocking|amazing|incredible|surprising|unbelievable|mind.?blowing|crazy|insane)\b", None, 87),
    (r"\b\d+\s+(?:reasons?|ways?|things?|facts?|secrets?|tips?|signs?|hacks?|tricks?|myths?|steps?|foods?|habits?|mistakes?|questions?)\b", None, 78),
    (r"\b(?:top|best|worst)\s+\d+\b",                    None,                           72),
    (r"\b\d+\s+times?\b",                                None,                           65),
    (r"\b\d+\s+(?:types?|kinds?) of\b",                  None,                           62),

    # 3. Emotional shock / outrage language
    (r"\bshock(?:ing|ed|er)?\b",                         None,                           78),
    (r"\bunbelievable\b",                                 None,                           74),
    (r"\bmind.?blow(?:ing|s)?\b",                        "mind-blowing",                 80),
    (r"\bheart.?break(?:ing)?\b",                        "heartbreaking",                72),
    (r"\boutrag(?:ed|ing|eous)\b",                       None,                           72),
    (r"\bfurious\b",                                     None,                           70),
    (r"\bdisgusting\b",                                  None,                           74),
    (r"\bcrazy\b",                                       None,                           55),
    (r"\binsane\b",                                      None,                           60),
    (r"\bterrif(?:ying|ied)\b",                          None,                           74),
    (r"\bhorr(?:ifying|ified|ific)\b",                   "horrifying",                   76),
    (r"\bnightmare\b",                                   None,                           68),
    (r"\bdevastating\b",                                 None,                           65),
    (r"\bappalling\b",                                   None,                           70),
    (r"\bsickening\b",                                   None,                           72),
    (r"\bstunning\b",                                    None,                           62),
    (r"\bincredible\b",                                  None,                           60),
    (r"\bamazing\b",                                     None,                           50),

    # 4. Attack / drama verbs used as engagement bait
    (r"\bslam(?:s|med|ming)?\b",                         None,                           70),
    (r"\bdestroy(?:s|ed|ing)?\b",                        None,                           68),
    (r"\bcrush(?:es|ed|ing)?\b",                         None,                           62),
    (r"\bblast(?:s|ed|ing)?\b",                          None,                           62),
    (r"\bexpose(?:s|d|ing)?\b",                          None,                           72),
    (r"\bhumiliat(?:e|es|ed|ing)\b",                     None,                           74),
    (r"\bowned?\b",                                      None,                           65),
    (r"\btriggered?\b",                                  None,                           68),
    (r"\bcaught on (?:camera|video|tape)\b",             "caught on camera",             84),
    (r"\bgoes? (?:viral|off on|ballistic|crazy|nuts|wild)\b", "goes viral/off",          80),
    (r"\blose(?:s|ing)?\s+it\b",                         "loses it",                     78),
    (r"\bmelts? down\b",                                 "meltdown",                     78),
    (r"\bbreaks? the internet\b",                        "breaks the internet",           90),
    (r"\bsavaged?\b",                                    None,                           65),
    (r"\bwrecked?\b",                                    None,                           65),
    (r"\bsilenced?\b",                                   None,                           67),
    (r"\bRIP\b",                                         None,                           60),

    # 5. FOMO / urgency bait
    (r"^\s*BREAKING\b",                                  "BREAKING",                     65),
    (r"\burgent(?:ly)?\b",                               None,                           68),
    (r"\bbefore it'?s? too late\b",                      "before it's too late",         88),
    (r"\bmust.?(?:see|watch|read|know)\b",               "must-see/watch",               80),
    (r"\bdon'?t miss\b",                                 "don't miss",                   72),
    (r"\bacts? (?:fast|now|quickly|immediately)\b",      "act now",                      72),
    (r"\blimited time\b",                                "limited time",                 70),
    (r"\bending soon\b",                                 "ending soon",                  68),

    # 6. Engagement / share bait
    (r"\beveryone (?:should|needs? to|must)\b",          "everyone should",              72),
    (r"\bshare (?:this|if you)\b",                       "share this",                   72),
    (r"\bOMG\b",                                         "OMG",                          85),
    (r"\bWOW\b",                                         "WOW",                          80),
    (r"\bviral\b",                                       "viral",                        62),
    (r"\bexclusive\b",                                   "exclusive",                    58),

    # 7. Vague-pronoun clickbait ("This teacher/nurse/teen did X")
    (r"\bthis (?:man|woman|guy|girl|dad|mom|teacher|doctor|nurse|kid|teen|student|cop|pastor|chef|barista)\b", None, 62),

    # 8. Fear-mongering / health panic
    (r"\bkilling you\b",                                 "killing you",                  90),
    (r"\bsecretly (?:killing|destroying|harming|poisoning)\b", "secretly killing",       90),
    (r"\bslowly (?:killing|destroying|poisoning)\b",     "slowly killing/poisoning",     88),
    (r"\byou (?:might|could|may) have\b",                None,                           56),
    (r"\bone food.{0,20}(?:never|avoid|stop)\b",         "one food to avoid",            85),
    (r"\bfoods? (?:you|to) (?:never|avoid|stop)\b",      "foods to avoid",               78),
    (r"\bfoods? (?:that are )?(?:secretly|slowly) (?:killing|destroying|toxic)\b", "foods secretly killing", 90),
    (r"\blose (?:\d+ )?(?:pound|kg|weight).{0,20}(?:fast|quickly|overnight|week)\b", "lose weight fast", 88),
    (r"\bburn (?:fat|calories).{0,20}(?:fast|overnight|trick)\b", "burn fat fast",        85),
    (r"\bmiracle (?:cure|remedy|treatment|diet|pill|solution)\b", "miracle cure",        90),
    (r"\bcure.{0,20}(?:cancer|diabetes|arthritis|depression|alzheimer).{0,20}(?:with|using|at home|naturally)\b", "cure disease at home", 92),
    (r"\bnatural remedy.{0,20}(?:cures?|treats?|heals?)\b", "natural remedy cures",     82),
    (r"\bdetox\b",                                       None,                           62),
    (r"\bwarning signs?\b",                              "warning signs",                68),
    (r"\bdanger(?:ous)?\s+(?:food|chemical|ingredient|drug)\b", "dangerous food/drug",   84),

    # 9. Political outrage bait
    (r"\bwoke\b",                                        None,                           65),
    (r"\bcancel culture\b",                              "cancel culture",               72),
    (r"\bwar on (?:christmas|religion|family|truth|america|freedom|women|men|children)\b", "war on", 84),
    (r"\bthey'?re (?:coming|trying) (?:for|to take)\b",  "they're coming for",           82),
    (r"\bcommunist(?:s)?\b",                             None,                           65),
    (r"\bsocialist (?:agenda|takeover|plot)\b",          "socialist agenda",             80),
    (r"\bfreedom (?:is|under) (?:attack|threat|siege)\b", "freedom under attack",       82),

    # 10. Relationship / celebrity drama bait
    (r"\bcaught cheating\b",                             "caught cheating",              88),
    (r"\bsecret (?:child|baby|affair|relationship|lover)\b", "secret affair",            85),
    (r"\bbroke up\b",                                    "broke up",                     68),
    (r"\bgetting divorced?\b",                           "getting divorced",              74),
    (r"\bpregnant\b",                                    None,                           58),
    (r"\bfeud\b",                                        None,                           65),
    (r"\bsigns (?:he|she|your partner|they)\b",          "signs he/she",                 78),
    (r"\bthings (?:men|women|guys|girls) (?:do|want|hate|love|say|never)\b", "things men/women", 74),
    (r"\bred flags?\b",                                  "red flag",                     68),
    (r"\btoxic (?:relationship|person|friend|partner)\b", "toxic relationship",          68),

    # 11. Finance / wealth bait
    (r"\bget rich\b",                                    "get rich",                     88),
    (r"\bmake money (?:fast|online|from home|while you sleep|easily)\b", "make money fast", 88),
    (r"\bpassive income\b",                              "passive income",               74),
    (r"\bfinancial freedom\b",                           "financial freedom",            68),
    (r"\bquit your job\b",                               "quit your job",                78),
    (r"\bsecret to (?:wealth|money|riches|success|millions)\b", "secret to wealth",     85),
    (r"\bI made \$[\d,]+\b",                             "I made $X",                    85),

    # 12. Supernatural / paranormal claims
    (r"\bufo\b",                                         None,                           62),
    (r"\balien(?:s)?\b",                                 None,                           60),
    (r"\bparanormal\b",                                  None,                           65),
    (r"\bprophecy\b",                                    None,                           65),

    # 13. Misinformation / conspiracy / pseudoscience
    (r"\bearth is flat\b",                               "earth is flat",                97),
    (r"\bflat earth\b",                                  "flat earth",                   97),
    (r"\bearth isn'?t (?:round|a sphere|spherical)\b",   "earth isn't round",            97),
    (r"\bmoon landing.{0,20}(?:fake|faked|hoax|staged)\b", "moon landing faked",         97),
    (r"\bvaccines? (?:cause|caused|causes?) autism\b",   "vaccines cause autism",        97),
    (r"\banti.?vax\b",                                   "anti-vax",                     84),
    (r"\bclimate.{0,15}(?:hoax|fake|lie|scam|conspiracy)\b", "climate hoax",            94),
    (r"\bstolen election\b",                             "stolen election",              94),
    (r"\bdeep state\b",                                  "deep state",                   92),
    (r"\b(?:a )?hoax\b",                                 "hoax",                         90),
    (r"\bconspiracy\b",                                  "conspiracy",                   87),
    (r"\bplandemic\b",                                   "plandemic",                    97),
    (r"\bchemtrail(?:s)?\b",                             "chemtrails",                   94),
    (r"\b(?:the )?illuminati\b",                         "illuminati",                   92),
    (r"\bnew world order\b",                             "new world order",              92),
    (r"\bsheeple\b",                                     "sheeple",                      92),
    (r"\bwake up (?:people|sheeple|america)\b",          "wake up sheeple",              90),
    (r"\bglobalist(?:s)?\b",                             "globalists",                   84),
    (r"\bsuppressed\b",                                  "suppressed",                   82),
    (r"\bpropaganda\b",                                  "propaganda",                   80),
    (r"\bcover.?up\b",                                   "cover-up",                     88),
    (r"\bsecret agenda\b",                               "secret agenda",                90),
    (r"\b(?:mind|thought) control\b",                    "mind control",                 92),
    (r"\b5G.{0,20}(?:dangerous|toxic|weapon|kill|cancer|control)\b", "5G dangerous",    94),
    (r"\bsatanic\b",                                     "satanic",                      90),
    (r"\bpedogate\b",                                    "pedogate",                     97),
    (r"\bmainstream media.{0,20}(?:lie|lied|lying|hiding|covers)\b", "MSM lies",        90),
    (r"\bbig pharma\b",                                  "big pharma",                   80),
    (r"\bfalse flag\b",                                  "false flag",                   94),
    (r"\bchip(?:ped|s).{0,20}(?:vaccine|inject|human)\b", "microchip in vaccine",       94),
    (r"\bgreat reset\b",                                 "the great reset",              90),
    (r"\bshadow (?:government|ban|banned)\b",            "shadow government",            88),
    (r"\blizard (?:people|person|man)\b",                "lizard people",                92),
    (r"\bscam\b",                                        "scam",                         84),
    (r"\bthey'?re (?:hiding|lying|covering)\b",          "they're hiding/lying",         90),
    (r"\bmisinformation\b",                              "misinformation",               68),
    (r"\bpseudoscience\b",                               "pseudoscience",                72),
    (r"\bwhistleblow\w*",                                None,                           65),
]

# ── REAL / CREDIBILITY SIGNALS ───────────────────────────────────────────────
REAL_SCORED: list[tuple[str, str | None, int]] = [

    # 1. Attribution language
    (r"\baccording to\b",                                "according to",                 90),
    (r"\bannounced\b",                                   None,                           76),
    (r"\bconfirmed\b",                                   None,                           80),
    (r"\bdenied\b",                                      None,                           70),
    (r"\bstated\b",                                      None,                           65),
    (r"\bpublished\b",                                   None,                           62),
    (r"\breported(?:ly)?\b",                             None,                           65),
    (r"\bspokesp(?:erson|eople)\b",                      "spokesperson",                 88),
    (r"\bstatement\b",                                   None,                           72),
    (r"\bsaid\b",                                        None,                           55),
    (r"\bresponded\b",                                   None,                           62),
    (r"\btold (?:reporters?|journalists?|media|press)\b", "told reporters",              82),
    (r"\bin (?:a )?(?:statement|interview|briefing|press conference)\b", "in a statement", 78),
    (r"\bdisclosed\b",                                   None,                           74),
    (r"\battributed to\b",                               "attributed to",                78),

    # 2. Research / data language
    (r"\bstudy (?:shows?|finds?|reveals?|suggests?|published)\b",  "study shows",       90),
    (r"\bresearch (?:shows?|finds?|reveals?|suggests?|indicates?|published)\b", "research finds", 90),
    (r"\bdata (?:shows?|reveals?|suggests?|indicates?)\b", "data shows",                85),
    (r"\bsurvey\b",                                      None,                           72),
    (r"\bpoll (?:shows?|finds?)\b",                      "poll finds",                  78),
    (r"\bstatistics?\b",                                 None,                           78),
    (r"\banalysis\b",                                    None,                           72),
    (r"\bfindings?\b",                                   None,                           70),
    (r"\bevidence\b",                                    None,                           74),
    (r"\breport(?:ed)?\b",                               None,                           60),
    (r"\bpeer.?reviewed?\b",                             "peer-reviewed",                92),
    (r"\bclinical trial\b",                              "clinical trial",               92),
    (r"\bjournal\b",                                     None,                           72),
    (r"\bpublished in\b",                                "published in",                 78),
    (r"\bmeta.?analysis\b",                              "meta-analysis",                90),

    # 3. Scientific / academic institutions
    (r"\buniversity\b",                                  None,                           80),
    (r"\bresearchers?\b",                                None,                           78),
    (r"\bscientists?\b",                                 None,                           75),
    (r"\bprofessor\b",                                   None,                           72),
    (r"\bdirector\b",                                    None,                           62),
    (r"\bharvard\b",                                     None,                           85),
    (r"\boxford\b",                                      None,                           85),
    (r"\bmit\b",                                         None,                           82),
    (r"\bstanford\b",                                    None,                           82),
    (r"\byale\b",                                        None,                           80),
    (r"\bpeer(?:s)?\b",                                  None,                           58),

    # 4. Government / international bodies
    (r"\bgovernment\b",                                  None,                           62),
    (r"\bparliament\b",                                  None,                           74),
    (r"\bcongress\b",                                    None,                           70),
    (r"\bsenate\b",                                      None,                           70),
    (r"\b(?:WHO|CDC|FDA|EPA|NIH|NATO|UN|EU|IMF|WTO|OECD)\b", None,                      88),
    (r"\bofficial(?:s|ly)?\b",                           None,                           68),
    (r"\bminister\b",                                    None,                           68),
    (r"\bcommission\b",                                  None,                           65),
    (r"\bagency\b",                                      None,                           62),
    (r"\bpresident.{0,20}(?:said|announced|confirmed|signed|ordered)\b", "president said", 72),

    # 5. Legal / court reporting
    (r"\bverdict\b",                                     None,                           80),
    (r"\bsentenced?\b",                                  None,                           75),
    (r"\bconvicted?\b",                                  None,                           78),
    (r"\bacquitted?\b",                                  None,                           78),
    (r"\bpleaded?\b",                                    None,                           70),
    (r"\bindicted?\b",                                   None,                           75),
    (r"\blawsuit\b",                                     None,                           70),
    (r"\bcharged with\b",                                "charged with",                 75),
    (r"\bruled (?:against|in favor|unconstitutional)\b", "court ruled",                  82),
    (r"\bjudge (?:ruled|ordered|said|blocked)\b",        "judge ruled",                  82),
    (r"\bcourt\b",                                       None,                           65),
    (r"\bappeal(?:s|ed)?\b",                             None,                           62),

    # 6. Financial / economic reporting
    (r"\bGDP\b",                                         None,                           80),
    (r"\binflation\b",                                   None,                           70),
    (r"\binterest rate\b",                               "interest rate",                78),
    (r"\bFederal Reserve\b",                             "Federal Reserve",              82),
    (r"\bstock (?:market|exchange)\b",                   "stock market",                 72),
    (r"\bquarterly (?:earnings?|results?|report)\b",     "quarterly earnings",           80),
    (r"\brevenue\b",                                     None,                           62),
    (r"\bdeficit\b",                                     None,                           68),
    (r"\btreasury\b",                                    None,                           70),
    (r"\bbudget\b",                                      None,                           62),
    (r"\bunemployment\b",                                None,                           70),
    (r"\bGDP (?:grows?|falls?|rises?|drops?|shrinks?)\b", "GDP changes",                80),

    # 7. Medical / clinical journalism
    (r"\bdiagnosed with\b",                              "diagnosed with",               74),
    (r"\btreatment\b",                                   None,                           58),
    (r"\bhospital\b",                                    None,                           62),
    (r"\bphysicians?\b",                                 None,                           68),
    (r"\bpatients?\b",                                   None,                           60),
    (r"\bsymptoms?\b",                                   None,                           55),
    (r"\bvaccination\b",                                 None,                           65),
    (r"\bdose(?:s)?\b",                                  None,                           58),

    # 8. Specific numbers as facts (not listicle-style)
    (r"\b\d+(?:\.\d+)?%\s+of\b",                        None,                           68),
    (r"\$\d+(?:\.\d+)?\s*(?:billion|million|trillion)\b", None,                         72),
    (r"\b\d+(?:\.\d+)?\s*(?:billion|million|trillion)\b", None,                         68),
    (r"\bby \d{4}\b",                                    None,                           55),
    (r"\bover \d{1,3},?\d{3}\b",                         None,                           58),

    # 9. Neutral / hedged language
    (r"\bsuggests?\b",                                   None,                           64),
    (r"\bindicates?\b",                                  None,                           64),
    (r"\bappears? to\b",                                 None,                           58),
    (r"\blikely\b",                                      None,                           54),
    (r"\bmay\b",                                         None,                           48),
    (r"\bcould\b",                                       None,                           46),
    (r"\bpotentially\b",                                 None,                           54),
    (r"\bexpected to\b",                                 "expected to",                  58),
    (r"\bprojected to\b",                                "projected to",                 62),

    # 10. Neutral reporting verbs
    (r"\bvoted\b",                                       None,                           68),
    (r"\belected\b",                                     None,                           70),
    (r"\bappointed\b",                                   None,                           68),
    (r"\barrested\b",                                    None,                           70),
    (r"\bcharged\b",                                     None,                           65),
    (r"\bpassed\b",                                      None,                           60),
    (r"\bsigned\b",                                      None,                           65),
    (r"\breleased\b",                                    None,                           58),
    (r"\bapproved\b",                                    None,                           65),
    (r"\brejected\b",                                    None,                           62),
    (r"\bproposed\b",                                    None,                           60),
    (r"\blaunched\b",                                    None,                           58),
    (r"\bimposed\b",                                     None,                           62),
    (r"\bresumed\b",                                     None,                           58),
    (r"\bhalted\b",                                      None,                           60),
    (r"\bsuspended\b",                                   None,                           62),
    (r"\bindicted\b",                                    None,                           72),
    (r"\bexpelled\b",                                    None,                           62),
    (r"\bsanctioned\b",                                  None,                           65),

    # 11. Named news agencies / editorial language
    (r"\b(?:Reuters|Associated Press|\bAP\b|AFP|BBC|CNN|NBC|ABC|CBS|NPR|Guardian|Times|Post)\b", None, 82),
    (r"\bjournalist\b",                                  None,                           68),
    (r"\beditor(?:ial)?\b",                              None,                           55),
    (r"\bcorrespond(?:ent|ing)\b",                       None,                           68),
    (r"\binvestigation\b",                               None,                           65),

    # 12. International / diplomatic language
    (r"\bdiplomatic\b",                                  None,                           70),
    (r"\bsanction(?:s|ed)?\b",                           None,                           68),
    (r"\btreaty\b",                                      None,                           72),
    (r"\bnegotiat(?:ion|ing|ed)?\b",                     None,                           65),
    (r"\bembassy\b",                                     None,                           68),
    (r"\bforeign minister\b",                            "foreign minister",             75),
    (r"\bsummit\b",                                      None,                           68),
    (r"\bcease.?fire\b",                                 "ceasefire",                    70),
    (r"\bhumanitarian\b",                                None,                           65),
    (r"\baid (?:workers?|organizations?)\b",             "aid workers",                  68),
]

CAPS_THRESHOLD = 0.35


def caps_ratio(text: str) -> float:
    alpha = [c for c in text if c.isalpha()]
    if not alpha:
        return 0.0
    return sum(1 for c in alpha if c.isupper()) / len(alpha)


def score_patterns(text: str, patterns: list[tuple[str, str | None, int]]) -> float:
    """Sum severity scores for all matched patterns (case-insensitive)."""
    text_l = text.lower()
    return sum(score for p, _, score in patterns if re.search(p, text_l))


def extract_keywords(headline: str, body: Optional[str]) -> list[KeywordMatch]:
    """Return matched clickbait signals sorted by severity."""
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
) -> list[str]:
    indicators: list[str] = []
    h_lower = headline.lower()

    if p_cb > 0.5:
        indicators.append("Sensational or emotionally-charged language detected")
    if re.search(r"!{2,}", headline):
        indicators.append("Multiple exclamation marks detected")
    elif re.search(r"!$", headline):
        indicators.append("Headline ends with exclamation mark")
    if re.search(r"\b\d+\s+(?:reasons?|ways?|things?|facts?|tips?|signs?|foods?|mistakes?|hacks?)\b", h_lower):
        indicators.append("Listicle-style headline (numbered bait)")
    if caps_ratio(headline) > CAPS_THRESHOLD:
        indicators.append("Excessive capital letters")
    curiosity_gap = [p for p, _, _ in CB_SCORED[:20]]
    if any(re.search(p, h_lower) for p in curiosity_gap):
        indicators.append("Curiosity-gap or manipulative framing detected")
    conspiracy_terms = ["hoax", "conspiracy", "flat earth", "faked", "illuminati",
                        "deep state", "chemtrail", "plandemic", "false flag", "great reset"]
    if any(t in h_lower for t in conspiracy_terms):
        indicators.append("Conspiracy or pseudoscience claim detected")
    if re.search(r"\baccording to\b|\bstudy (?:shows?|finds?)\b|\bresearch\b|\bconfirmed\b|\bpeer.?reviewed\b", h_lower):
        indicators.append("Attribution or research reference in headline")
    if body:
        body_lower = body.lower()
        cred_hits = sum(1 for w in
            ["according to", "study", "research", "official", "confirmed", "spokesperson", "published"]
            if w in body_lower)
        if cred_hits >= 2:
            indicators.append("Article body contains multiple credible source citations")
        elif cred_hits == 1:
            indicators.append("Article body contains a credible source citation")
        if len(body.strip()) < 80:
            indicators.append("Very short article body — limited factual context")

    return indicators if indicators else ["No strong signals detected — content appears neutral"]


def ml_predict_proba(headline: str) -> Optional[float]:
    """Return P(clickbait) from the trained TF-IDF+LogReg model, or None if unavailable."""
    if ml_vectorizer is None or ml_classifier is None:
        return None
    vec = ml_vectorizer.transform([headline])
    # classifier.classes_ is [0, 1] where 1 == clickbait
    return float(ml_classifier.predict_proba(vec)[0][1])


# Body carries real signal but should never outweigh the headline — the model
# was trained on headlines only, so the body's contribution is capped at 25%.
BODY_ML_WEIGHT = 0.25
HEADLINE_ML_WEIGHT = 1.0 - BODY_ML_WEIGHT

# When the fine-tuned DistilBERT model is available, it gets more weight than
# the TF-IDF+structural model in the headline-style score — contextual
# embeddings understand sentence semantics, so they should generalize better
# to phrasing/vocabulary combinations neither model saw during training.
TRANSFORMER_HEADLINE_WEIGHT = 0.6
TFIDF_HEADLINE_WEIGHT = 1.0 - TRANSFORMER_HEADLINE_WEIGHT


def ml_predict_combined(headline: str, body: Optional[str]) -> Optional[float]:
    """Blend the headline-only model score(s) with a body-only score (if body given)."""
    p_tfidf = ml_predict_proba(headline)
    p_transformer = transformer_predict_proba(headline)
    if p_tfidf is None and p_transformer is None:
        return None
    if p_tfidf is not None and p_transformer is not None:
        p_headline = TFIDF_HEADLINE_WEIGHT * p_tfidf + TRANSFORMER_HEADLINE_WEIGHT * p_transformer
    else:
        p_headline = p_transformer if p_transformer is not None else p_tfidf

    body_text = (body or "").strip()
    if not body_text:
        return p_headline
    # DistilBERT isn't applied to the (potentially long) body — too slow for
    # the inference path; the TF-IDF model handles body scoring as before.
    p_body = ml_predict_proba(body_text[:1000])
    if p_body is None:
        return p_headline
    return HEADLINE_ML_WEIGHT * p_headline + BODY_ML_WEIGHT * p_body


def predict(
    headline: str,
    body: Optional[str],
    fact_result: Optional[dict],
    verified_result: Optional[dict] = None,
    wiki_result: Optional[dict] = None,
) -> PredictResponse:
    # ── Rule-based scoring ─────────────────────────────────────────────────
    # Clickbait: headline 100%, body 25% (body rarely contains bait)
    cb_hl   = score_patterns(headline, CB_SCORED)
    cb_body = score_patterns(body[:400] if body else "", CB_SCORED) * 0.25
    cb_raw  = cb_hl + cb_body

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
    real_hl   = score_patterns(headline, REAL_SCORED)
    real_body = score_patterns(body[:600] if body else "", REAL_SCORED) * 0.60
    real_raw  = real_hl + real_body

    # ── Normalisation ──────────────────────────────────────────────────────
    # Max realistic raw: heavy clickbait ~650, credible article ~600
    p_cb_rule   = min(cb_raw  / 600.0, 1.0)
    p_real_rule = min(real_raw / 560.0, 1.0)

    has_signals = (p_cb_rule + p_real_rule) > 0.04

    if has_signals:
        # Clickbait signals actively suppress real credibility score —
        # a journalist using click-bait language is still writing clickbait.
        cb_suppression = p_cb_rule * 0.45
        p_real_adj = max(0.0, p_real_rule - cb_suppression)
        p_cb_adj   = p_cb_rule

        total = p_real_adj + p_cb_adj
        if total < 0.01:
            total = 1.0
        p_real_final = p_real_adj / total
        p_cb_final   = p_cb_adj   / total
    else:
        # No signals — genuinely neutral sentence.
        # Default: REAL at 60% moderate confidence.
        p_real_final = 0.60
        p_cb_final   = 0.40

    # ── Blend in the trained ML model (if available) ───────────────────────
    p_cb_ml = ml_predict_combined(headline, body)
    if p_cb_ml is not None:
        p_cb_final   = ML_WEIGHT * p_cb_ml + RULE_WEIGHT * p_cb_final
        p_real_final = 1.0 - p_cb_final

    # ── Tier 1 snapshot: pure writing-style verdict ─────────────────────────
    # Captured here, before any fact-check/corroboration signal is blended
    # in below, so it reflects *only* headline/body phrasing — never whether
    # the underlying claim is actually true. Exposed separately in the API
    # response as `style_analysis` (see "Decouple style vs fact checking").
    style_indicators = extract_indicators(headline, body, p_cb_final, p_real_final)
    style_keywords = extract_keywords(headline, body)
    style_verdict = (
        "CLICKBAIT_STYLE" if p_cb_final > 0.58
        else "CREDIBLE_STYLE" if p_real_final > 0.58
        else "NEUTRAL_STYLE"
    )
    style_analysis = StyleAnalysis(
        verdict=style_verdict,
        clickbait_score=round(p_cb_final * 100, 1),
        credible_score=round(p_real_final * 100, 1),
        indicators=style_indicators,
        keywords=style_keywords,
        model_used=model_name,
    )

    # ── Blend in real-world fact-check corroboration (if available) ───────
    factcheck_note: Optional[str] = None
    sources: list[SourceRef] = []
    if fact_result is not None:
        status = fact_result["status"]
        top_article = fact_result.get("top_article")
        weak_match = fact_result.get("weak_match", False)
        news_label = "Possibly related article (unverified — please check yourself)" if weak_match else "News coverage"
        if status == "corroborated":
            p_cb_factcheck = 0.08
            p_cb_final = 0.55 * p_cb_final + 0.45 * p_cb_factcheck
            p_real_final = 1.0 - p_cb_final
            source_note = f" (e.g. {fact_result['top_source']})" if fact_result.get("top_source") else ""
            factcheck_note = f"Corroborated by {fact_result['articles_found']} real news article(s){source_note}"
            if top_article and top_article.get("title"):
                sources.append(SourceRef(
                    label=news_label, title=top_article["title"],
                    url=top_article.get("url"), publisher=top_article.get("publisher"),
                ))
        elif status == "partially_corroborated":
            p_cb_factcheck = 0.15
            p_cb_final = 0.5 * p_cb_final + 0.5 * p_cb_factcheck
            p_real_final = 1.0 - p_cb_final
            source_note = f" (e.g. {fact_result['top_source']})" if fact_result.get("top_source") else ""
            factcheck_note = f"Found closely related real news coverage{source_note} supporting this claim"
            if top_article and top_article.get("title"):
                sources.append(SourceRef(
                    label=news_label, title=top_article["title"],
                    url=top_article.get("url"), publisher=top_article.get("publisher"),
                ))
        elif status == "no_coverage":
            p_cb_factcheck = 0.8
            p_cb_final = 0.5 * p_cb_final + 0.5 * p_cb_factcheck
            p_real_final = 1.0 - p_cb_final
            factcheck_note = "No corroborating news coverage found for this claim — may be unverified or fabricated"
        elif status == "unconfirmed_claim":
            # The subject is real/newsworthy, but none of the actual coverage
            # about them supports this specific claim — a strong hoax signal
            # (e.g. a celebrity/politician "death" claim with no matching
            # obituary or news story anywhere).
            p_cb_factcheck = 0.92
            p_cb_final = 0.3 * p_cb_final + 0.7 * p_cb_factcheck
            p_real_final = 1.0 - p_cb_final
            source_note = f" (e.g. {fact_result['top_source']})" if fact_result.get("top_source") else ""
            factcheck_note = (
                f"Found {fact_result['articles_found']} article(s) about this subject{source_note}, "
                "but none report this specific claim — likely false or unverified"
            )
            if top_article and top_article.get("title"):
                sources.append(SourceRef(
                    label="Related coverage (claim not found in it)", title=top_article["title"],
                    url=top_article.get("url"), publisher=top_article.get("publisher"),
                ))
        elif status == "inconclusive":
            factcheck_note = f"Found {fact_result['articles_found']} loosely related article(s) — corroboration inconclusive"
            if top_article and top_article.get("title"):
                sources.append(SourceRef(
                    label="Loosely related article — verify yourself", title=top_article["title"],
                    url=top_article.get("url"), publisher=top_article.get("publisher"),
                ))

    verdict  = "CLICKBAIT" if p_cb_final > 0.5 else "REAL"
    raw_conf = p_cb_final if verdict == "CLICKBAIT" else p_real_final
    verified_note: Optional[str] = None

    # ── Professional fact-check verdict (if found) — takes top priority ────
    # A human-reviewed rating on this exact claim beats every heuristic
    # signal above, since it's an actual verified answer, not an inference.
    if verified_result is not None:
        vstatus = verified_result["status"]
        publisher = verified_result.get("publisher", "a fact-checker")
        rating = verified_result.get("rating", "")
        if vstatus == "false":
            verdict = "CLICKBAIT"
            p_cb_final, p_real_final = 0.95, 0.05
            raw_conf = 0.95
            verified_note = f"Fact-checked by {publisher}: rated \"{rating}\" — this specific claim has been reviewed and found false"
        elif vstatus == "true":
            verdict = "REAL"
            p_cb_final, p_real_final = 0.05, 0.95
            raw_conf = 0.95
            verified_note = f"Fact-checked by {publisher}: rated \"{rating}\" — this specific claim has been verified"
        elif vstatus == "mixed":
            verified_note = f"Fact-checked by {publisher}: rated \"{rating}\" — partially true, treat with caution"
        if verified_result.get("url"):
            sources.insert(0, SourceRef(
                label="Fact-check", title=f"Rated \"{rating}\" by {publisher}",
                url=verified_result.get("url"), publisher=publisher,
            ))

    # ── Wikipedia biographical grounding (alive/dead) — quiet secondary
    # signal only. Per design choice: Wikipedia is user-editable and not a
    # news source, so it never appears in the "sources" evidence list and
    # can never single-handedly force a confident verdict. Real, credible
    # news coverage (NewsAPI / Google Fact Check) always takes priority —
    # this only nudges the score, and only when no decisive news signal
    # already exists.
    real_news_decisive = verified_result is not None or (
        fact_result is not None and fact_result["status"] in (
            "corroborated", "no_coverage", "unconfirmed_claim", "partially_corroborated",
        )
    )
    if not real_news_decisive and wiki_result is not None:
        wstatus = wiki_result["status"]
        if wstatus == "contradicted":
            p_cb_wiki = 0.75
            p_cb_final = 0.55 * p_cb_final + 0.45 * p_cb_wiki
            p_real_final = 1.0 - p_cb_final
            if not factcheck_note:
                factcheck_note = f"{wiki_result['note']} — treat with caution (not independently confirmed by news coverage)"
        elif wstatus == "confirmed":
            p_cb_wiki = 0.25
            p_cb_final = 0.55 * p_cb_final + 0.45 * p_cb_wiki
            p_real_final = 1.0 - p_cb_final
            if not factcheck_note:
                factcheck_note = f"{wiki_result['note']} — for reference only, not a verified news report"
        elif wstatus == "related" and not factcheck_note:
            p_cb_final = 0.7 * p_cb_final + 0.3 * 0.3
            p_real_final = 1.0 - p_cb_final
        if verified_result is None:
            # Re-derive the verdict/confidence now that the wiki nudge has
            # been folded in (verified_result, when present, already fixed
            # these above and takes priority over this secondary signal).
            verdict  = "CLICKBAIT" if p_cb_final > 0.5 else "REAL"
            raw_conf = p_cb_final if verdict == "CLICKBAIT" else p_real_final

    # ── Tier 2 status label exposed in the API response ────────────────────
    if verified_result is not None:
        fact_tier_status = f"fact_checked_{verified_result['status']}"  # fact_checked_true/false/mixed
    elif fact_result is not None:
        fact_tier_status = fact_result["status"]
    elif wiki_result is not None:
        fact_tier_status = f"wiki_{wiki_result['status']}"
    else:
        fact_tier_status = "not_checked"

    fact_check_analysis = FactCheckAnalysis(
        status=fact_tier_status,
        note=verified_note or factcheck_note,
        sources=sources,
    )

    # ── Admit uncertainty rather than force a confident guess ──────────────
    # If no professional fact-check exists, no fact-check API signal was
    # decisive, and the model+rules genuinely can't separate REAL/CLICKBAIT
    # (near a coin-flip), it is more honest to say so than to output a
    # confident-sounding but essentially random verdict.
    decisive_factcheck = fact_result is not None and fact_result["status"] in (
        "corroborated", "no_coverage", "unconfirmed_claim", "partially_corroborated",
    )
    if verified_result is None and not decisive_factcheck and raw_conf < 0.58:
        verdict = "UNCERTAIN"
        confidence = round(50 + abs(p_cb_final - 0.5) * 100, 1)
        indicators = extract_indicators(headline, body, p_cb_final, p_real_final)
        indicators.insert(
            0,
            "Not enough evidence (writing style or independent news coverage) to confidently verify this "
            "claim — treat it with caution and check a trusted source",
        )
        keywords = extract_keywords(headline, body)
        return PredictResponse(
            verdict=verdict,
            confidence=confidence,
            scores={
                "real":      round(p_real_final * 100, 1),
                "clickbait": round(p_cb_final   * 100, 1),
            },
            indicators=indicators,
            keywords=keywords,
            sources=sources,
            model_used=model_name,
            style_analysis=style_analysis,
            fact_check=fact_check_analysis,
        )

    if has_signals or p_cb_ml is not None or fact_result is not None or verified_result is not None or wiki_result is not None:
        confidence = round(max(54.0, min(97.0, raw_conf * 100)), 1)
    else:
        # Neutral — stay humble about confidence
        confidence = round(max(54.0, min(68.0, raw_conf * 100)), 1)

    indicators = extract_indicators(headline, body, p_cb_final, p_real_final)
    if verified_note:
        indicators.insert(0, verified_note)
    elif factcheck_note:
        indicators.insert(0, factcheck_note)
    keywords   = extract_keywords(headline, body)

    return PredictResponse(
        verdict=verdict,
        confidence=confidence,
        scores={
            "real":      round(p_real_final * 100, 1),
            "clickbait": round(p_cb_final   * 100, 1),
        },
        indicators=indicators,
        keywords=keywords,
        sources=sources,
        model_used=model_name,
        style_analysis=style_analysis,
        fact_check=fact_check_analysis,
    )


@app.post("/predict", response_model=PredictResponse)
async def predict_route(req: PredictRequest):
    if not model_ready:
        raise HTTPException(status_code=503, detail="Model not ready yet, please retry")
    fact_result, verified_result, wiki_result = await asyncio.gather(
        fact_check_headline(req.headline),
        google_fact_check(req.headline),
        wikipedia_check(req.headline),
    )
    return predict(req.headline, req.body, fact_result, verified_result, wiki_result)


@app.get("/status", response_model=StatusResponse)
async def status_route():
    return StatusResponse(
        ready=model_ready,
        model_name=model_name,
        feedback_count=model_feedback_examples_used,
        last_retrained=model_last_retrained,
        gemini_examples_used=model_gemini_examples_used,
        gemini_accuracy=model_gemini_accuracy,
        groq_examples_used=model_groq_examples_used,
        groq_accuracy=model_groq_accuracy,
        teacher_examples_used=model_teacher_examples_used,
        teacher_accuracy=model_teacher_accuracy,
        ready_for_local_only=_is_ready_for_local_only(),
    )


@app.post("/retrain", response_model=StatusResponse)
async def retrain_route():
    """Manually trigger an immediate retrain from current feedback (normally runs on a timer)."""
    import asyncio
    from retrain import retrain as retrain_model

    if not DATABASE_URL:
        raise HTTPException(status_code=400, detail="DATABASE_URL not configured — cannot retrain from feedback")
    await asyncio.to_thread(retrain_model, DATABASE_URL)
    _load_model_from_disk()
    _load_transformer_model()
    return StatusResponse(
        ready=model_ready,
        model_name=model_name,
        feedback_count=model_feedback_examples_used,
        last_retrained=model_last_retrained,
        gemini_examples_used=model_gemini_examples_used,
        gemini_accuracy=model_gemini_accuracy,
        groq_examples_used=model_groq_examples_used,
        groq_accuracy=model_groq_accuracy,
        teacher_examples_used=model_teacher_examples_used,
        teacher_accuracy=model_teacher_accuracy,
        ready_for_local_only=_is_ready_for_local_only(),
    )


@app.get("/healthz")
async def health():
    return {"status": "ok", "model_ready": model_ready}


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("ML_PORT", "8001"))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
