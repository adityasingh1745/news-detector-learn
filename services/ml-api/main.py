"""
News Credibility & Clickbait Detection — Rule-Based Classifier v3
Classifies news as REAL (credible journalism) or CLICKBAIT (sensational/bait/misinfo).
Pure pattern-based — no external model required. Starts instantly.
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

model_name = "rule-based-v3"
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
    global model_ready
    logger.info("Rule-based classifier v3 ready — instant startup, no model download")
    model_ready = True
    yield


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


def predict(headline: str, body: Optional[str]) -> PredictResponse:
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

    verdict  = "CLICKBAIT" if p_cb_final > 0.5 else "REAL"
    raw_conf = p_cb_final if verdict == "CLICKBAIT" else p_real_final

    if has_signals:
        confidence = round(max(54.0, min(97.0, raw_conf * 100)), 1)
    else:
        # Neutral — stay humble about confidence
        confidence = round(max(54.0, min(68.0, raw_conf * 100)), 1)

    indicators = extract_indicators(headline, body, p_cb_final, p_real_final)
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
        model_used=model_name,
    )


@app.post("/predict", response_model=PredictResponse)
async def predict_route(req: PredictRequest):
    if not model_ready:
        raise HTTPException(status_code=503, detail="Model not ready yet, please retry")
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
