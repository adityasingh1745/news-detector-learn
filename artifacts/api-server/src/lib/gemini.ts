import { logger } from "./logger";

const GEMINI_API_KEY = process.env.GEMINI_API_KEY;
// Gemini's free-tier models periodically return 503 "high demand" errors
// independently of each other (observed: gemini-3.8-flash can be overloaded
// while gemini-3.6-flash/3.7-flash respond fine). Instead of relying on a
// single model name — which also drifts as Google deprecates/renames models
// — try a short, ordered list of models and move to the next one on a
// 429/503, only falling back to the local ML pipeline if all are down.
// Override with GEMINI_MODEL (single) or GEMINI_MODELS (comma-separated).
const GEMINI_MODELS: string[] = (
  process.env.GEMINI_MODELS
    ? process.env.GEMINI_MODELS.split(",").map((m) => m.trim()).filter(Boolean)
    : process.env.GEMINI_MODEL
      ? [process.env.GEMINI_MODEL]
      : ["gemini-3.8-flash", "gemini-3.6-flash"]
);

function geminiUrl(model: string): string {
  return `https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent`;
}

// Free-tier Gemini quota is 15 requests/minute. Track call timestamps in a
// rolling 60s window so we never exceed it; callers fall back to the local
// ML pipeline when the budget is exhausted instead of hitting a 429.
const RATE_LIMIT_PER_MINUTE = 15;
const WINDOW_MS = 60_000;
const callTimestamps: number[] = [];

function hasQuota(): boolean {
  const now = Date.now();
  while (callTimestamps.length > 0 && now - callTimestamps[0]! > WINDOW_MS) {
    callTimestamps.shift();
  }
  return callTimestamps.length < RATE_LIMIT_PER_MINUTE;
}

function recordCall(): void {
  callTimestamps.push(Date.now());
}

export interface GeminiVerdict {
  verdict: "REAL" | "CLICKBAIT";
  confidence: number;
  scores: { real: number; clickbait: number };
  indicators: string[];
  modelUsed: string;
}

export function isGeminiConfigured(): boolean {
  return Boolean(GEMINI_API_KEY);
}

export function isGeminiRateLimited(): boolean {
  return !hasQuota();
}

/**
 * Classifies a headline (and optional body) as REAL or CLICKBAIT using
 * Google Gemini. Returns null if Gemini is not configured, is currently
 * rate-limited (15 req/min free tier), or the call/parse fails — callers
 * should fall back to the local ML pipeline in that case.
 */
export async function classifyWithGemini(
  headline: string,
  body?: string | null,
): Promise<GeminiVerdict | null> {
  if (!GEMINI_API_KEY) return null;
  if (!hasQuota()) {
    logger.warn("Gemini rate limit (15/min) reached; falling back to local ML service");
    return null;
  }

  const prompt = [
    "You are a news verification assistant. Classify the following news headline",
    "(and optional article body) as either REAL or CLICKBAIT.",
    '- "REAL": a straightforward, credible news headline.',
    '- "CLICKBAIT": sensationalized, exaggerated, vague, or manipulative wording designed to bait clicks,',
    "  regardless of whether the underlying facts are true.",
    "",
    `Headline: ${headline}`,
    body ? `Body: ${body}` : "Body: (none provided)",
    "",
    "Respond with ONLY a JSON object matching this exact shape, no markdown fences:",
    '{"verdict": "REAL" | "CLICKBAIT", "confidence": <integer 0-100>, "indicators": [<short strings explaining the reasoning>]}',
  ].join("\n");

  try {
    let res: Response | undefined;
    let modelUsed: string | undefined;
    // On a transient overload (503) or rate-limit (429) response, move on to
    // the next model immediately — no backoff delay — so a provider-wide
    // outage degrades to the next fallback in well under a second instead
    // of stacking multi-second waits per model. Each attempt uses a short
    // timeout for the same reason: a hung request shouldn't stall the whole
    // user-facing analyze call.
    outer: for (const model of GEMINI_MODELS) {
      if (!hasQuota()) break outer;
      recordCall();
      res = await fetch(`${geminiUrl(model)}?key=${GEMINI_API_KEY}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          contents: [{ parts: [{ text: prompt }] }],
          generationConfig: {
            temperature: 0.1,
            responseMimeType: "application/json",
          },
        }),
        signal: AbortSignal.timeout(8000),
      });
      if (res.ok) {
        modelUsed = model;
        break outer;
      }
      if (res.status !== 429 && res.status !== 503) break outer;
    }

    if (!res || !res.ok || !modelUsed) {
      const text = res ? await res.text() : "no response";
      logger.warn({ status: res?.status, text, modelsTried: GEMINI_MODELS }, "Gemini API call failed for all models");
      return null;
    }

    const data = (await res.json()) as {
      candidates?: { content?: { parts?: { text?: string }[] } }[];
    };
    const raw = data.candidates?.[0]?.content?.parts?.[0]?.text;
    if (!raw) {
      logger.warn({ data }, "Gemini response missing text content");
      return null;
    }

    const parsed = JSON.parse(raw) as {
      verdict?: string;
      confidence?: number;
      indicators?: string[];
    };

    const verdict = parsed.verdict === "CLICKBAIT" ? "CLICKBAIT" : "REAL";
    const confidence = Math.max(0, Math.min(100, Math.round(Number(parsed.confidence) || 70)));
    const indicators =
      Array.isArray(parsed.indicators) && parsed.indicators.length > 0
        ? parsed.indicators
        : [`Classified as ${verdict} by Gemini`];

    return {
      verdict,
      confidence,
      scores:
        verdict === "CLICKBAIT"
          ? { real: 100 - confidence, clickbait: confidence }
          : { real: confidence, clickbait: 100 - confidence },
      indicators,
      modelUsed: `gemini:${modelUsed}`,
    };
  } catch (err) {
    logger.warn({ err }, "Gemini classification failed");
    return null;
  }
}
