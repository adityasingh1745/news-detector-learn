import { logger } from "./logger";

const GROQ_API_KEY = process.env.GROQ_API_KEY;
const GROQ_URL = "https://api.groq.com/openai/v1/chat/completions";

// Secondary teacher model: used only when Gemini is unconfigured, rate
// limited, or down (all configured models exhausted). Try a short ordered
// list of currently-available Groq models in case one is overloaded.
// Override with GROQ_MODEL (single) or GROQ_MODELS (comma-separated).
const GROQ_MODELS: string[] = (
  process.env.GROQ_MODELS
    ? process.env.GROQ_MODELS.split(",").map((m) => m.trim()).filter(Boolean)
    : process.env.GROQ_MODEL
      ? [process.env.GROQ_MODEL]
      : ["openai/gpt-oss-120b", "openai/gpt-oss-20b"]
);

// Groq's free tier is generally more generous than Gemini's (commonly
// 30 req/min depending on model), but we still track a rolling window so
// we never hammer it past quota. Override with GROQ_RATE_LIMIT_PER_MINUTE.
const RATE_LIMIT_PER_MINUTE = Number(process.env.GROQ_RATE_LIMIT_PER_MINUTE ?? "30");
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

export interface GroqVerdict {
  verdict: "REAL" | "CLICKBAIT";
  confidence: number;
  scores: { real: number; clickbait: number };
  indicators: string[];
  modelUsed: string;
}

export function isGroqConfigured(): boolean {
  return Boolean(GROQ_API_KEY);
}

export function isGroqRateLimited(): boolean {
  return !hasQuota();
}

/**
 * Classifies a headline (and optional body) as REAL or CLICKBAIT using
 * Groq as a secondary teacher model, used only when Gemini is unavailable.
 * Returns null if Groq is not configured, rate-limited, or the call/parse
 * fails — callers should fall back to the local ML pipeline in that case.
 */
export async function classifyWithGroq(
  headline: string,
  body?: string | null,
): Promise<GroqVerdict | null> {
  if (!GROQ_API_KEY) return null;
  if (!hasQuota()) {
    logger.warn("Groq rate limit reached; falling back to local ML service");
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
    // No backoff delay between models — move on immediately on 429/503 so
    // a provider-wide outage doesn't stack multi-second waits (see gemini.ts
    // for the same reasoning). Short timeout per attempt for the same
    // reason.
    outer: for (const model of GROQ_MODELS) {
      if (!hasQuota()) break outer;
      recordCall();
      res = await fetch(GROQ_URL, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${GROQ_API_KEY}`,
        },
        body: JSON.stringify({
          model,
          messages: [{ role: "user", content: prompt }],
          temperature: 0.1,
          response_format: { type: "json_object" },
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
      logger.warn({ status: res?.status, text, modelsTried: GROQ_MODELS }, "Groq API call failed for all models");
      return null;
    }

    const data = (await res.json()) as {
      choices?: { message?: { content?: string } }[];
    };
    const raw = data.choices?.[0]?.message?.content;
    if (!raw) {
      logger.warn({ data }, "Groq response missing text content");
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
        : [`Classified as ${verdict} by Groq`];

    return {
      verdict,
      confidence,
      scores:
        verdict === "CLICKBAIT"
          ? { real: 100 - confidence, clickbait: confidence }
          : { real: confidence, clickbait: 100 - confidence },
      indicators,
      modelUsed: `groq:${modelUsed}`,
    };
  } catch (err) {
    logger.warn({ err }, "Groq classification failed");
    return null;
  }
}
