import { logger } from "./logger";

const ML_SERVICE_URL = process.env.ML_SERVICE_URL ?? "http://localhost:8001";

export interface MlServiceStatus {
  ready: boolean;
  model_name: string;
  feedback_count: number;
  last_retrained: string | null;
  gemini_examples_used: number;
  gemini_accuracy: number | null;
  groq_examples_used: number;
  groq_accuracy: number | null;
  teacher_examples_used: number;
  teacher_accuracy: number | null;
  ready_for_local_only: boolean;
}

// Cached with a short TTL so the per-analyze "should we still call Gemini?"
// check doesn't add an extra network round-trip to every single request.
const CACHE_TTL_MS = 30_000;
let cached: MlServiceStatus | null = null;
let cachedAt = 0;

export async function getMlStatus(forceRefresh = false): Promise<MlServiceStatus | null> {
  const now = Date.now();
  if (!forceRefresh && cached && now - cachedAt < CACHE_TTL_MS) {
    return cached;
  }
  try {
    const res = await fetch(`${ML_SERVICE_URL}/status`, { signal: AbortSignal.timeout(5000) });
    if (!res.ok) throw new Error(`ML status endpoint failed: ${res.status}`);
    const status = (await res.json()) as MlServiceStatus;
    cached = status;
    cachedAt = now;
    return status;
  } catch (err) {
    logger.warn({ err }, "ML status check failed");
    return cached; // serve stale cache rather than nothing, if we have one
  }
}

/**
 * True once the local ML model has distilled enough Gemini- and
 * Groq-labeled examples — and reproduces their combined judgment accurately
 * enough — that calling either API for new classifications is no longer
 * necessary.
 */
export async function isLocalModelReadyToReplaceGemini(): Promise<boolean> {
  const status = await getMlStatus();
  return status?.ready_for_local_only ?? false;
}
