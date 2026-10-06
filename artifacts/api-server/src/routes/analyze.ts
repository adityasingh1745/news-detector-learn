import { Router } from "express";
import { db, analysesTable, feedbackTable } from "@workspace/db";
import { eq, desc, count, avg, sql } from "drizzle-orm";
import { AnalyzeNewsBody, SubmitFeedbackBody, GetHistoryQueryParams } from "@workspace/api-zod";
import { logger } from "../lib/logger";
import { classifyWithGemini } from "../lib/gemini";
import { classifyWithGroq } from "../lib/groq";
import { getMlStatus, isLocalModelReadyToReplaceGemini } from "../lib/mlStatus";

const router = Router();

const ML_SERVICE_URL = process.env.ML_SERVICE_URL ?? "http://localhost:8001";

async function callMlService(headline: string, body?: string | null) {
  const res = await fetch(`${ML_SERVICE_URL}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ headline, body: body ?? null }),
    signal: AbortSignal.timeout(30000),
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`ML service error ${res.status}: ${text}`);
  }
  return res.json() as Promise<{
    verdict: string;
    confidence: number;
    scores: { real: number; clickbait: number };
    indicators: string[];
    keywords: { word: string; score: number }[];
    sources?: { label: string; title: string; url: string | null; publisher: string | null }[];
    model_used: string;
    style_analysis?: {
      verdict: string;
      clickbait_score: number;
      credible_score: number;
      indicators: string[];
      keywords: { word: string; score: number }[];
      model_used: string;
    } | null;
    fact_check?: {
      status: string;
      note: string | null;
      sources: { label: string; title: string; url: string | null; publisher: string | null }[];
    } | null;
  }>;
}

// POST /api/analyze
router.post("/analyze", async (req, res) => {
  const parse = AnalyzeNewsBody.safeParse(req.body);
  if (!parse.success) {
    res.status(400).json({ error: "Invalid request: " + parse.error.message });
    return;
  }
  const { headline, body } = parse.data;

  let mlResult: Awaited<ReturnType<typeof callMlService>>;
  try {
    mlResult = await callMlService(headline, body);
  } catch (err) {
    req.log.error({ err }, "ML service call failed");
    res.status(503).json({ error: "ML analysis service is unavailable. Please ensure the ML server is running." });
    return;
  }

  // ── Community feedback majority-vote override ──────────────────────────
  // If enough past users have flagged this exact headline the same way,
  // trust that crowd signal over a single fresh model prediction. We
  // require a genuine majority (>=2 votes, no tie) so a single differing
  // opinion — or two users disagreeing with each other — never overrides
  // the model on its own.
  let finalVerdict = mlResult.verdict;
  let finalScores = mlResult.scores;
  let finalConfidence = mlResult.confidence;
  let finalIndicators = mlResult.indicators;
  let finalModelUsed = mlResult.model_used;

  // ── Gemini (primary) → Groq (secondary) → local ML (last resort) ───────
  // Gemini's verdict takes priority. If Gemini is unconfigured, rate-limited,
  // or every candidate model is down, Groq is tried next as a second
  // "teacher" model before falling back to the weaker local ML result.
  // Once the local model has distilled enough Gemini+Groq-labeled examples
  // to reproduce their judgment accurately on its own, we stop calling
  // either API — this is how the system "graduates" off needing API keys.
  try {
    const graduated = await isLocalModelReadyToReplaceGemini();
    if (graduated) {
      req.log.info("Local model has graduated past Gemini/Groq; skipping both API calls");
    } else {
      const geminiResult = await classifyWithGemini(headline, body);
      if (geminiResult) {
        finalVerdict = geminiResult.verdict;
        finalScores = geminiResult.scores;
        finalConfidence = geminiResult.confidence;
        finalIndicators = geminiResult.indicators;
        finalModelUsed = geminiResult.modelUsed;
      } else {
        req.log.warn("Gemini unavailable; trying Groq as secondary classifier");
        const groqResult = await classifyWithGroq(headline, body);
        if (groqResult) {
          finalVerdict = groqResult.verdict;
          finalScores = groqResult.scores;
          finalConfidence = groqResult.confidence;
          finalIndicators = groqResult.indicators;
          finalModelUsed = groqResult.modelUsed;
        }
      }
    }
  } catch (err) {
    req.log.warn({ err }, "Gemini/Groq classification failed; using ML service result as-is");
  }

  try {
    const voteRows = await db
      .select({ correctLabel: feedbackTable.correctLabel, cnt: count() })
      .from(feedbackTable)
      .innerJoin(analysesTable, eq(feedbackTable.analysisId, analysesTable.id))
      .where(sql`lower(trim(${analysesTable.headline})) = lower(trim(${headline}))`)
      .groupBy(feedbackTable.correctLabel);

    const sorted = voteRows
      .map((r) => ({ label: r.correctLabel, votes: Number(r.cnt) }))
      .sort((a, b) => b.votes - a.votes);

    const top = sorted[0];
    const runnerUp = sorted[1];
    if (top && top.votes >= 2 && (!runnerUp || top.votes > runnerUp.votes)) {
      finalVerdict = top.label;
      finalConfidence = 80;
      finalScores =
        top.label === "CLICKBAIT" ? { real: 20, clickbait: 80 } : { real: 80, clickbait: 20 };
      finalIndicators = [
        `Verdict adjusted based on ${top.votes} user report(s) (community feedback)`,
        ...mlResult.indicators,
      ];
    }
  } catch (err) {
    req.log.warn({ err }, "Feedback majority-vote lookup failed; using model result as-is");
  }

  try {
    const [inserted] = await db
      .insert(analysesTable)
      .values({
        headline,
        body: body ?? null,
        verdict: finalVerdict,
        confidence: String(finalConfidence),
        scoreReal: String(finalScores.real),
        scoreClickbait: String(finalScores.clickbait),
        scoreFake: "0",
        indicators: finalIndicators,
        modelUsed: finalModelUsed,
      })
      .returning();

    res.json({
      id: inserted.id,
      headline: inserted.headline,
      body: inserted.body ?? null,
      verdict: inserted.verdict,
      confidence: Number(inserted.confidence),
      scores: {
        real: Number(inserted.scoreReal),
        clickbait: Number(inserted.scoreClickbait),
      },
      indicators: inserted.indicators,
      keywords: mlResult.keywords ?? [],
      sources: mlResult.sources ?? [],
      analyzedAt: inserted.analyzedAt.toISOString(),
      modelUsed: inserted.modelUsed,
      styleAnalysis: mlResult.style_analysis
        ? {
            verdict: mlResult.style_analysis.verdict,
            clickbaitScore: mlResult.style_analysis.clickbait_score,
            credibleScore: mlResult.style_analysis.credible_score,
            indicators: mlResult.style_analysis.indicators,
            keywords: mlResult.style_analysis.keywords,
            modelUsed: mlResult.style_analysis.model_used,
          }
        : undefined,
      factCheck: mlResult.fact_check
        ? {
            status: mlResult.fact_check.status,
            note: mlResult.fact_check.note,
            sources: mlResult.fact_check.sources,
          }
        : undefined,
    });
  } catch (err) {
    req.log.error({ err }, "Database insert failed");
    res.status(500).json({ error: "Failed to save analysis result" });
  }
});

// POST /api/feedback
router.post("/feedback", async (req, res) => {
  const parse = SubmitFeedbackBody.safeParse(req.body);
  if (!parse.success) {
    res.status(400).json({ error: "Invalid request: " + parse.error.message });
    return;
  }
  const { analysisId, correctLabel } = parse.data;

  try {
    const [inserted] = await db
      .insert(feedbackTable)
      .values({ analysisId, correctLabel })
      .returning();

    const [{ total }] = await db
      .select({ total: count() })
      .from(feedbackTable);

    res.status(201).json({
      id: inserted.id,
      message: "Thank you! Your feedback helps improve the model.",
      totalFeedback: Number(total),
    });
  } catch (err) {
    req.log.error({ err }, "Feedback insert failed");
    res.status(500).json({ error: "Failed to save feedback" });
  }
});

// GET /api/history
router.get("/history", async (req, res) => {
  const parse = GetHistoryQueryParams.safeParse(req.query);
  const limit = parse.success ? (parse.data.limit ?? 20) : 20;

  try {
    const rows = await db
      .select({
        id: analysesTable.id,
        headline: analysesTable.headline,
        verdict: analysesTable.verdict,
        confidence: analysesTable.confidence,
        analyzedAt: analysesTable.analyzedAt,
        feedbackId: feedbackTable.id,
      })
      .from(analysesTable)
      .leftJoin(feedbackTable, eq(analysesTable.id, feedbackTable.analysisId))
      .orderBy(desc(analysesTable.analyzedAt))
      .limit(limit);

    const result = rows.map((r) => ({
      id: r.id,
      headline: r.headline,
      verdict: r.verdict,
      confidence: Number(r.confidence),
      analyzedAt: r.analyzedAt.toISOString(),
      hasFeedback: r.feedbackId !== null,
    }));

    res.json(result);
  } catch (err) {
    req.log.error({ err }, "History query failed");
    res.status(500).json({ error: "Failed to fetch history" });
  }
});

// GET /api/stats
router.get("/stats", async (req, res) => {
  try {
    const [totals] = await db
      .select({
        totalAnalyzed: count(),
        avgConfidence: avg(analysesTable.confidence),
      })
      .from(analysesTable);

    const verdictRows = await db
      .select({
        verdict: analysesTable.verdict,
        cnt: count(),
      })
      .from(analysesTable)
      .groupBy(analysesTable.verdict);

    const verdictCounts = { REAL: 0, CLICKBAIT: 0, UNCERTAIN: 0 } as Record<string, number>;
    for (const row of verdictRows) {
      if (row.verdict === "REAL" || row.verdict === "CLICKBAIT" || row.verdict === "UNCERTAIN") {
        verdictCounts[row.verdict] = Number(row.cnt);
      }
    }

    const [feedbackTotal] = await db.select({ total: count() }).from(feedbackTable);

    let accuracyFromFeedback: number | null = null;
    if (Number(feedbackTotal.total) >= 5) {
      const correctRows = await db
        .select({ cnt: count() })
        .from(feedbackTable)
        .innerJoin(analysesTable, eq(feedbackTable.analysisId, analysesTable.id))
        .where(eq(feedbackTable.correctLabel, analysesTable.verdict));

      const correct = Number(correctRows[0]?.cnt ?? 0);
      const total = Number(feedbackTotal.total);
      accuracyFromFeedback = total > 0 ? Math.round((correct / total) * 100) : null;
    }

    res.json({
      totalAnalyzed: Number(totals.totalAnalyzed),
      verdictCounts,
      avgConfidence: totals.avgConfidence ? Math.round(Number(totals.avgConfidence) * 10) / 10 : 0,
      totalFeedback: Number(feedbackTotal.total),
      accuracyFromFeedback,
    });
  } catch (err) {
    req.log.error({ err }, "Stats query failed");
    res.status(500).json({ error: "Failed to fetch statistics" });
  }
});

// GET /api/ml-status
router.get("/ml-status", async (req, res) => {
  try {
    const mlStatus = await getMlStatus(true);
    if (!mlStatus) throw new Error("ML status endpoint failed");

    const [feedbackTotal] = await db.select({ total: count() }).from(feedbackTable);

    res.json({
      ready: mlStatus.ready,
      modelName: mlStatus.model_name,
      feedbackCount: Number(feedbackTotal.total),
      lastRetrained: mlStatus.last_retrained ?? null,
      geminiExamplesUsed: mlStatus.gemini_examples_used ?? 0,
      geminiAccuracy: mlStatus.gemini_accuracy ?? null,
      groqExamplesUsed: mlStatus.groq_examples_used ?? 0,
      groqAccuracy: mlStatus.groq_accuracy ?? null,
      teacherExamplesUsed: mlStatus.teacher_examples_used ?? 0,
      teacherAccuracy: mlStatus.teacher_accuracy ?? null,
      readyForLocalOnly: mlStatus.ready_for_local_only ?? false,
    });
  } catch (err) {
    req.log.warn({ err }, "ML status check failed");
    res.json({
      ready: false,
      modelName: "unavailable",
      feedbackCount: 0,
      lastRetrained: null,
      geminiExamplesUsed: 0,
      geminiAccuracy: null,
      groqExamplesUsed: 0,
      groqAccuracy: null,
      teacherExamplesUsed: 0,
      teacherAccuracy: null,
      readyForLocalOnly: false,
    });
  }
});

// POST /api/retrain
// Manually triggers an immediate retrain (normally runs automatically every
// few hours). Useful to pull in newly accumulated Gemini-labeled examples
// without waiting for the scheduled interval.
router.post("/retrain", async (req, res) => {
  try {
    const retrainRes = await fetch(`${ML_SERVICE_URL}/retrain`, {
      method: "POST",
      signal: AbortSignal.timeout(60 * 60 * 1000), // retraining on the full dataset can take tens of minutes
    });
    if (!retrainRes.ok) {
      const text = await retrainRes.text();
      throw new Error(`ML retrain endpoint error ${retrainRes.status}: ${text}`);
    }
    const mlStatus = (await retrainRes.json()) as {
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
    };
    await getMlStatus(true); // refresh the shared cache immediately
    res.json({
      ready: mlStatus.ready,
      modelName: mlStatus.model_name,
      feedbackCount: mlStatus.feedback_count,
      lastRetrained: mlStatus.last_retrained ?? null,
      geminiExamplesUsed: mlStatus.gemini_examples_used ?? 0,
      geminiAccuracy: mlStatus.gemini_accuracy ?? null,
      groqExamplesUsed: mlStatus.groq_examples_used ?? 0,
      groqAccuracy: mlStatus.groq_accuracy ?? null,
      teacherExamplesUsed: mlStatus.teacher_examples_used ?? 0,
      teacherAccuracy: mlStatus.teacher_accuracy ?? null,
      readyForLocalOnly: mlStatus.ready_for_local_only ?? false,
    });
  } catch (err) {
    req.log.error({ err }, "Manual retrain trigger failed");
    res.status(503).json({ error: "Failed to trigger retraining. Is the ML service running?" });
  }
});

export default router;
