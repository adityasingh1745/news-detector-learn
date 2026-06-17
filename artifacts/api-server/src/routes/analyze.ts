import { Router } from "express";
import { db, analysesTable, feedbackTable } from "@workspace/db";
import { eq, desc, count, avg } from "drizzle-orm";
import { AnalyzeNewsBody, SubmitFeedbackBody, GetHistoryQueryParams } from "@workspace/api-zod";
import { logger } from "../lib/logger";

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
    model_used: string;
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

  try {
    const [inserted] = await db
      .insert(analysesTable)
      .values({
        headline,
        body: body ?? null,
        verdict: mlResult.verdict,
        confidence: String(mlResult.confidence),
        scoreReal: String(mlResult.scores.real),
        scoreClickbait: String(mlResult.scores.clickbait),
        scoreFake: "0",
        indicators: mlResult.indicators,
        modelUsed: mlResult.model_used,
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
      analyzedAt: inserted.analyzedAt.toISOString(),
      modelUsed: inserted.modelUsed,
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

    const verdictCounts = { REAL: 0, CLICKBAIT: 0 } as Record<string, number>;
    for (const row of verdictRows) {
      if (row.verdict === "REAL" || row.verdict === "CLICKBAIT") {
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
    const statusRes = await fetch(`${ML_SERVICE_URL}/status`, {
      signal: AbortSignal.timeout(5000),
    });
    if (!statusRes.ok) throw new Error("ML status endpoint failed");
    const mlStatus = (await statusRes.json()) as {
      ready: boolean;
      model_name: string;
      last_retrained: string | null;
    };

    const [feedbackTotal] = await db.select({ total: count() }).from(feedbackTable);

    res.json({
      ready: mlStatus.ready,
      modelName: mlStatus.model_name,
      feedbackCount: Number(feedbackTotal.total),
      lastRetrained: mlStatus.last_retrained ?? null,
    });
  } catch (err) {
    req.log.warn({ err }, "ML status check failed");
    res.json({
      ready: false,
      modelName: "unavailable",
      feedbackCount: 0,
      lastRetrained: null,
    });
  }
});

export default router;
