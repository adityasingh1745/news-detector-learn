import { pgTable, text, serial, timestamp, numeric, integer, boolean } from "drizzle-orm/pg-core";
import { createInsertSchema } from "drizzle-zod";
import { z } from "zod/v4";

export const analysesTable = pgTable("analyses", {
  id: serial("id").primaryKey(),
  headline: text("headline").notNull(),
  body: text("body"),
  verdict: text("verdict").notNull(), // REAL | CLICKBAIT | FAKE
  confidence: numeric("confidence", { precision: 5, scale: 2 }).notNull(),
  scoreReal: numeric("score_real", { precision: 5, scale: 2 }).notNull(),
  scoreClickbait: numeric("score_clickbait", { precision: 5, scale: 2 }).notNull(),
  scoreFake: numeric("score_fake", { precision: 5, scale: 2 }).notNull(),
  indicators: text("indicators").array().notNull().default([]),
  modelUsed: text("model_used").notNull(),
  analyzedAt: timestamp("analyzed_at", { withTimezone: true }).notNull().defaultNow(),
});

export const feedbackTable = pgTable("feedback", {
  id: serial("id").primaryKey(),
  analysisId: integer("analysis_id").notNull().references(() => analysesTable.id),
  correctLabel: text("correct_label").notNull(), // REAL | CLICKBAIT | FAKE
  createdAt: timestamp("created_at", { withTimezone: true }).notNull().defaultNow(),
});

export const insertAnalysisSchema = createInsertSchema(analysesTable).omit({ id: true, analyzedAt: true });
export type InsertAnalysis = z.infer<typeof insertAnalysisSchema>;
export type Analysis = typeof analysesTable.$inferSelect;

export const insertFeedbackSchema = createInsertSchema(feedbackTable).omit({ id: true, createdAt: true });
export type InsertFeedback = z.infer<typeof insertFeedbackSchema>;
export type Feedback = typeof feedbackTable.$inferSelect;
