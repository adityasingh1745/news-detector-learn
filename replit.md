# EVAL.OSINT — Fake News & Clickbait Detector

A web app that uses RoBERTa (HuggingFace) and rule-based heuristics to detect whether a news headline/article is **Real**, **Clickbait**, or **Fake**. Users can submit labeled corrections that accumulate as training samples.

## Run & Operate

- `pnpm --filter @workspace/api-server run dev` — run the Node.js API server (port 8080)
- `python3 services/ml-api/main.py` — run the Python ML service (port 8001, ML_PORT env var)
- `pnpm --filter @workspace/news-detector run dev` — run the React frontend (port auto-assigned)
- `pnpm run typecheck` — full typecheck across all packages
- `pnpm run build` — typecheck + build all packages
- `pnpm --filter @workspace/api-spec run codegen` — regenerate API hooks and Zod schemas from the OpenAPI spec
- `pnpm --filter @workspace/db run push` — push DB schema changes (dev only)
- Required env: `DATABASE_URL` — Postgres connection string
- Optional env: `ML_SERVICE_URL` — URL of Python ML service (default: `http://localhost:8001`)

## Stack

- pnpm workspaces, Node.js 24, TypeScript 5.9
- Frontend: React + Vite + Tailwind CSS + shadcn/ui
- API: Express 5 (Node.js, port 8080)
- ML Service: FastAPI + uvicorn (Python 3.11, port 8001)
- ML Model: `hamzab/roberta-fake-news-classification` (RoBERTa, ~499MB), with rule-based fallback
- DB: PostgreSQL + Drizzle ORM
- Validation: Zod (`zod/v4`), `drizzle-zod`
- API codegen: Orval (from OpenAPI spec)
- Build: esbuild (CJS bundle)

## Where things live

- `lib/api-spec/openapi.yaml` — OpenAPI spec (source of truth for API contracts)
- `lib/db/src/schema/analyses.ts` — DB schema for analyses and feedback tables
- `artifacts/api-server/src/routes/analyze.ts` — all analysis/feedback/stats routes
- `services/ml-api/main.py` — Python FastAPI ML inference service
- `artifacts/news-detector/src/pages/Home.tsx` — main frontend page

## Architecture decisions

- **Two-service backend**: Node.js Express handles HTTP routing, DB persistence, and validation. Python FastAPI handles ML inference only. Node.js calls Python at `localhost:8001` internally.
- **Verdict system**: Three-way classification — REAL / CLICKBAIT / FAKE. Clickbait detection is rule-based (heuristics on headline), fake/real is ML-based (RoBERTa). Scores are blended 60% ML + 40% rules.
- **Learning via feedback**: Users can mark whether a verdict was correct. Feedback is stored in the `feedback` table. Model accuracy is computed from feedback when ≥5 samples exist.
- **Fallback classifier**: If HuggingFace model fails to load, the service uses a comprehensive rule-based classifier (phrase matching, capitalization ratio, credibility signals).

## Product

- Users paste a news headline and optional article body text
- The system returns: verdict (REAL/CLICKBAIT/FAKE), confidence % (0-100), three-way score breakdown, and detected linguistic indicators
- Recent analysis history is shown on the right panel with stats
- Users can correct verdicts to improve the model over time

## User preferences

_Populate as you build — explicit user instructions worth remembering across sessions._

## Gotchas

- The Python ML service downloads the RoBERTa model (~499MB) on first startup — takes ~30s
- Always run `pnpm --filter @workspace/api-spec run codegen` after changing `openapi.yaml`
- The ML service port is 8001 by default; set `ML_SERVICE_URL` env var on the API server if changed
- `pnpm run dev` at workspace root is intentionally missing — use per-artifact filter commands

## Running locally

See the "How to run locally" instructions provided to the user in chat.
