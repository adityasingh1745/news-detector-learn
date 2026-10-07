# News Detector — Fake News & Clickbait Detector

A web app that detects whether a news headline (and optional article body) is
**REAL**, **CLICKBAIT**, or **FAKE**, using a hybrid pipeline of a trained
TF‑IDF + Logistic Regression model, rule‑based heuristics, and LLM fallbacks
(Gemini / Groq). Users can submit corrections ("feedback") on verdicts, which
the system uses to continuously retrain and improve the local model over time.

**Live demo:** https://news-detector-learn.vercel.app/

---

## Table of contents

- [What this project does](#what-this-project-does)
- [Architecture](#architecture)
- [Tech stack](#tech-stack)
- [Project structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Environment variables](#environment-variables)
- [Getting started (local development)](#getting-started-local-development)
- [Running everything with one script (Windows)](#running-everything-with-one-script-windows)
- [Useful scripts](#useful-scripts)
- [Deployment](#deployment)
- [Keeping the deployed services warm](#keeping-the-deployed-services-warm)
- [Troubleshooting](#troubleshooting)

---

## What this project does

1. A user pastes a news **headline** (and optionally the article **body**).
2. The backend sends the text to the ML service, which:
   - Runs a trained **TF‑IDF + Logistic Regression** classifier (trained on a
     large labeled headline dataset).
   - Applies **rule-based heuristics** (capitalization ratio, sensational
     phrase matching, credibility signals, vague wording detection, etc.).
   - Optionally cross-checks the claim against **Gemini** and **Groq** LLMs
     and real news sources for fact-checking / corroboration.
3. The three-way score (REAL / CLICKBAIT / FAKE) is blended (~60% ML + 40%
   rules) into a final **verdict + confidence score + indicators**.
4. The result, along with any linked source articles, is shown to the user.
5. Users can mark a verdict as correct/incorrect. This feedback is stored and
   periodically used to **retrain** the local model, gradually reducing
   reliance on the Gemini/Groq APIs ("distillation").

---

## Architecture

```
┌────────────────────┐      HTTP       ┌──────────────────────┐      HTTP      ┌──────────────────────┐
│   React Frontend    │ ───────────▶   │   Node.js API Server  │ ───────────▶  │   Python ML Service    │
│ (Vite, Tailwind,     │  /api/*        │ (Express 5, port 8080)│  /predict      │ (FastAPI, port 8001)   │
│  shadcn/ui)          │ ◀───────────   │                        │ ◀───────────  │                        │
└────────────────────┘                 └──────────┬─────────────┘                └──────────┬─────────────┘
                                                    │                                         │
                                          Postgres (Drizzle ORM)                 TF-IDF + LogReg model (joblib)
                                          analyses / feedback tables             + rule-based scoring
                                                    │                                         │
                                        Gemini API / Groq API (fallback classification + fact-check sourcing)
```

- **Frontend** — pure UI, calls the API server over `/api/*`.
- **API server (Node/Express)** — validates requests, persists analyses &
  feedback to Postgres, calls the ML service for predictions, and calls
  Gemini/Groq as an additional classification signal / fact-check source.
- **ML service (Python/FastAPI)** — owns the trained model, does inference,
  and runs a background loop that periodically retrains on accumulated user
  feedback + distilled Gemini/Groq judgments.

---

## Tech stack

| Layer      | Technology |
|------------|------------|
| Frontend   | React 19, Vite, TypeScript, Tailwind CSS, shadcn/ui, TanStack Query, wouter |
| API server | Node.js 22+, Express 5, Zod, Drizzle ORM, pino (logging) |
| ML service | Python 3.11/3.12, FastAPI, uvicorn, scikit-learn, joblib |
| Database   | PostgreSQL (Neon serverless Postgres in production) |
| Package mgmt | pnpm workspaces (monorepo), pip (Python) |
| Build      | esbuild (API server bundle), Vite (frontend bundle) |
| API contracts | OpenAPI spec + Orval codegen (Zod schemas + React Query hooks) |
| Deployment | Vercel (frontend) + Render (API server & ML service) |

---

## Project structure

```
.
├── artifacts/
│   ├── api-server/         # Express API server (port 8080)
│   ├── news-detector/      # React frontend (Vite)
│   └── mockup-sandbox/     # UI sandbox/playground
├── lib/
│   ├── api-spec/           # OpenAPI spec — source of truth for API contracts
│   ├── api-zod/            # Generated Zod schemas (from OpenAPI)
│   ├── api-client-react/   # Generated React Query hooks (from OpenAPI)
│   └── db/                 # Drizzle ORM schema + Postgres client
├── services/
│   └── ml-api/             # Python FastAPI ML inference service (port 8001)
│       ├── main.py             # FastAPI app, /predict, /status, /retrain, /healthz
│       ├── train_model.py      # Trains the TF-IDF + LogReg model
│       ├── train_distilbert.py # Experimental DistilBERT fine-tuning
│       ├── retrain.py          # Scheduled retraining logic
│       ├── features.py         # Feature engineering helpers
│       └── models/             # Trained model artifacts (joblib, DistilBERT)
├── scripts/                 # Workspace-level scripts
├── .github/workflows/
│   └── keep-alive.yml       # Pings prod services every 10 min to prevent cold starts
├── start.ps1                # One-command local startup script (Windows)
├── render.yaml               # Render deployment config (API + ML services)
├── pnpm-workspace.yaml        # pnpm monorepo workspace config
└── package.json
```

---

## Prerequisites

Install these before running locally:

| Tool | Version | Notes |
|------|---------|-------|
| [Node.js](https://nodejs.org/) | 22+ | JS runtime |
| [pnpm](https://pnpm.io/) | 9+ | Package manager (**required** — npm/yarn are blocked by a preinstall check) |
| [Python](https://www.python.org/) | 3.11–3.12 | For the ML service |
| [PostgreSQL](https://www.postgresql.org/) | any recent version | Or use a hosted Postgres (e.g. [Neon](https://neon.tech)) |

Install pnpm if you don't have it:
```powershell
npm install -g pnpm
```

---

## Environment variables

Each service reads its own `.env` file.

### `artifacts/api-server/.env`
```env
DATABASE_URL=postgres://user:password@host:5432/dbname
ML_SERVICE_URL=http://localhost:8001
GEMINI_API_KEY=your-gemini-api-key
GROQ_API_KEY=your-groq-api-key
```

### `lib/db/.env`
```env
DATABASE_URL=postgres://user:password@host:5432/dbname
```

### `services/ml-api/.env` (optional)
```env
DATABASE_URL=postgres://user:password@host:5432/dbname
ML_PORT=8001
NEWS_API_KEY=your-news-api-key          # optional, used for fact-check source lookup
GOOGLE_FACTCHECK_API_KEY=your-api-key   # optional
RETRAIN_INTERVAL_SECONDS=21600          # optional, default 6 hours
```

| Variable | Required | Description |
|----------|----------|--------------|
| `DATABASE_URL` | ✅ | Postgres connection string, shared by API server, DB package, and ML service |
| `ML_SERVICE_URL` | API server only | URL of the Python ML service (default `http://localhost:8001`) |
| `GEMINI_API_KEY` | For LLM fallback/fact-check | Google Gemini API key |
| `GROQ_API_KEY` | For LLM fallback/fact-check | Groq API key |
| `NEWS_API_KEY` | Optional | Used to corroborate claims with real news coverage |
| `GOOGLE_FACTCHECK_API_KEY` | Optional | Google Fact Check Tools API key |
| `ML_PORT` | Optional | Port for the ML service (default `8001`) |

> ⚠️ Never commit real API keys/secrets to git. The `.env` files already exist
> locally in this repo for development convenience — treat them as sensitive
> and rotate the keys if this repo is ever made public.

---

## Getting started (local development)

### 1. Install dependencies
```powershell
pnpm install
python -m pip install -r services/ml-api/requirements.txt
```

### 2. Set up the database
Create a Postgres database and set `DATABASE_URL` in both `.env` files above, then push the schema:
```powershell
pnpm --filter @workspace/db run push
```

### 3. Start each service (3 terminals)

**Terminal 1 — ML service (port 8001):**
```powershell
cd services/ml-api
python main.py
```

**Terminal 2 — API server (port 8080):**
```powershell
pnpm --filter @workspace/api-server run dev
```

**Terminal 3 — Frontend (Vite, auto-assigned port, usually 5173):**
```powershell
pnpm --filter @workspace/news-detector run dev
```

Then open the URL printed by the Vite terminal (e.g. `http://localhost:5173`).

---

## Running everything with one script (Windows)

Instead of opening 3 terminals manually, run:
```powershell
.\start.ps1
```
This launches the ML service, API server, and frontend each in their own
PowerShell window, using the existing `.env` files for configuration.

---

## Useful scripts

| Command | Description |
|---------|-------------|
| `pnpm install` | Install all workspace dependencies |
| `pnpm run typecheck` | Typecheck all packages |
| `pnpm run build` | Typecheck + build all packages |
| `pnpm --filter @workspace/api-server run dev` | Run the API server |
| `pnpm --filter @workspace/news-detector run dev` | Run the frontend |
| `pnpm --filter @workspace/db run push` | Push Drizzle schema changes to Postgres (dev only) |
| `pnpm --filter @workspace/api-spec run codegen` | Regenerate Zod schemas + React Query hooks from `openapi.yaml` |
| `python services/ml-api/main.py` | Run the ML service directly |
| `python services/ml-api/train_model.py` | Retrain the TF-IDF + LogReg model from scratch |

> After changing `lib/api-spec/openapi.yaml`, always re-run the `codegen`
> script so the frontend/API types stay in sync.

---

## Deployment

- **Frontend** → deployed on **Vercel**. See `artifacts/news-detector/vercel.json` —
  it builds the Vite app and rewrites `/api/*` requests to the Render API server.
- **API server** + **ML service** → deployed on **Render** (free tier), as
  two separate web services defined in `render.yaml`.

Production URLs:
- Frontend: https://news-detector-learn.vercel.app/
- API server: https://news-detector-api-e753.onrender.com
- ML service: https://news-detector-ml.onrender.com

---

## Keeping the deployed services warm

Render's **free tier** spins services down after ~15 minutes of inactivity.
The next request then triggers a cold start (~30–60s), which can surface as
a **"model unavailable"** error on the first visit.

To prevent this, `.github/workflows/keep-alive.yml` runs a scheduled GitHub
Action **every 10 minutes** that pings both services' health endpoints
(`/api/healthz` and `/healthz`), keeping them continuously warm. You can check
its run history under the repo's **Actions** tab.

If you ever still see "model unavailable", simply reload the page after ~30-60
seconds — the service will have finished waking up.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `Use pnpm instead` error on install | Used `npm install`/`yarn install` | Use `pnpm install` — enforced by a preinstall check |
| Frontend shows "model unavailable" | Render service cold start | Reload after ~30-60s, or check the keep-alive workflow is running |
| API server fails on startup | `DATABASE_URL` missing/invalid | Set it in `artifacts/api-server/.env` and `lib/db/.env` |
| ML service responds slowly on first request | Model files loading from disk | Expected — subsequent requests are fast |
| `pnpm --filter ... run dev` does nothing on Windows | Shell-specific script syntax (`export ...`) | Already handled by `start.ps1`; otherwise run `pnpm run dev` inside a bash-compatible shell or set `NODE_ENV` manually before running |
