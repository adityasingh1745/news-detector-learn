# News Detector

A web app I built to detect whether a news headline (and article body, if you paste one in) is **Real**, **Clickbait**, or **Fake**. It combines a trained ML model with rule-based heuristics and LLM cross-checking (Gemini/Groq), and it learns from user corrections over time.

Live: https://news-detector-learn.vercel.app/

## Why I built this

Clickbait and misinformation are everywhere, and most "fake news detectors" out there are either black-box APIs or toy projects that only look at the headline. I wanted something that:

- actually explains *why* it flagged something (indicators, not just a score)
- cross-checks claims against real news coverage instead of guessing blind
- improves itself from feedback instead of staying static after training

## How it works

When you submit a headline:

1. The API server sends it to the Python ML service.
2. The ML service runs a TF-IDF + Logistic Regression model (trained on a large labeled headline dataset) plus a bunch of rule-based checks — capitalization ratio, sensational phrasing, vague/no-source wording, credibility signals, etc.
3. Gemini and Groq are used as a secondary opinion and to pull in related real news coverage for fact-checking.
4. The final verdict blends the ML score (~60%) with the rule-based score (~40%).
5. You get back a verdict, confidence %, and the specific indicators that drove the decision.

If you think a verdict is wrong, you can correct it. That correction gets stored and is used in periodic retraining, so the local model gradually gets better at matching what Gemini/Groq + users agree on, and relies on them less over time.

## Architecture

```
React frontend  --->  Node/Express API server  --->  Python FastAPI ML service
(Vite, Tailwind)       (port 8080, Postgres)          (port 8001, scikit-learn)
                             |
                        Gemini / Groq APIs
                        (classification + fact-check sourcing)
```

- **Frontend** is a plain React app, talks to the API server over `/api/*`.
- **API server** does validation, stores analyses + feedback in Postgres, and calls the ML service and the LLM APIs.
- **ML service** owns the actual model and does inference, plus a background loop that retrains periodically on accumulated feedback.

## Stack

- Frontend: React 19, Vite, TypeScript, Tailwind, shadcn/ui, TanStack Query
- API server: Node.js, Express 5, Zod, Drizzle ORM, pino
- ML service: Python, FastAPI, scikit-learn, joblib
- DB: PostgreSQL (Neon in production)
- Monorepo: pnpm workspaces
- Deployed on Vercel (frontend) + Render (API server & ML service)

## Project layout

```
artifacts/
  api-server/       Express API (port 8080)
  news-detector/     React frontend
  mockup-sandbox/    UI playground
lib/
  api-spec/          OpenAPI spec (source of truth for API contracts)
  api-zod/           Generated Zod schemas
  api-client-react/  Generated React Query hooks
  db/                Drizzle schema + Postgres client
services/
  ml-api/            FastAPI ML service (main.py, train_model.py, models/)
.github/workflows/
  keep-alive.yml     Pings prod services every 10 min so they don't cold-start
start.ps1            One-command local startup (Windows)
render.yaml          Render deployment config
```

## Running it locally

You'll need Node 22+, pnpm 9+, Python 3.11 or 3.12, and a Postgres database (local or hosted — I use Neon).

```powershell
# install JS deps
pnpm install

# install Python deps for the ML service
python -m pip install -r services/ml-api/requirements.txt

# push the DB schema (after setting DATABASE_URL — see below)
pnpm --filter @workspace/db run push
```

Set up env vars before running anything:

**`artifacts/api-server/.env`**
```
DATABASE_URL=postgres://user:password@host:5432/dbname
ML_SERVICE_URL=http://localhost:8001
GEMINI_API_KEY=your-gemini-key
GROQ_API_KEY=your-groq-key
```

**`lib/db/.env`**
```
DATABASE_URL=postgres://user:password@host:5432/dbname
```

**`services/ml-api/.env`** (optional)
```
ML_PORT=8001
DATABASE_URL=postgres://user:password@host:5432/dbname
NEWS_API_KEY=your-key          # optional, for fact-check source lookup
GOOGLE_FACTCHECK_API_KEY=key   # optional
```

Then start the three services — easiest way on Windows is just:

```powershell
.\start.ps1
```

which opens three windows for the ML service (8001), API server (8080), and frontend (Vite, usually 5173).

Or run them manually in separate terminals if you prefer:

```powershell
cd services/ml-api && python main.py
pnpm --filter @workspace/api-server run dev
pnpm --filter @workspace/news-detector run dev
```

## Other useful commands

- `pnpm run typecheck` — typecheck everything
- `pnpm run build` — typecheck + build
- `pnpm --filter @workspace/api-spec run codegen` — regenerate API types/hooks after editing `openapi.yaml`
- `python services/ml-api/train_model.py` — retrain the model from scratch

## Deployment notes

Frontend is on Vercel, which proxies `/api/*` to the Render-hosted API server (see `artifacts/news-detector/vercel.json`). The API server and ML service are two separate Render web services (`render.yaml`).

Render's free tier spins services down after ~15 min of no traffic, which causes a cold-start delay (and can look like "model unavailable" on the first request after idling). To avoid that, `keep-alive.yml` pings both services every 10 minutes via GitHub Actions so they stay warm.

## Troubleshooting

- **"Use pnpm instead" error on install** — this repo blocks `npm`/`yarn`, use `pnpm install`.
- **"Model unavailable" on the live site** — usually a Render cold start; reload after ~30-60s, or check the Actions tab to confirm the keep-alive workflow is running.
- **API server won't start** — check `DATABASE_URL` is set correctly in both `.env` files.

---

Author: **Aditya Singh**
