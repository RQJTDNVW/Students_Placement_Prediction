# Student Placement Intelligence

A separated React frontend for the Student Placement Intelligence prediction system. It presents historical prediction activity, transparent model context, student profile indicators, rule-based improvement guidance, and a meaningful placement-probability result without exposing model artifacts to the browser.

## Stack

- React 18 + TypeScript + Vite
- Tailwind CSS v4 with CSS-first semantic tokens
- React Router, Recharts, Axios
- Lucide React, Framer Motion
- React Hook Form + Zod
- FastAPI remains a separate backend responsibility

## Run locally

```bash
npm install
npm run dev
```

Copy `.env.example` to `.env` when connecting to FastAPI:

```bash
VITE_API_BASE_URL=http://127.0.0.1:5000
```

The UI starts in explicitly marked demo mode so that all pages remain useful before the REST endpoints exist. Switch to Production API in Settings after the FastAPI service is available. A failed production request never silently falls back to demo data.

## Routes

- `/` — historical dashboard overview
- `/predict` — validated student form for the V4 model
- `/predict/result` — probability gauge, risk interpretation, profile indicators, guidance, and summary actions
- `/history` — searchable, filterable, sortable, paginated prediction history
- `/analytics` — aggregate probability, status, risk, and profile views
- `/model-performance` — offline evaluation metrics, comparison chart, and pipeline
- `/about` — methods and responsible-use documentation
- `/settings` — theme, motion, data-mode, and API connection preferences

## API boundary

Production request functions live in `src/services/api.ts`. The frontend sends student input to `POST /predict`; it never reads `.joblib`, dataset, or trained-model files. Clearly marked sample responses live in `src/data/mockData.ts`.

## Verification

```bash
npm run build
npm run lint
```

The lint command currently reports only advisory maintainability warnings from the starter Oxlint configuration; it reports zero errors.
