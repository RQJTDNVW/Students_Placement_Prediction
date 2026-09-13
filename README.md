# Student Placement Intelligence

Student Placement Intelligence is an end-to-end academic machine-learning application that estimates a student's placement probability and, for placed predictions, an expected salary package. It combines a React/Vite frontend, a FastAPI backend, a reproducible V4 feature pipeline, XGBoost models, native TreeSHAP explanations, and a local prediction-history API.

> **Important:** the included dataset is synthetic. The reported metrics describe performance on that synthetic data and must not be interpreted as evidence of real-world hiring accuracy.

![Student Placement Intelligence interface](frontend/src/assets/hero.png)

## Architecture

```text
React + TypeScript + Vite
        |
        | HTTP / JSON
        v
FastAPI (app.py)
        |
        +--> V4 feature engineering (ml_pipeline/features.py)
        +--> XGBoost placement classifier
        +--> Native TreeSHAP explanations
        +--> XGBoost salary regressor
        +--> Local JSON prediction history
```

The canonical production path is **FastAPI + V4 artifacts + React**. Older V2/V3 training scripts, datasets, and model directories are retained as research history; they are not required for the current application.

## Features

- Placement probability and risk-level prediction
- Optional technical profile fields for the V4 model
- Native TreeSHAP positive and negative feature explanations
- Expected salary and salary range for placed predictions
- Dashboard, analytics, model-performance, and prediction-history pages
- Explicit demo mode and production-API mode
- Pydantic backend validation and Zod frontend validation
- Reproducible V4 training command and artifact manifest

## Project structure

| Path | Purpose |
| --- | --- |
| `app.py` | FastAPI service and inference endpoints |
| `ml_pipeline/features.py` | Shared V4 feature definitions and feature engineering |
| `ml_pipeline/train_pipeline.py` | Canonical training entry point and manifest writer |
| `model_training_tech_v4.py` | V4 classifier and salary-regressor training implementation |
| `trained_models_v4/` | Serialized V4 models, preprocessor, metrics, and manifest |
| `dataset/student_placement_synthetic.csv` | Synthetic training dataset |
| `frontend/` | React/Vite frontend |
| `test_api.py` | FastAPI integration checks |
| `test_ml_pipeline.py` | Feature and artifact contract checks |
| `ML_PIPELINE.md` | Focused training and verification notes |

Generated local files such as `frontend/node_modules/`, `frontend/dist/`, `.env` files, logs, and `prediction_history.json` are excluded by `.gitignore`.

## Setup

### Backend

From the repository root:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
py -3 -m pip install -r requirements.txt
```

Start the API:

```powershell
py -3 -m uvicorn app:app --host 127.0.0.1 --port 5000
```

The interactive API documentation is available at <http://127.0.0.1:5000/docs>.

### Frontend

In a second terminal:

```powershell
cd frontend
npm install
Copy-Item .env.example .env
npm run dev
```

Open <http://localhost:5173>. The UI starts in clearly labeled demo mode. Select **Production API** in Settings to use the FastAPI service. The API URL is configured through `frontend/.env`:

```text
VITE_API_BASE_URL=http://127.0.0.1:5000
```

## API usage

Health check:

```powershell
Invoke-RestMethod http://127.0.0.1:5000/health
```

Prediction:

```powershell
Invoke-RestMethod `
  -Uri http://127.0.0.1:5000/predict `
  -Method Post `
  -ContentType "application/json" `
  -Body '{"cgpa":8.4,"aptitude_score":82,"internships":2,"projects_count":3,"certifications":2,"communication_skills":7,"extracurricular_activities":6,"coding_skills":8,"dsa_score":7,"backlogs":0,"college_tier":"Tier-2","branch":"CSE"}'
```

Available endpoints:

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Service and model readiness |
| `POST` | `/predict` | Placement, salary, risk, and SHAP prediction |
| `GET` | `/predictions` | Read local prediction history |
| `POST` | `/predictions` | Save a prediction record |
| `DELETE` | `/predictions/{prediction_id}` | Delete a saved record |
| `GET` | `/dashboard/statistics` | History-based dashboard statistics |
| `GET` | `/model/metrics` | Offline model evaluation values |
| `GET` | `/analytics` | Analytics payload for the frontend |

## Model metrics

The current V4 metrics were generated from the included **synthetic** dataset:

| Task | Metric | Value |
| --- | --- | ---: |
| Placement classifier | Accuracy | 69.69% |
| Placement classifier | Precision | 71.56% |
| Placement classifier | Recall | 92.49% |
| Placement classifier | F1 score | 80.69% |
| Placement classifier | ROC-AUC | 0.6822 |
| Placement classifier | PR-AUC | 0.8154 |
| Salary regressor | R² | 0.7880 |
| Salary regressor | MAE | 0.95 LPA |
| Salary regressor | RMSE | 1.19 LPA |

These are offline test-set measurements, not live accuracy guarantees. The application should support human review and should not be used as the sole basis for academic or employment decisions.

## Retraining and verification

Retrain the canonical V4 artifacts:

```powershell
py -3 -m ml_pipeline.train_pipeline
```

Run the available checks:

```powershell
py -3 test_ml_pipeline.py
py -3 test_api.py
cd frontend
npm run build
npm run lint
```

## Limitations and responsible use

- The primary dataset is synthetic and may not represent actual institutions, hiring markets, or student populations.
- Historical or generated labels can encode bias and unequal opportunity.
- Probability calibration, fairness analysis, drift monitoring, authentication, and multi-user storage are not implemented.
- Prediction history currently uses a local JSON file and is intended for a single-machine demonstration.
- Production deployment requires security hardening, restricted CORS, a database, observability, access control, and validation against an appropriately governed real dataset.
