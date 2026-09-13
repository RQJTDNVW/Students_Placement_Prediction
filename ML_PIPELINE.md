# Canonical V4 ML pipeline

The production path uses the V4 technical placement classifier and salary regressor.
The shared feature definitions live in `ml_pipeline/features.py`, so training and
FastAPI inference use the same derived features and feature order.

## Train everything

From the project root:

```powershell
py -m ml_pipeline.train_pipeline
```

This validates the dataset, trains both XGBoost models, evaluates them, and writes:

- `trained_models_v4/tech_preprocessor_v4.joblib`
- `trained_models_v4/best_tech_model_xgboost_v4.joblib`
- `trained_models_v4/salary_regressor_v4.joblib`
- `trained_models_v4/feature_names_v4.json`
- `trained_models_v4/tech_model_metrics_v4.json`
- `trained_models_v4/model_manifest.json`

The salary model is trained only on rows with `placement_status == 1` and a valid
`salary_package_lpa`. The current dataset is synthetic, so metrics describe this
dataset and are not guarantees of real-world placement or salary outcomes.

## Verify

```powershell
py test_ml_pipeline.py
py test_api.py
```

Run the API with `py -m uvicorn app:app --host 127.0.0.1 --port 5000` and the
React application with `cd frontend; npm run dev`.
