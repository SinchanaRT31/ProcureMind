# ProcureMind

ProcureMind is an enterprise procurement anomaly-detection project. The current implementation pairs a leakage-safe scikit-learn baseline pipeline with a separate static investigation dashboard.

## Current implementation

- 200,000 labelled procurement transactions for evaluation.
- Isolation Forest baseline: `n_estimators=200`, `contamination="auto"`, `random_state=42`, and `n_jobs=-1`.
- A train-fitted `ProcurementFeatureTransformer` that derives vendor, department, and vendor-category behavioral baselines, frequency encodings, and imputation values from training rows only.
- An 80/20 stratified split (`random_state=42`) evaluated only on held-out data with precision, recall, F1, confusion matrix, anomaly detection rate, false-positive rate, and false-negative rate.
- A prediction module that loads the paired model and preprocessing transformer to score new raw procurement data.
- A separate, transparent evidence engine that produces training-thresholded price, spend, quantity, and timing signals alongside contextual vendor and categorical information. It does not produce a fraud probability or combined risk score.
- A deterministic investigation-priority layer that compresses correlated signals into PRICE, SPEND, QUANTITY, and TIMING families. It returns LOW/MEDIUM/HIGH/CRITICAL review priorities and recommended actions; it is a triage heuristic, not an autonomous fraud decision.
- Deterministic document verification: potential duplicate candidates based on exact business metadata and close invoice dates, plus date/arithmetic consistency checks. A separate chronological vendor-cadence experiment is reported independently from the random-split benchmark.
- A vanilla HTML/CSS/JavaScript dashboard served by a Python standard-library HTTP server. It uses sample dashboard data and is not yet connected to ML output.

The project does not currently implement FastAPI, MongoDB, SHAP, an LLM investigator, or a production database.

## Run the ML baseline

```bash
python ml/train.py
```

Generated, ignored artifacts in `ml/model/` include the model, fitted transformer, feature order, held-out metrics, split metadata, and test predictions. Label and downstream outcome fields—including `risk_score`, `risk_level`, `investigation_status`, and `potential_savings`—are never model inputs.

The evidence engine derives moderate/high thresholds from the training partition's 95th/99th percentiles of each deviation measure. Amount-tail alerts use a training-derived department/category empirical percentile with a global fallback. Vendor frequency, vendor rating, department, item category, purchase type, payment status, and vendor location are contextual information only and do not independently create an anomaly alert.

Score raw data after training:

```bash
python ml/predict.py path/to/raw_procurement.csv --output ml/model/predictions.csv
```

Append evidence and investigation-priority output without changing the default CLI result:

```bash
python ml/predict.py path/to/raw_procurement.csv --include-risk
```

Input data must contain the raw fields required by `ml.preprocess`; it does not need labels or outcome columns.

The risk layer uses ML-score 95th/99th training percentiles for normal/elevated/high ML context. A single strongest signal per evidence family contributes to priority, preventing three correlated price deviations from being counted as three independent reasons. One moderate family or ML-only indication is MEDIUM; a high family or multiple families is HIGH; two high independent families is CRITICAL.

## Tests

```bash
python -m pytest
```

## Dashboard

```bash
python backend/main.py
```

Open `http://127.0.0.1:8000`. The existing dashboard and case workflow are unchanged.
