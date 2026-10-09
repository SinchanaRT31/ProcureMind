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
- An optional SHAP explainability layer that attributes the exact Isolation Forest anomaly score without altering predictions, evidence, or risk logic.

The project does not currently implement FastAPI, MongoDB, an LLM investigator, or a production database.

## Run the ML baseline

```bash
python ml/train.py
```

Generated, ignored artifacts in `ml/model/` include the model, fitted transformer, feature order, held-out metrics, split metadata, and test predictions. Label and downstream outcome fields—including `risk_score`, `risk_level`, `investigation_status`, and `potential_savings`—are never model inputs.
Training also saves `shap_background_features.pkl`: a deterministic sample of up to 50 transformed training rows, without labels or outcome fields. Explanations require this artifact; they fail clearly if it is absent rather than using prediction rows as a reference.

The evidence engine derives moderate/high thresholds from the training partition's 95th/99th percentiles of each deviation measure. Amount-tail alerts use a training-derived department/category empirical percentile with a global fallback. Vendor frequency, vendor rating, department, item category, purchase type, payment status, and vendor location are contextual information only and do not independently create an anomaly alert.

Score raw data after training:

```bash
python ml/predict.py path/to/raw_procurement.csv --output ml/model/predictions.csv
```

Append evidence and investigation-priority output without changing the default CLI result:

```bash
python ml/predict.py path/to/raw_procurement.csv --include-risk
```

Append SHAP feature-contribution summaries without changing the default CLI result. Explanations are opt-in, use batches of 10 rows by default, and process every requested row; runtime grows with the number of rows and features:

```bash
python ml/predict.py path/to/raw_procurement.csv --include-explanations
```

The exact explained function is `f(X) = -model.score_samples(X)`, the same anomaly score returned as `ml_anomaly_score`; higher scores mean stronger model anomaly signals. Permutation SHAP uses the saved training-only background as its baseline. A positive contribution raises the score relative to that baseline, and a negative contribution lowers it. The code checks that baseline plus contributions reconstructs the score within absolute tolerance `1e-6` (and relative tolerance `1e-6`).

Use `--explanation-batch-size N` to change rows evaluated per SHAP call. Batching controls working memory, not total computation: do not request explanations for very large files without allowing for longer runtimes. The standalone command accepts the same `--batch-size` option.

Or generate the explanation table directly:

```bash
python ml/explain.py path/to/raw_procurement.csv --top-k 5
```

Input data must contain the raw fields required by `ml.preprocess`; it does not need labels or outcome columns. SHAP attributes this model score only; it is separate from behavioral evidence, document verification, and investigation priority. Contributions describe how the model score changes relative to the selected background, not fraud, intent, causality, or vendor wrongdoing. Permutation SHAP estimates feature attributions, and its result depends on the training reference sample and feature dependence assumptions.

The risk layer uses ML-score 95th/99th training percentiles for normal/elevated/high ML context. A single strongest signal per evidence family contributes to priority, preventing three correlated price deviations from being counted as three independent reasons. One moderate family or ML-only indication is MEDIUM; a high family or multiple families is HIGH; two high independent families is CRITICAL.

## Tests

```bash
python -m pytest
```

## Dashboard

```bash
python backend/main.py
```

Open `http://127.0.0.1:8000`. The dashboard continues to use sample transaction and evidence data; it is not connected to live ML predictions.

## Human review workflow

Review cases can be opened from a dashboard transaction or through the API. Review status is separate from any decision: statuses are `OPEN`, `REVIEWING`, and `RESOLVED`; decisions are `CONFIRMED_ISSUE`, `NO_ISSUE_FOUND`, `NEEDS_MORE_INFORMATION`, or `ESCALATED`. `CONFIRMED_ISSUE` is a reviewer-entered investigation outcome, not an automatic fraud finding. Reviewer identifiers are optional text; the application has no authentication or authorization.

Cases and review history are persisted in the ignored runtime file `backend/data/investigations.json`, separate from `sample_data.json` and all generated ML artifacts. The first API read or write idempotently seeds existing sample cases into the review store. Later starts fill in any newly added sample cases by ID while preserving saved review status, decisions, notes, and history. Updates are serialized within the server process and written using a temporary file plus atomic replacement. The sample JSON remains unchanged by case reviews.

The dashboard supports opening a case, changing its review status, recording or updating a decision, adding notes, and viewing event history. Existing dashboard case fields (`id`, `transaction_id`, `status`, and `notes`, among others) remain in API responses for compatibility. Legacy `POST /api/cases/{case_id}` accepts existing status labels and the `notes` field; submitted notes are appended to preserve history.

## Deterministic investigator report

The read-only report API accepts only an exact transaction ID from the fixed trusted dataset at `dataset/ProcureMind/procuremind_procurement_dataset.csv`:

```text
GET /api/investigations/report?transaction_id=TXN-0000001
```

A successful response wraps the report in `{"report": ...}`. The report schema is versioned and contains:

- `transaction_id` and `schema_version`.
- `why_flagged`: the Isolation Forest prediction and anomaly score, plus SHAP contributions where available. Positive contributions raise the model anomaly score relative to the training reference; the score is not a probability.
- `supporting_findings`: selected evidence and priority summary returned by the existing risk engine, retaining that source attribution.
- `inconclusive_or_conflicting`: duplicate candidates, consistency checks, and verification signals from the existing verification engine. These findings require human verification.
- `missing_information`: supporting documents, approvals, and business justification not supplied to this report service.
- `follow_up_checks`: the existing risk engine's recommended actions.
- `limitations`: interpretation and human-review boundaries.

The service reads the fixed source in chunks, matches `transaction_id` by exact equality, and passes only the 14 required raw fields to the existing predictor. It does not accept caller-supplied paths, CSV data, or transaction objects; source labels and outcome fields are excluded. Missing IDs return 404, duplicate exact IDs return 409, invalid parameters return 400, and unavailable source/model/explanation artifacts return 503. Other unexpected failures return a generic 500 response.

The model, fitted transformer, SHAP background, evidence engine, risk engine, and verification engine artifacts generated by `python ml/train.py` must be present under `ml/model/`. Report generation calls the existing predictor with risk and explanations enabled. Its current interface loads artifacts during each call; warm single-row inference was about 0.28 seconds in the audit environment, while the first explanation call took about 19 seconds due to environment-specific SHAP/Matplotlib cache initialization. Cold-start and request times vary by environment.

Reports assist human investigation. They do not determine fraud and never create or change Phase 6 decisions or review history. The dashboard still uses sample transaction IDs and has no trusted mapping to dataset IDs, so reports are not attached to sample cases and the frontend remains unchanged.

API routes:

- `GET /api/cases` — list cases.
- `GET /api/cases/{case_id}` — retrieve a case and its history.
- `GET /api/cases/{case_id}/history` — retrieve review events.
- `POST /api/cases` with `{"transaction_id": "tx-2048"}` — open a case; returns an existing active case for that transaction when present.
- `PATCH /api/cases/{case_id}` — update `review_status`, `investigation_decision`, optional `reviewer_id`, and/or append a `note`.
- `POST /api/cases/{case_id}` — compatible legacy update route; also accepts the validated review fields.
- `GET /api/investigations/report?transaction_id=TXN-0000001` — read-only deterministic report for an exact trusted dataset ID.
- `GET /api/dashboard`, `GET /api/transactions`, and `GET /api/health` remain available.

Human review is separate from anomaly detection, SHAP model explanations, behavioral evidence, and document verification. The UI shows sample transaction context and does not claim those fields are live ML outputs. Review history is application-level history, not a tamper-proof audit log. The process lock does not coordinate multiple server processes or hosts; this local JSON store is not intended as a production multi-instance database.
