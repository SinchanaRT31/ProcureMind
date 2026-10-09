from pathlib import Path

import joblib
from sklearn.ensemble import IsolationForest

from ml.predict import predict_procurement_data
from ml.preprocess import ProcurementFeatureTransformer
from test_preprocess import raw_frame



def test_prediction_loads_matching_model_and_transformer(tmp_path: Path):
    raw = raw_frame().iloc[:3]
    transformer = ProcurementFeatureTransformer().fit(raw)
    model = IsolationForest(n_estimators=5, random_state=42).fit(transformer.transform(raw))
    model_path, transformer_path = tmp_path / "model.pkl", tmp_path / "transformer.pkl"
    joblib.dump(model, model_path)
    joblib.dump(transformer, transformer_path)
    result = predict_procurement_data(raw.drop(columns="is_anomaly"), model_path, transformer_path)
    assert result.columns.tolist() == ["transaction_id", "predicted_anomaly", "ml_anomaly_score"]
    assert len(result) == len(raw)
