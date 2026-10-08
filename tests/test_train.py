import pandas as pd

from ml.train import evaluate_model


def test_evaluation_reports_requested_rates():
    metrics = evaluate_model(pd.Series([0, 0, 1, 1]), pd.Series([0, 1, 0, 1]))
    assert metrics["confusion_matrix"] == [[1, 1], [1, 1]]
    assert metrics["false_positive_rate"] == 0.5
    assert metrics["false_negative_rate"] == 0.5
    assert metrics["anomaly_detection_rate"] == 0.5
