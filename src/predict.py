"""Load a saved fitted pipeline and classify raw KDD feature dictionaries."""
from pathlib import Path
import joblib
import pandas as pd

try:
    from .data_preprocessing import KDD_COLUMNS
except ImportError:
    from data_preprocessing import KDD_COLUMNS


def predict_network_traffic(input_data, model_path: str | Path = "models/random_forest.pkl"):
    """Return human-readable prediction(s) for one dict or a sequence of dicts.

    Dicts must contain KDD feature names (excluding label and target). Missing features
    are treated with the same imputation learned by the fitted pipeline.
    """
    model = joblib.load(model_path)
    frame = pd.DataFrame([input_data] if isinstance(input_data, dict) else input_data)
    expected = [column for column in KDD_COLUMNS if column != "label"]
    for column in expected:
        if column not in frame:
            frame[column] = pd.NA
    frame = frame[expected]
    predictions = model.predict(frame)
    # Outlier detectors follow sklearn's convention: -1=anomaly, +1=inlier.
    anomaly_model = Path(model_path).stem in {"isolation_forest", "one_class_svm"}
    labels = [
        "Malicious Traffic" if (int(value) == -1 if anomaly_model else int(value) == 1)
        else "Normal Traffic"
        for value in predictions
    ]
    return labels[0] if isinstance(input_data, dict) else labels
