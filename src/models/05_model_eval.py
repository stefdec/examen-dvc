import pandas as pd
import numpy as np
from joblib import load
import json
from pathlib import Path

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

X_test = pd.read_csv("data/processed_data/X_test_scaled.csv")
y_test = pd.read_csv("data/processed_data/y_test.csv")
y_test = np.ravel(y_test)


def main(repo_path):
    model = load(repo_path / "models/lgbm_model.pkl")
    predictions = model.predict(X_test)

    metrics = {
        "mae": float(mean_absolute_error(y_test, predictions)),
        "mse": float(mean_squared_error(y_test, predictions)),
        "rmse": float(mean_squared_error(y_test, predictions) ** 0.5),
        "r2": float(r2_score(y_test, predictions)),
    }
    metrics_path = repo_path / "metrics/metrics.json"
    metrics_path.write_text(json.dumps(metrics))


if __name__ == "__main__":
    repo_path = Path(__file__).parent.parent.parent
    main(repo_path)
