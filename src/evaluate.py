from __future__ import annotations

from pathlib import Path
import json
import joblib
import pandas as pd

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score, accuracy_score


def _clean_target(y: pd.Series) -> pd.Series:
    y = y.astype(str).str.strip()
    y = y.str.replace(".", "", regex=False)
    return y


def main() -> None:
    X_test = pd.read_parquet("data/processed/X_test.parquet")
    y_test = pd.read_parquet("data/processed/y_test.parquet").squeeze()
    y_test = _clean_target(y_test)

    preprocessor = joblib.load("artifacts/preprocessor.joblib")
    model = joblib.load("models/model.pkl")

    X_test_t = preprocessor.transform(X_test)
    preds = model.predict(X_test_t)

    metrics = {
        "test_f1_macro": float(f1_score(y_test, preds, average="macro")),
        "test_accuracy": float(accuracy_score(y_test, preds)),
    }

    Path("artifacts").mkdir(exist_ok=True)
    with open("artifacts/eval_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    cm = confusion_matrix(y_test, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)

    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax, values_format="d")
    plt.tight_layout()
    fig.savefig("artifacts/confusion_matrix.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()