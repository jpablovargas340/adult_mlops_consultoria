from __future__ import annotations

from pathlib import Path
import json
import joblib
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, classification_report


def _clean_target(y: pd.Series) -> pd.Series:
    # Adult a veces trae espacios y/o punto final (<=50K.)
    y = y.astype(str).str.strip()
    y = y.str.replace(".", "", regex=False)  # quita punto final si existe
    return y


def main(random_state: int = 42) -> None:
    X_train = pd.read_parquet("data/processed/X_train.parquet")
    X_test = pd.read_parquet("data/processed/X_test.parquet")
    y_train = pd.read_parquet("data/processed/y_train.parquet").squeeze()
    y_test = pd.read_parquet("data/processed/y_test.parquet").squeeze()

    y_train = _clean_target(y_train)
    y_test = _clean_target(y_test)

    # Sanity check: asegurar 2 clases
    classes = sorted(y_train.unique().tolist())
    if len(classes) != 2:
        raise ValueError(f"Target tiene {len(classes)} clases: {classes}")

    preprocessor = joblib.load("artifacts/preprocessor.joblib")
    X_train_t = preprocessor.transform(X_train)
    X_test_t = preprocessor.transform(X_test)

    # Solver que funciona bien para binario y sparse (OneHot)
    model = LogisticRegression(
        solver="saga",
        max_iter=3000,
        random_state=random_state,
    )

    model.fit(X_train_t, y_train)
    preds = model.predict(X_test_t)

    f1 = f1_score(y_test, preds, average="macro")
    acc = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds, output_dict=True)

    metrics = {"f1_macro": float(f1), "accuracy": float(acc)}

    Path("models").mkdir(exist_ok=True)
    joblib.dump(model, "models/model.pkl")

    Path("artifacts").mkdir(exist_ok=True)
    with open("artifacts/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    with open("artifacts/classification_report.json", "w") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()