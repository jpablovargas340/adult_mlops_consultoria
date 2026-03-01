from __future__ import annotations

from pathlib import Path
import json
import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score, classification_report


def main(random_state: int = 42) -> None:
    # Cargar datos
    X_train = pd.read_parquet("data/processed/X_train.parquet")
    X_test = pd.read_parquet("data/processed/X_test.parquet")
    y_train = pd.read_parquet("data/processed/y_train.parquet").squeeze()
    y_test = pd.read_parquet("data/processed/y_test.parquet").squeeze()

    # Cargar preprocesador
    preprocessor = joblib.load("artifacts/preprocessor.joblib")

    # Transformar
    X_train_trans = preprocessor.transform(X_train)
    X_test_trans = preprocessor.transform(X_test)

    # Modelo
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=random_state,
        n_jobs=-1,
    )

    model.fit(X_train_trans, y_train)

    # Evaluación
    preds = model.predict(X_test_trans)

    f1 = f1_score(y_test, preds, average="macro")
    acc = accuracy_score(y_test, preds)

    report = classification_report(y_test, preds, output_dict=True)

    metrics = {
        "f1_macro": float(f1),
        "accuracy": float(acc),
    }

    # Guardar modelo
    Path("models").mkdir(exist_ok=True)
    joblib.dump(model, "models/model.pkl")

    # Guardar métricas
    Path("artifacts").mkdir(exist_ok=True)
    with open("artifacts/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    with open("artifacts/classification_report.json", "w") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()