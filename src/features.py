from __future__ import annotations

from pathlib import Path
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer


TARGET_COL = "income"


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    num_cols = X.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, num_cols),
            ("cat", categorical_pipe, cat_cols),
        ],
        remainder="drop",
    )
    return preprocessor


def main(test_size: float = 0.2, random_state: int = 42) -> None:
    X = pd.read_parquet("data/raw/features.parquet")
    y = pd.read_parquet("data/raw/targets.parquet")

    # y es DF de 1 col; normalizamos nombre de target a "income"
    y_series = y.squeeze()
    y_series.name = TARGET_COL

    # Limpieza típica del Adult: strings con espacios y '?' como missing
    for c in X.columns:
        if X[c].dtype == "object":
            X[c] = (
                X[c].astype(str)
                .str.strip()
                .replace("?", pd.NA)
            )

    # Split reproducible
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_series, test_size=test_size, random_state=random_state, stratify=y_series
    )

    preprocessor = build_preprocessor(X_train)
    preprocessor.fit(X_train)

    Path("artifacts").mkdir(exist_ok=True)
    joblib.dump(preprocessor, "artifacts/preprocessor.joblib")

    # Guardar datasets procesados "crudos" (antes de transformar) para trazabilidad
    # y el train script puede aplicar el preprocessor al vuelo.
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    X_train.to_parquet("data/processed/X_train.parquet", index=False)
    X_test.to_parquet("data/processed/X_test.parquet", index=False)
    y_train.to_frame().to_parquet("data/processed/y_train.parquet", index=False)
    y_test.to_frame().to_parquet("data/processed/y_test.parquet", index=False)


if __name__ == "__main__":
    main()