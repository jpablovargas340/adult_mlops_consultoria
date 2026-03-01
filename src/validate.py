from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


EXPECTED_COLS = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education-num",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
    "native-country",
]


def validate() -> dict:
    X = pd.read_parquet("data/raw/features.parquet")
    y = pd.read_parquet("data/raw/targets.parquet")

    # --- checks de estructura ---
    cols_ok = list(X.columns) == EXPECTED_COLS
    missing_cols = sorted(list(set(EXPECTED_COLS) - set(X.columns)))
    extra_cols = sorted(list(set(X.columns) - set(EXPECTED_COLS)))

    # --- nulos ---
    null_pct = X.isnull().mean().round(6).to_dict()

    # --- duplicados ---
    dup_rows = int(X.duplicated().sum())

    # --- checks simples numéricos (rangos razonables) ---
    numeric_checks = {
        "age_in_range_17_90": bool(X["age"].between(17, 90).all()),
        "education_num_in_range_1_16": bool(X["education-num"].between(1, 16).all()),
        "hours_per_week_in_range_1_99": bool(X["hours-per-week"].between(1, 99).all()),
    }

    # --- target ---
    # y viene como DataFrame 1 col
    y_series = y.squeeze()
    target_dist = y_series.value_counts().to_dict()

    report = {
        "n_rows": int(len(X)),
        "n_features": int(X.shape[1]),
        "columns": {
            "expected_match_order": cols_ok,
            "missing": missing_cols,
            "extra": extra_cols,
        },
        "null_percentage": {k: float(v) for k, v in null_pct.items()},
        "duplicates": dup_rows,
        "numeric_checks": numeric_checks,
        "target_distribution": {str(k): int(v) for k, v in target_dist.items()},
        "status": "PASS" if (cols_ok and not missing_cols) else "WARN",
    }
    return report


def main() -> None:
    report = validate()
    Path("artifacts").mkdir(exist_ok=True)
    with open("artifacts/validation_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()