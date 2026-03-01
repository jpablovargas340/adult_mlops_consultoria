from __future__ import annotations

from pathlib import Path
import json
from ucimlrepo import fetch_ucirepo


def ingest_adult(output_dir: str = "data/raw") -> dict:
    adult = fetch_ucirepo(id=2)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    X = adult.data.features
    y = adult.data.targets

    X.to_parquet(out / "features.parquet", index=False)
    y.to_parquet(out / "targets.parquet", index=False)

    vc = y.squeeze().value_counts()
    summary = {
        "n_rows": int(len(X)),
        "n_features": int(X.shape[1]),
        "target_distribution": {str(k): int(v) for k, v in vc.items()},
    }
    return summary


def main() -> None:
    summary = ingest_adult()
    Path("artifacts").mkdir(exist_ok=True)
    with open("artifacts/ingest_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()