"""Figure 1 EDA with a correct fraud overlay and a data-quality audit."""

from __future__ import annotations

import json

from fraud import OUTPUT_DIR
from fraud.data import load_and_split
from fraud.plots import plot_eda
from fraud import DATA_PATH
import pandas as pd


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(DATA_PATH)
    cleaned = raw.drop_duplicates().reset_index(drop=True)
    X_train, X_test, y_train, y_test, audit = load_and_split()
    print(json.dumps(audit.__dict__, indent=2))
    print(
        f"After duplicate drop: n={len(cleaned)} fraud={int(cleaned['Class'].sum())} "
        f"train={len(X_train)} holdout={len(X_test)} "
        f"train_fraud={int(y_train.sum())} holdout_fraud={int(y_test.sum())}"
    )
    path = plot_eda(cleaned)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
