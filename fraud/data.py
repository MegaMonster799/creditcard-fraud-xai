"""Load, audit, and leak-free transform of the ULB credit-card data."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from fraud.config import (
    AMOUNT_WINSOR_HI,
    AMOUNT_WINSOR_LO,
    SEED,
    TEST_SIZE,
)
from fraud import DATA_PATH


@dataclass
class DataAudit:
    n_rows: int
    n_features: int
    n_fraud: int
    n_legit: int
    n_missing: int
    n_duplicate_rows: int
    n_dropped_duplicates: int
    amount_iqr_outlier_rate: float
    fraud_rate: float


class AmountWinsorizer(BaseEstimator, TransformerMixin):
    """Clip Amount using training-set percentiles only (no test leakage)."""

    def __init__(self, lower: float = AMOUNT_WINSOR_LO, upper: float = AMOUNT_WINSOR_HI):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        values = np.asarray(X["Amount"], dtype=float)
        self.lo_, self.hi_ = np.quantile(values, [self.lower, self.upper])
        return self

    def transform(self, X):
        X = X.copy()
        X["Amount"] = np.clip(np.asarray(X["Amount"], dtype=float), self.lo_, self.hi_)
        return X


class TimeAmountScaler(BaseEstimator, TransformerMixin):
    """Standardize Time and Amount in place so SHAP keeps original column names."""

    def fit(self, X, y=None):
        self.scaler_ = StandardScaler().fit(X[["Time", "Amount"]])
        return self

    def transform(self, X):
        X = X.copy()
        X[["Time", "Amount"]] = self.scaler_.transform(X[["Time", "Amount"]])
        return X


def _iqr_outlier_rate(series: pd.Series) -> float:
    q1, q3 = series.quantile([0.25, 0.75])
    iqr = q3 - q1
    if iqr == 0:
        return 0.0
    mask = (series < q1 - 1.5 * iqr) | (series > q3 + 1.5 * iqr)
    return float(mask.mean())


def load_and_split(path=DATA_PATH, seed: int = SEED) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, DataAudit]:
    df = pd.read_csv(path)
    n_missing = int(df.isna().sum().sum())
    n_dup = int(df.duplicated().sum())
    fraud_before = int((df["Class"] == 1).sum())
    df = df.drop_duplicates().reset_index(drop=True)
    dropped = n_dup
    y = df["Class"].astype(int)
    X = df.drop(columns=["Class"])
    audit = DataAudit(
        n_rows=int(len(df) + dropped),
        n_features=int(X.shape[1]),
        n_fraud=int(y.sum()),
        n_legit=int((y == 0).sum()),
        n_missing=n_missing,
        n_duplicate_rows=n_dup,
        n_dropped_duplicates=dropped,
        amount_iqr_outlier_rate=_iqr_outlier_rate(df["Amount"]),
        fraud_rate=float(y.mean()),
    )
    # Split before any scaling / clipping — the previous script leaked holdout moments.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=seed, stratify=y
    )
    return X_train, X_test, y_train, y_test, audit
