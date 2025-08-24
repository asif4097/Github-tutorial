from __future__ import annotations

from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, roc_auc_score

from src.config import DEFAULT_SETTINGS
from src.data.fetch_ccxt import fetch_ohlcv_history
from src.features.indicators import add_technical_indicators, make_labels
from src.utils.logging import get_logger

logger = get_logger(__name__)


def make_dataset(settings=DEFAULT_SETTINGS) -> Tuple[pd.DataFrame, pd.Series]:
    raw = fetch_ohlcv_history(
        exchange_name=settings.exchange_name,
        symbol=settings.symbol,
        timeframe=settings.timeframe,
        limit_bars=settings.num_bars_history,
    )
    feats = add_technical_indicators(raw)
    y = make_labels(raw, lookahead_bars=settings.lookahead_bars, threshold=settings.label_threshold)
    dataset = feats.dropna().copy()
    y_aligned = y.loc[dataset.index]
    # drop last lookahead rows where label becomes NaN post-shift
    y_aligned = y_aligned.dropna()
    dataset = dataset.loc[y_aligned.index]
    return dataset, y_aligned.astype(int)


def train_and_evaluate(X: pd.DataFrame, y: pd.Series) -> XGBClassifier:
    tscv = TimeSeriesSplit(n_splits=5)
    aucs = []
    model = XGBClassifier(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        eval_metric="logloss",
    )

    last_train_idx = None
    for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, proba)
        aucs.append(auc)
        logger.info(f"Fold {fold_idx + 1} AUC: {auc:.4f}")
        last_train_idx = train_idx

    logger.info(f"CV AUC mean: {np.mean(aucs):.4f} +/- {np.std(aucs):.4f}")
    # Refit on all data
    model.fit(X, y)
    return model


def save_artifacts(model: XGBClassifier, feature_columns, artifacts_dir: Path) -> None:
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, artifacts_dir / "model.joblib")
    joblib.dump(feature_columns, artifacts_dir / "features.joblib")


def main() -> None:
    logger.info("Preparing dataset ...")
    X, y = make_dataset()
    logger.info(f"Dataset: X={X.shape}, positives={int(y.sum())}, baseline={y.mean():.3f}")

    logger.info("Training model ...")
    model = train_and_evaluate(X, y)

    save_artifacts(model, X.columns.tolist(), DEFAULT_SETTINGS.artifacts_dir)
    logger.info(f"Artifacts saved to: {DEFAULT_SETTINGS.artifacts_dir}")


if __name__ == "__main__":
    main()