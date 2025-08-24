from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import json

import joblib
import numpy as np

from src.config import DEFAULT_SETTINGS
from src.simple_model.data_ccxt_np import fetch_ohlcv_history_np
from src.simple_model.features_np import make_features, make_labels, build_matrix
from src.simple_model.logreg import LogisticRegressionNP, standardize_train, auc_score
from src.utils.logging import get_logger

logger = get_logger(__name__)


def make_dataset():
    settings = DEFAULT_SETTINGS
    raw = fetch_ohlcv_history_np(
        exchange_name=settings.exchange_name,
        symbol=settings.symbol,
        timeframe=settings.timeframe,
        limit_bars=settings.num_bars_history,
    )
    feats = make_features(raw)
    X_all, mask = build_matrix(feats)
    y_all = make_labels(raw, lookahead_bars=settings.lookahead_bars, threshold=settings.label_threshold)
    finite_mask = mask & np.isfinite(y_all)

    X_all = X_all[finite_mask]
    y_all = y_all[finite_mask]

    # Train/validation split by time
    split = int(X_all.shape[0] * 0.8)
    X_train, X_val = X_all[:split], X_all[split:]
    y_train, y_val = y_all[:split], y_all[split:]

    X_train_s, mean, std = standardize_train(X_train)
    X_val_s = (X_val - mean) / std

    return (X_train_s, y_train), (X_val_s, y_val), (mean, std), feats.keys()


def main() -> None:
    (X_train, y_train), (X_val, y_val), (mean, std), feature_names = make_dataset()

    model = LogisticRegressionNP(learning_rate=0.1, num_epochs=1200, l2=1e-3)
    model.fit(X_train, y_train)

    val_proba = model.predict_proba(X_val)[:, 1]
    auc = auc_score(y_val, val_proba)
    logger.info(f"Validation AUC: {auc:.4f}")

    # Fit on all data
    X_all = np.vstack([X_train, X_val])
    y_all = np.concatenate([y_train, y_val])
    model.fit(X_all, y_all)

    artifacts = DEFAULT_SETTINGS.artifacts_dir
    artifacts.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "weights": model.weights,
        "fit_intercept": model.fit_intercept,
        "mean": mean,
        "std": std,
        "feature_names": list(feature_names),
    }, artifacts / "simple_model.joblib")
    logger.info(f"Saved artifacts to {artifacts}")


if __name__ == "__main__":
    main()