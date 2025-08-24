from __future__ import annotations

import joblib
import numpy as np
import pandas as pd

from src.config import DEFAULT_SETTINGS
from src.data.fetch_ccxt import fetch_latest_closed_window
from src.features.indicators import add_technical_indicators
from src.utils.logging import get_logger

logger = get_logger(__name__)


class Signal:
    def __init__(self, prob_up: float, threshold: float = 0.5) -> None:
        self.prob_up = float(prob_up)
        self.threshold = float(threshold)

    @property
    def action(self) -> str:
        if self.prob_up >= self.threshold:
            return "BUY"
        return "SELL"

    def to_dict(self) -> dict:
        return {"prob_up": self.prob_up, "action": self.action, "threshold": self.threshold}


def load_artifacts(artifacts_dir=DEFAULT_SETTINGS.artifacts_dir):
    model = joblib.load(artifacts_dir / "model.joblib")
    feature_columns = joblib.load(artifacts_dir / "features.joblib")
    return model, feature_columns


def build_feature_row(raw_window: pd.DataFrame, feature_columns) -> pd.DataFrame:
    feats = add_technical_indicators(raw_window).dropna()
    if feats.empty:
        raise RuntimeError("No features generated for the latest window")
    latest_row = feats.iloc[-1:]
    # Align columns
    for col in feature_columns:
        if col not in latest_row.columns:
            latest_row[col] = np.nan
    latest_row = latest_row[feature_columns]
    latest_row = latest_row.fillna(method="ffill").fillna(method="bfill")
    return latest_row


def predict_signal() -> Signal:
    model, feature_columns = load_artifacts()
    raw_window = fetch_latest_closed_window(
        exchange_name=DEFAULT_SETTINGS.exchange_name,
        symbol=DEFAULT_SETTINGS.symbol,
        timeframe=DEFAULT_SETTINGS.timeframe,
        window=600,
    )
    row = build_feature_row(raw_window, feature_columns)
    prob_up = float(model.predict_proba(row)[:, 1][0])
    logger.info(f"prob_up={prob_up:.3f}")
    return Signal(prob_up=prob_up, threshold=0.5)


if __name__ == "__main__":
    signal = predict_signal()
    logger.info(signal.to_dict())