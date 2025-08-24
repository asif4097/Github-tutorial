from __future__ import annotations

import joblib
import numpy as np

from src.config import DEFAULT_SETTINGS
from src.simple_model.data_ccxt_np import fetch_latest_window_np
from src.simple_model.features_np import make_features, build_matrix
from src.simple_model.logreg import standardize_apply
from src.utils.logging import get_logger

logger = get_logger(__name__)


class Signal:
    def __init__(self, prob_up: float, threshold: float = 0.5) -> None:
        self.prob_up = float(prob_up)
        self.threshold = float(threshold)

    @property
    def action(self) -> str:
        return "BUY" if self.prob_up >= self.threshold else "SELL"

    def to_dict(self) -> dict:
        return {"prob_up": self.prob_up, "action": self.action, "threshold": self.threshold}


def load_artifacts():
    data = joblib.load(DEFAULT_SETTINGS.artifacts_dir / "simple_model.joblib")
    return data


def predict_signal() -> Signal:
    art = load_artifacts()
    raw = fetch_latest_window_np(
        exchange_name=DEFAULT_SETTINGS.exchange_name,
        symbol=DEFAULT_SETTINGS.symbol,
        timeframe=DEFAULT_SETTINGS.timeframe,
        window=600,
    )
    feats = make_features(raw)
    X, mask = build_matrix(feats)
    # latest row
    x_row = X[mask][-1:]
    x_std = standardize_apply(x_row, art["mean"], art["std"])

    # predict proba with logistic regression weights
    w = art["weights"]
    fit_intercept = art["fit_intercept"]
    if fit_intercept:
        x_row_b = np.hstack([np.ones((1, 1)), x_std])
    else:
        x_row_b = x_std
    logit = float(x_row_b @ w)
    prob = 1.0 / (1.0 + np.exp(-logit))
    logger.info(f"prob_up={prob:.3f}")
    return Signal(prob_up=prob, threshold=0.5)


if __name__ == "__main__":
    sig = predict_signal()
    logger.info(sig.to_dict())