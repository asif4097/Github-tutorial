from __future__ import annotations

import numpy as np
from typing import Dict, Tuple


def sliding_window_view(a: np.ndarray, window: int) -> np.ndarray:
    from numpy.lib.stride_tricks import sliding_window_view as swv
    return swv(a, window_shape=window)


def compute_ema(values: np.ndarray, span: int) -> np.ndarray:
    alpha = 2.0 / (span + 1.0)
    ema = np.empty_like(values)
    ema[:] = np.nan
    if values.size == 0:
        return ema
    ema[0] = values[0]
    for i in range(1, values.size):
        ema[i] = alpha * values[i] + (1 - alpha) * ema[i - 1]
    return ema


def compute_rsi(close: np.ndarray, window: int = 14) -> np.ndarray:
    delta = np.diff(close, prepend=close[0])
    gain = np.maximum(delta, 0.0)
    loss = np.maximum(-delta, 0.0)

    avg_gain = np.empty_like(close)
    avg_loss = np.empty_like(close)
    avg_gain[:] = np.nan
    avg_loss[:] = np.nan

    # Wilder's smoothing
    avg_gain[window] = np.mean(gain[1:window + 1])
    avg_loss[window] = np.mean(loss[1:window + 1])
    for i in range(window + 1, close.size):
        avg_gain[i] = (avg_gain[i - 1] * (window - 1) + gain[i]) / window
        avg_loss[i] = (avg_loss[i - 1] * (window - 1) + loss[i]) / window

    rs = avg_gain / np.where(avg_loss == 0, np.nan, avg_loss)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


def make_features(ohlcv: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    close = ohlcv["close"]
    high = ohlcv["high"]
    low = ohlcv["low"]

    ret_1 = np.concatenate([[np.nan], np.diff(close) / close[:-1]])
    ret_3 = np.concatenate([np.full(3, np.nan), (close[3:] / close[:-3]) - 1.0])
    ret_5 = np.concatenate([np.full(5, np.nan), (close[5:] / close[:-5]) - 1.0])

    # rolling std of ret_1
    from numpy.lib.stride_tricks import sliding_window_view as swv
    if ret_1.size >= 10:
        vol_10 = np.concatenate([
            np.full(9, np.nan),
            np.nanstd(swv(ret_1, 10), axis=1, ddof=1)
        ])
    else:
        vol_10 = np.full_like(ret_1, np.nan)

    ema_9 = compute_ema(close, 9)
    ema_21 = compute_ema(close, 21)

    rsi_14 = compute_rsi(close, 14)

    return {
        "ret_1": ret_1,
        "ret_3": ret_3,
        "ret_5": ret_5,
        "vol_10": vol_10,
        "ema_9": ema_9,
        "ema_21": ema_21,
        "rsi_14": rsi_14,
    }


def make_labels(ohlcv: Dict[str, np.ndarray], lookahead_bars: int, threshold: float) -> np.ndarray:
    close = ohlcv["close"]
    future = np.concatenate([close[lookahead_bars:], np.full(lookahead_bars, np.nan)])
    future_ret = future / close - 1.0
    y = (future_ret > threshold).astype(float)
    return y


def build_matrix(features: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    keys = sorted(features.keys())
    X = np.column_stack([features[k] for k in keys])
    finite_mask = np.isfinite(X).all(axis=1)
    return X, finite_mask