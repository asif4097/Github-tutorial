from __future__ import annotations

import time
from typing import Dict, List, Optional

import ccxt  # type: ignore
import numpy as np

from src.utils.logging import get_logger

logger = get_logger(__name__)


def get_exchange(exchange_name: str) -> ccxt.Exchange:
    exchange_class = getattr(ccxt, exchange_name)
    exchange = exchange_class({"enableRateLimit": True})
    return exchange


def fetch_ohlcv_batch(
    exchange: ccxt.Exchange,
    symbol: str,
    timeframe: str,
    since: Optional[int] = None,
    limit: int = 1000,
) -> List[List[float]]:
    return exchange.fetch_ohlcv(symbol=symbol, timeframe=timeframe, since=since, limit=limit)


def fetch_ohlcv_history_np(
    exchange_name: str,
    symbol: str,
    timeframe: str,
    limit_bars: int,
) -> Dict[str, np.ndarray]:
    exchange = get_exchange(exchange_name)
    all_rows: List[List[float]] = []
    since: Optional[int] = None

    while len(all_rows) < limit_bars:
        remaining = limit_bars - len(all_rows)
        batch_limit = min(1000, remaining)
        rows = fetch_ohlcv_batch(exchange, symbol, timeframe, since=since, limit=batch_limit)
        if not rows:
            break
        all_rows.extend(rows)
        since = int(rows[-1][0]) + 1
        time.sleep(exchange.rateLimit / 1000.0)

    if not all_rows:
        raise RuntimeError("No OHLCV data returned")

    data = np.array(all_rows, dtype=float)
    timestamps = data[:, 0]
    opens = data[:, 1]
    highs = data[:, 2]
    lows = data[:, 3]
    closes = data[:, 4]
    volumes = data[:, 5]

    # sort by timestamp and dedup
    order = np.argsort(timestamps)
    timestamps = timestamps[order]
    opens = opens[order]
    highs = highs[order]
    lows = lows[order]
    closes = closes[order]
    volumes = volumes[order]

    return {
        "timestamp": timestamps,
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volumes,
    }


def fetch_latest_window_np(
    exchange_name: str,
    symbol: str,
    timeframe: str,
    window: int = 600,
) -> Dict[str, np.ndarray]:
    exchange = get_exchange(exchange_name)
    rows = exchange.fetch_ohlcv(symbol=symbol, timeframe=timeframe, limit=window + 1)
    if not rows:
        raise RuntimeError("No OHLCV data returned")
    data = np.array(rows[-window:], dtype=float)
    return {
        "timestamp": data[:, 0],
        "open": data[:, 1],
        "high": data[:, 2],
        "low": data[:, 3],
        "close": data[:, 4],
        "volume": data[:, 5],
    }