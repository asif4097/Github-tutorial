from __future__ import annotations

import time
from typing import List, Optional

import ccxt  # type: ignore
import pandas as pd

from src.utils.logging import get_logger

logger = get_logger(__name__)


def get_exchange(exchange_name: str) -> ccxt.Exchange:
    exchange_class = getattr(ccxt, exchange_name)
    exchange = exchange_class({"enableRateLimit": True})
    return exchange


def fetch_ohlcv(
    exchange: ccxt.Exchange,
    symbol: str,
    timeframe: str,
    since: Optional[int] = None,
    limit: int = 1000,
) -> List[List[float]]:
    return exchange.fetch_ohlcv(symbol=symbol, timeframe=timeframe, since=since, limit=limit)


def fetch_ohlcv_history(
    exchange_name: str,
    symbol: str,
    timeframe: str,
    limit_bars: int,
) -> pd.DataFrame:
    """Fetch up to limit_bars closed candles and return as DataFrame with columns [timestamp, open, high, low, close, volume]."""
    exchange = get_exchange(exchange_name)

    all_bars: List[List[float]] = []
    since: Optional[int] = None

    while len(all_bars) < limit_bars:
        remaining = limit_bars - len(all_bars)
        batch_limit = min(1000, remaining)
        ohlcv = fetch_ohlcv(exchange, symbol, timeframe, since=since, limit=batch_limit)
        if not ohlcv:
            break
        all_bars.extend(ohlcv)
        # since is milliseconds; add 1 ms to avoid duplicates
        since = int(ohlcv[-1][0]) + 1
        # Respect rate limits
        time.sleep(exchange.rateLimit / 1000.0)

    df = pd.DataFrame(all_bars, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df.drop_duplicates(subset=["timestamp"], keep="last", inplace=True)
    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)
    # Convert timestamp to datetime
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df.set_index("datetime", inplace=True)
    return df[["open", "high", "low", "close", "volume"]]


def fetch_latest_closed_window(
    exchange_name: str,
    symbol: str,
    timeframe: str,
    window: int = 500,
) -> pd.DataFrame:
    exchange = get_exchange(exchange_name)
    ohlcv = exchange.fetch_ohlcv(symbol=symbol, timeframe=timeframe, limit=window + 1)
    if not ohlcv:
        raise RuntimeError("No OHLCV data returned")
    # Drop the last candle if it's still forming (ccxt returns last closed by default for most exchanges)
    df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df.set_index("datetime", inplace=True)
    return df[["open", "high", "low", "close", "volume"]]