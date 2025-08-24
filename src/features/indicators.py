import pandas as pd
import numpy as np
import ta


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    features = df.copy()

    # Price-based returns
    features["ret_1"] = features["close"].pct_change(1)
    features["ret_3"] = features["close"].pct_change(3)
    features["ret_5"] = features["close"].pct_change(5)
    features["ret_10"] = features["close"].pct_change(10)

    # Rolling volatility
    features["vol_10"] = features["ret_1"].rolling(10).std()
    features["vol_20"] = features["ret_1"].rolling(20).std()

    # RSI
    rsi = ta.momentum.RSIIndicator(close=features["close"], window=14)
    features["rsi_14"] = rsi.rsi()

    # MACD
    macd = ta.trend.MACD(close=features["close"], window_slow=26, window_fast=12, window_sign=9)
    features["macd"] = macd.macd()
    features["macd_signal"] = macd.macd_signal()
    features["macd_hist"] = macd.macd_diff()

    # Stochastic Oscillator
    stoch = ta.momentum.StochasticOscillator(high=features["high"], low=features["low"], close=features["close"], window=14, smooth_window=3)
    features["stoch_k"] = stoch.stoch()
    features["stoch_d"] = stoch.stoch_signal()

    # ATR
    atr = ta.volatility.AverageTrueRange(high=features["high"], low=features["low"], close=features["close"], window=14)
    features["atr_14"] = atr.average_true_range()

    # EMAs
    features["ema_9"] = features["close"].ewm(span=9, adjust=False).mean()
    features["ema_21"] = features["close"].ewm(span=21, adjust=False).mean()
    features["ema_50"] = features["close"].ewm(span=50, adjust=False).mean()

    # Bollinger Bands percB
    bb = ta.volatility.BollingerBands(close=features["close"], window=20, window_dev=2)
    features["bb_perc_b"] = bb.bollinger_pband()

    return features


def make_labels(df: pd.DataFrame, lookahead_bars: int, threshold: float) -> pd.Series:
    future_return = df["close"].shift(-lookahead_bars) / df["close"] - 1.0
    label = (future_return > threshold).astype(int)
    return label