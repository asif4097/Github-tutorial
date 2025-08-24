from dataclasses import dataclass
from pathlib import Path


@dataclass
class ModelSettings:
    exchange_name: str = "bitstamp"
    symbol: str = "BTC/USD"
    timeframe: str = "1m"
    num_bars_history: int = 5000
    lookahead_bars: int = 3
    label_threshold: float = 0.001  # 0.1% forward return
    artifacts_dir: Path = Path("/workspace/artifacts")


DEFAULT_SETTINGS = ModelSettings()