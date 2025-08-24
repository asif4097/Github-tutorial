# AI Signal from Live OHLC

Quickstart (NumPy-based, Python 3.13 compatible):

1. Create venv and install deps:
```
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

2. Train the model (downloads OHLC via CCXT):
```
python -m src.simple_model.train
```

3. Run live inference (uses latest closed candles):
```
python -m src.simple_model.infer
```

Artifacts are saved under `/workspace/artifacts`.
