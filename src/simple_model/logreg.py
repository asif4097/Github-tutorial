from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class LogisticRegressionNP:
    learning_rate: float = 0.05
    num_epochs: int = 800
    l2: float = 0.0
    fit_intercept: bool = True
    seed: int = 42

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)
        self.weights: np.ndarray | None = None

    def _add_bias(self, X: np.ndarray) -> np.ndarray:
        if not self.fit_intercept:
            return X
        ones = np.ones((X.shape[0], 1), dtype=X.dtype)
        return np.hstack([ones, X])

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegressionNP":
        Xb = self._add_bias(X)
        n_features = Xb.shape[1]
        self.weights = self.rng.normal(scale=0.01, size=n_features)

        for _ in range(self.num_epochs):
            logits = Xb @ self.weights
            probs = 1.0 / (1.0 + np.exp(-logits))
            error = probs - y
            grad = (Xb.T @ error) / Xb.shape[0]
            if self.l2 > 0.0:
                grad += self.l2 * np.r_[0.0, self.weights[1:]] if self.fit_intercept else self.l2 * self.weights
            self.weights -= self.learning_rate * grad
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.weights is None:
            raise RuntimeError("Model is not fitted")
        Xb = self._add_bias(X)
        logits = Xb @ self.weights
        probs = 1.0 / (1.0 + np.exp(-logits))
        return np.column_stack([1.0 - probs, probs])


def standardize_train(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.nanmean(X, axis=0)
    std = np.nanstd(X, axis=0)
    std[std == 0] = 1.0
    Xs = (X - mean) / std
    return Xs, mean, std


def standardize_apply(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (X - mean) / std


def auc_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    order = np.argsort(-y_score)
    y_sorted = y_true[order]
    cum_pos = np.cumsum(y_sorted)
    cum_neg = np.cumsum(1 - y_sorted)
    total_pos = cum_pos[-1]
    total_neg = cum_neg[-1]
    if total_pos == 0 or total_neg == 0:
        return float("nan")
    # Rank-sum AUC approximation
    rank_sum = np.sum(cum_pos[y_sorted == 0])
    auc = 1.0 - rank_sum / (total_pos * total_neg)
    return float(auc)