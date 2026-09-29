"""Linear probes. Each is fit on one checkpoint and layer and scores held-out activations."""

from typing import Protocol

import numpy as np


class Probe(Protocol):
    name: str
    version: str

    def fit(self, X: np.ndarray, y: np.ndarray) -> "Probe": ...
    def score(self, X: np.ndarray) -> np.ndarray: ...


class DiffMeans:
    """Direction from the negative mean to the positive mean; the score is the projection."""
    name = "diff_means"
    version = "0"

    def fit(self, X, y):
        y = np.asarray(y, dtype=bool)
        self.direction = X[y].mean(0) - X[~y].mean(0)
        return self

    def score(self, X):
        return X @ self.direction


class Logistic:
    """L2-regularised logistic regression on standardised features, fit by Newton's method.
    Each step solves a (d+1)x(d+1) system: fine up to a few thousand features on CPU."""
    name = "logistic"
    version = "0"

    def __init__(self, l2: float = 1.0, iters: int = 50, tol: float = 1e-6):
        self.l2, self.iters, self.tol = l2, iters, tol

    def fit(self, X, y):
        y = np.asarray(y, dtype=float)
        self.mean, self.std = X.mean(0), X.std(0) + 1e-8
        Z = np.hstack([(X - self.mean) / self.std, np.ones((len(X), 1))])
        w = np.zeros(Z.shape[1])
        reg = np.full(Z.shape[1], self.l2)
        reg[-1] = 0.0                                   # no penalty on the bias
        for _ in range(self.iters):
            p = 1 / (1 + np.exp(-np.clip(Z @ w, -30, 30)))
            grad = Z.T @ (p - y) + reg * w
            hess = (Z * (p * (1 - p))[:, None]).T @ Z + np.diag(reg)
            step = np.linalg.solve(hess, grad)
            w -= step
            if np.abs(step).max() < self.tol:
                break
        self.w = w
        return self

    def score(self, X):
        return ((X - self.mean) / self.std) @ self.w[:-1] + self.w[-1]
