"""Sparse autoencoders: read a stream, encode it into features.

    f = relu((x - b_dec) @ W_enc + b_enc)          x_hat = f @ W_dec + b_dec

Weights are numpy or torch arrays with W_enc (d_model, n_features) and W_dec (n_features, d_model).
"""

import numpy as np


class SAE:
    def __init__(self, W_enc, b_enc, W_dec, b_dec):
        self.W_enc, self.b_enc = np.asarray(W_enc, dtype=np.float32), np.asarray(b_enc, dtype=np.float32)
        self.W_dec, self.b_dec = np.asarray(W_dec, dtype=np.float32), np.asarray(b_dec, dtype=np.float32)
        if self.W_enc.shape != self.W_dec.T.shape:
            raise ValueError("W_enc is (d_model, n_features) and W_dec is (n_features, d_model)")

    @property
    def n_features(self) -> int:
        return self.W_enc.shape[1]

    def encode(self, x) -> np.ndarray:
        return np.maximum((np.asarray(x, dtype=np.float32) - self.b_dec) @ self.W_enc + self.b_enc, 0.0)

    def decode(self, f) -> np.ndarray:
        return np.asarray(f, dtype=np.float32) @ self.W_dec + self.b_dec
