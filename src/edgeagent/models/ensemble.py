from __future__ import annotations

import numpy as np


class WeightedEnsemble:
    """Weights are set from out-of-sample log loss (softmax of -k * mean log loss). The Brainbox refreshes them."""
    def __init__(self, names, floor: float = 0.03):
        self.names, self.floor = list(names), floor
        self.w = np.ones(len(self.names)) / len(self.names)
    def set_weights(self, w):
        w = np.maximum(np.asarray(w, float), self.floor)
        self.w = w / w.sum()
    def weights(self) -> dict:
        return dict(zip(self.names, map(float, self.w)))
    def combine(self, member_probs: dict) -> np.ndarray:
        P = sum(wi * np.asarray(member_probs[n]) for wi, n in zip(self.w, self.names))
        return P / P.sum(-1, keepdims=True)
    @staticmethod
    def softmax_weights(mean_ll: dict, names, k: float = 12.0) -> np.ndarray:
        v = np.array([mean_ll[n] for n in names])
        w = np.exp(-k * (v - v.min()))
        return w / w.sum()
