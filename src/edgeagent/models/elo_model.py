from __future__ import annotations

import numpy as np

from ..features.state import HOME_ADV
from .base import Model, outcome


class EloModel(Model):
    name = "elo"
    def __init__(self):
        self.draw0 = 0.27
    def fit(self, matches, X):
        self.draw0 = float(np.clip((outcome(matches) == 1).mean(), 0.15, 0.35))
        return self
    def predict_many(self, matches, X):
        d = X["elo_diff"].to_numpy() + HOME_ADV
        e = 1 / (1 + 10 ** (-d / 400))
        draw = self.draw0 * np.exp(-(d / 450) ** 2)
        P = np.column_stack([e - draw / 2, draw, 1 - e - draw / 2]).clip(0.02)
        return P / P.sum(1, keepdims=True)
