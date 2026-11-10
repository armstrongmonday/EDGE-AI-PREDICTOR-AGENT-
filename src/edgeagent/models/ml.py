from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ..features.state import FEATURES
from .base import Model, outcome


class MLModel(Model):
    def __init__(self, name, est):
        self.name, self.est = name, est
    def fit(self, matches, X):
        self.est.fit(X[FEATURES].to_numpy(), outcome(matches))
        return self
    def predict_many(self, matches, X):
        P = self.est.predict_proba(X[FEATURES].to_numpy())
        out = np.full((len(P), 3), 1e-3)
        out[:, list(self.est.classes_)] = P
        return out / out.sum(1, keepdims=True)

def logistic(C=1.0):
    return MLModel("logistic", make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=500)))

def gbm(depth=3):
    return MLModel("gbm", HistGradientBoostingClassifier(max_depth=depth, learning_rate=0.05, max_iter=120,
                                                         l2_regularization=1.0, early_stopping=False, random_state=0))
