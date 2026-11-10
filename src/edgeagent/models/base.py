from __future__ import annotations

import numpy as np
import pandas as pd


def outcome(matches: pd.DataFrame) -> np.ndarray:
    hg, ag = matches["hg"].to_numpy(), matches["ag"].to_numpy()
    return np.where(hg > ag, 0, np.where(hg == ag, 1, 2))

class Model:
    name = "base"
    def fit(self, matches: pd.DataFrame, X: pd.DataFrame):
        return self
    def predict_many(self, matches: pd.DataFrame, X: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError
    def predict(self, home: str, away: str, feats: dict) -> np.ndarray:
        return self.predict_many(pd.DataFrame({"home": [home], "away": [away]}), pd.DataFrame([feats]))[0]
