import numpy as np

from sklearn.linear_model import LogisticRegression


class SigmoidCalibrator:

    def __init__(self):
        self.model = LogisticRegression(
            C=1.0,
            solver="lbfgs",
            max_iter=2000,
            random_state=42,
        )

    def fit(self, raw_scores, y):
        raw_scores = np.asarray(raw_scores).reshape(-1, 1)
        y = np.asarray(y)

        self.model.fit(raw_scores, y)

        return self

    def predict_proba(self, raw_scores):
        raw_scores = np.asarray(raw_scores).reshape(-1, 1)

        return self.model.predict_proba(raw_scores)[:, 1]