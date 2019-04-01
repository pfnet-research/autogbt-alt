import numpy as np


class AveragingLGBMClassifier:

    def __init__(self, models):
        self.models = models

    def predict(self, X):
        y = np.zeros((len(X)))
        for model in self.models:
            y += model.predict(X) / len(self.models)
        return y
