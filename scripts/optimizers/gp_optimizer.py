from typing import Dict

import numpy as np
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RationalQuadratic, Matern, RBF

from ACTC.scripts.optimizers.simple_pred_optimizer import PredictionOptimizer


class GaussianProcessOptimizer(PredictionOptimizer):
    def __init__(
            self,
            all_scores,
            partial_labels: Dict = None,
            thres_gap: float = 0.01,
            iteration_limit: int = 2,
            metric: str = "accuracy",
            selection_mechanism: str = None,
            **kwargs
    ):
        super().__init__(
            all_scores, partial_labels, thres_gap, iteration_limit, metric, selection_mechanism=selection_mechanism,
            **kwargs
        )
        self.kernel = kwargs["kernel"]

    def estimate_predictor(self) -> None:
        scores, labels = self.get_partial_scores_labels()

        if self.kernel == "Matern":
            kernel = Matern(length_scale=0.1)  # F1@10: 70.2%
        elif self.kernel == "RationalQuadratic":
            kernel = RationalQuadratic(length_scale=0.1)  # F1@10: 72.2%
        elif self.kernel == "RBF":
            kernel = 10 * RBF(10)  # Regression: 10 * RBF(10) # 69 # Classifier: kernel = 0.1 * RBF(0.1) # 69%
        else:
            raise NotImplementedError()

        self.predictor = GaussianProcessClassifier(kernel, optimizer=None)

        train_x = np.array(scores).reshape(-1, 1)
        train_y = np.array(labels)
        self.predictor.fit(train_x, train_y)

    def predict(self, score) -> float:
        return float(self.predictor.predict_proba([[score]])[0, 1])
