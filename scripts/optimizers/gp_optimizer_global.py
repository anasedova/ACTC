from typing import Dict

import numpy as np
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RationalQuadratic

from ACTC.scripts.optimizers.simple_pred_optimizer import UniPredictionOptimizer


class UniGaussianProcessOptimizer(UniPredictionOptimizer):
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
        self.gp = None

    def estimate_predictor(self) -> None:
        scores, labels = self.get_partial_scores_labels()

        kernel = RationalQuadratic(length_scale=0.1)  # F1@10: 72.2%
        self.predictor = GaussianProcessClassifier(kernel, optimizer=None)

        train_x = np.array(scores).reshape(-1, 1)
        train_y = np.array(labels)
        self.predictor.fit(train_x, train_y)

    def predict(self, score) -> float:
        return float(self.predictor.predict_proba([[score]])[0, 1])
