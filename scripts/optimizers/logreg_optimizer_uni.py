from typing import Dict
from sklearn import linear_model
from ACTC.scripts.optimizers.simple_pred_optimizer import UniPredictionOptimizer


class UniLogisticRegressionOptimizer(UniPredictionOptimizer):
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

    def estimate_predictor(self) -> None:
        """
        Initialize and train a predictor (current implementation: logistic regression) that predicts the labels basing
            on the scores obtained by embedding model.
        Features are the scores: the minimal and maximal scores predicted by embedding model + scores of the selected
            gold entries).
        Labels are the gold labels: 0 class for the minimal score, 1 class for the maximal score + 0/1 known gold label
            for the selected gold entries.
        """

        train_x, train_y = self.get_partial_scores_labels()
        self.predictor = linear_model.LogisticRegression(penalty='l2', C=100.0)
        self.predictor.fit(train_x, train_y)

    def predict(self, score) -> float:
        return float(self.predictor.predict_proba([[score]])[0, 1])
