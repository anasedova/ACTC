from typing import Dict

from ACTC.scripts.optimizers.base_optimizer import BaseOptimizer
from ACTC.scripts.utils import choose_threshold_with_expected_metric


class UniLocalOptF1(BaseOptimizer):
    def __init__(
            self,
            all_scores,
            partial_labels: Dict = None,
            thres_gap: float = 0.01,
            iteration_limit: int = 2,
            metric: str = "f1_score",
            init_threshold_option: str = "trivial",
            **kwargs
    ):
        super().__init__(
            all_scores, partial_labels, thres_gap, iteration_limit, metric, init_threshold_option, **kwargs
        )
        if self.metric != "f1_score":
            print("As LocalOptF1 is selected, the metric will be changed to 'f1_score'")
            self.metric = "f1_score"

    def optimal_thresholds(self, **kwargs) -> Dict:
        global_threshold = choose_threshold_with_expected_metric(self.all_scores, self.partial_labels, self.metric)
        thresholds = {self.rel_key: global_threshold}

        relations = self.get_relations()
        for r in relations:
            thresholds[r] = global_threshold
        return thresholds
