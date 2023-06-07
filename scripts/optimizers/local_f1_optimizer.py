from typing import Dict

from ACTC.scripts.optimizers.base_optimizer import BaseOptimizer
from ACTC.scripts.utils import choose_threshold_with_expected_metric


class LocalOptF1(BaseOptimizer):
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
        thresholds = {
            # the threshold for all not-yet-covered relations basing on all known scores and labels
            self.rel_key: choose_threshold_with_expected_metric(self.all_scores, self.partial_labels, self.metric)
        }

        relations = self.get_relations()            # get the relations from the triples with known gold labels

        for r in relations:
            # get the dictionary {triple: label} for each relation
            partial_labels_with_r = {triple: label for triple, label in self.partial_labels.items() if triple[1] == r}
            assert len(partial_labels_with_r) > 0

            # get the dictionary {triple: score} for each relation
            scores_with_r = {k: self.all_scores[k] for k in partial_labels_with_r.keys()}

            # recalculate the thresholds for the known relations and use the
            thresholds[r] = choose_threshold_with_expected_metric(scores_with_r, partial_labels_with_r, self.metric)

        return thresholds
