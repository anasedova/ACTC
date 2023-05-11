import random
from typing import Dict, Set, Union

import numpy as np


class BaseOptimizer:

    def __init__(
            self,
            all_scores,
            partial_labels: Dict = None,
            thres_gap: float = 0.01,
            iteration_limit: int = 2,
            metric: str = "accuracy",
            init_threshold_option: str = "trivial",
            **kwargs
    ):
        if partial_labels is None:
            partial_labels = {}
        self.all_scores = all_scores
        self.partial_labels = partial_labels
        self.init_threshold_option = init_threshold_option
        self.thres_gap = thres_gap
        self.iteration_limit = iteration_limit
        self.metric = metric
        self.rel_key = -1

        self.predefined_thres = list(np.arange(self.thres_gap, 1, self.thres_gap))

        # self.print_params()

    def get_relations(self) -> Set:
        """ Extract the relations from the triples with known gold labels """
        return set([triple[1] for triple in list(self.partial_labels.keys())])

    def optimal_thresholds(self) -> Union[float, Dict]:
        pass

    def more_labels(self, n_more: int, gold_labels: Dict) -> Dict:
        """ Randomly select and return n_more items from gold_labels dictionary """
        triples_keys = list(gold_labels.keys())
        random.shuffle(triples_keys)
        result = {}
        for triple_key in triples_keys:
            if triple_key not in self.partial_labels.keys():
                if len(result) < n_more:
                    result[triple_key] = gold_labels[triple_key]
                else:
                    break
        return result

    def print_params(self):
        print(f"init_threshold_option: {self.init_threshold_option}")
        print(f"thres_gap: {self.thres_gap}")
        print(f"iteration_limit: {self.iteration_limit}")
        print(f"rel_key: {self.rel_key}")
