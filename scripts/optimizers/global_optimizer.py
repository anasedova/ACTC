from typing import Dict, Set

from scripts.optimizers.base_optimizer import BaseOptimizer
from scripts.utils import evaluate_acc_f1, choose_threshold_with_expected_metric


class GlobalOptimizer(BaseOptimizer):
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

        assert init_threshold_option in ["trivial", "accuracy"]
        self.options = {
            "trivial": self.initialize_thresholds_trivial, "accuracy": self.initialize_thresholds_with_accuracy
        }

    def initialize_thresholds_trivial(self, relations: Set) -> Dict:
        """ Initialize the thresholds for all relations with 0.5 """
        thresholds_init = {self.rel_key: 0.5}       # set "default" threshold for non-known relations to 0.5
        for relation in relations:
            thresholds_init[relation] = 0.5         # set thresholds for all other known relations also to 0.5
        return thresholds_init

    def initialize_thresholds_with_accuracy(self, relations: Set) -> Dict:
        """ Initialize the thresholds with accuracy basing on the provided gold entries """
        thresholds_init = {
            self.rel_key: choose_threshold_with_expected_metric(self.all_scores, self.partial_labels, "accuracy")
        }
        for relation in relations:
            r_labels = {
                triple: self.partial_labels[triple] for triple in self.partial_labels.keys() if triple[1] == relation
            }
            thresholds_init[relation] = choose_threshold_with_expected_metric(self.all_scores, r_labels, "accuracy")
        return thresholds_init

    def optimal_thresholds(self, **kwargs) -> Dict:
        relations = self.get_relations()            # get all relations
        # initialize thresholds for each relation
        tuned_thresholds = self.options[self.init_threshold_option](relations)

        # select the best universal threshold for the unknown yet relations basing on all known gold entries
        best_global_metric = -float("inf")
        best_threshold_global = -float("inf")

        for pre_t in self.predefined_thres:         # try out all predefined thresholds and select the best global one
            current_acc_global, current_f1_global = evaluate_acc_f1(self.all_scores, self.partial_labels, pre_t)

            if self.metric == "accuracy":
                curr_metric = current_acc_global
            elif self.metric == "f1_score":
                curr_metric = current_f1_global
            else:
                raise ValueError(f"Unsupported metric {self.metric}! Currently supported are 'accuracy' and 'f1_score'")

            if curr_metric > best_global_metric:
                best_threshold_global = pre_t
                best_global_metric = curr_metric

        tuned_thresholds[self.rel_key] = best_threshold_global
        # tuned_thresholds[REL_KEY] = -float("inf")

        i = 0
        while i < self.iteration_limit:
            for relation in relations:
                best_metric = -float("inf")
                best_threshold = -float("inf")

                for pre_t in self.predefined_thres:
                    tuned_thresholds[relation] = pre_t
                    current_acc, current_f1 = evaluate_acc_f1(self.all_scores, self.partial_labels, tuned_thresholds)

                    if self.metric == "accuracy":
                        curr_metric = current_acc
                    elif self.metric == "f1_score":
                        curr_metric = current_f1
                    else:
                        raise ValueError(
                            f"Unsupported metric {self.metric}! Currently supported are 'accuracy' and 'f1_score'")

                    if curr_metric > best_metric:
                        best_threshold = pre_t
                        best_metric = curr_metric

                tuned_thresholds[relation] = best_threshold
            i += 1

        return tuned_thresholds
