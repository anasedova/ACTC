import random
from abc import ABC
from typing import Dict, List, Tuple, Union

import numpy as np

from scripts.optimizers.base_optimizer import BaseOptimizer
from scripts.utils import choose_threshold_with_expected_metric, calculate_sample_distances


# num_samples = 1000


class BasePredictionOptimizer(BaseOptimizer):
    def __init__(
            self,
            all_scores,
            partial_labels: Dict = None,
            thres_gap: float = 0.01,
            iteration_limit: int = 2,
            metric: str = "f1_score",
            selection_mechanism: str = "random",
            estimation_type: str = "estimate_all",
            **kwargs
    ):
        super().__init__(
            all_scores, partial_labels, thres_gap, iteration_limit, metric, **kwargs
        )
        self.selection_mechanism = selection_mechanism
        self.estimation_type = estimation_type

        assert self.selection_mechanism in ["random", "uncertainty", "density_weighted", "density_only"]
        assert self.estimation_type in ["estimate_all", "estimate_unknown", "estimate_none"]

        self.lr = 0.0

    def estimate_predictor(self):
        raise NotImplementedError()

    def predict(self, score) -> float:
        """Returns predictions between 0 and 1, decision boundary 0.5."""
        raise NotImplementedError()

    def more_labels(self, n_more: int, gold_labels: Dict) -> Dict:
        """
        Select the entries that will be used as gold ones.
        Depending on the self.selection_mechanism value, the entries are selected either randomly, or by uncertainty
            (alternatively weighted by distance), or by distance only.
        Returns: a dictionary {entry: label}

        """
        triples = list(gold_labels.keys())  # all triples from which we select some entries with gold labels
        random.shuffle(triples)  # shuffle them

        if self.selection_mechanism == "random":  # select n_more entries randomly
            return self.select_partial(gold_labels, triples, n_more)

        else:
            if not hasattr(self, 'predictor'):  # initialize estimator if there is non yet
                # print("INITIAL ESTIMATION")
                self.estimate_predictor()

            if self.selection_mechanism == "uncertainty":
                return self.select_by_uncertainty(gold_labels, n_more, weight_by_distance=False)

            elif self.selection_mechanism == "density_weighted":
                return self.select_by_uncertainty(gold_labels, n_more, weight_by_distance=True)

            elif self.selection_mechanism == "density_only":
                return self.select_by_density_only(gold_labels, n_more)

            else:
                raise NotImplementedError()

    def find_optimal_threshold(self, params: Dict, curr_relation: str = None) -> float:
        """
        Optimize thresholds basing on selected samples. The samples are selected based on the self.estimation_type
            parameter value.
        Returns: thresholds dict {relation : threshold}
        """
        assert self.estimation_type is not None
        self.estimate_predictor()

        # select the estimated triples that will be used for the threshold optimization
        estimated_labels = self.select_samples_for_estimation(params)

        # print(f"All labels the threshold is estimated from: {len(estimated_labels)}")

        if curr_relation:
            partial_scores = {}
            for triple, score in self.all_scores.items():
                if triple[1] == curr_relation:
                    partial_scores[triple] = score
            rel_global = {**partial_scores, **estimated_labels}
        else:
            rel_global = self.all_scores

        # return optimized threshold
        return choose_threshold_with_expected_metric(rel_global, estimated_labels, metric=self.metric)

    def select_samples_for_estimation(self, params: Dict) -> Dict:
        """
        Select the triples and their labels that will be used for threshold optimization.
            - if estimate_all: use the triples with newly calculated labels for threshold optimization.
                The num_samples_update triples (may equal all samples if num_samples_update is None) are selected for
                the estimation randomly (every nth).
                Some samples might be already included to the self.partial_labels, but their labels will be recalculated
                as well.
            - if estimate_unknown: use self.partial_labels + newly calculated (num_samples_update-|self.partial_labels|)
                triples for threshold optimization.
            - if estimate_none: use only self.partial_labels for threshold optimization
        """
        if params["num_samples_update"] is None:
            num_samples_update = len(self.all_scores)
        else:
            num_samples_update = params["num_samples_update"]

        if self.estimation_type == "estimate_all":
            # the labels for the samples (#samples is specified by num_samples_update) will be estimated
            estimated_labels = dict()
            additional_samples = num_samples_update
        elif self.estimation_type == "estimate_unknown":
            # the labels for the samples (#samples = num_samples_update - #partial labels) will be estimated
            estimated_labels = self.partial_labels.copy()
            additional_samples = max(0, num_samples_update - len(self.partial_labels))
        elif self.estimation_type == "estimate_none":
            # no additional sample labels will be estimates
            estimated_labels = self.partial_labels.copy()
            additional_samples = 0
        else:
            raise NotImplementedError()

        # if estimate_all or estimate_unknown: estimate additional labels and use them for threshold optimization
        if self.estimation_type != "estimate_none" and additional_samples > 0:
            every_nth = len(self.all_scores) // additional_samples + 1
            estimated_labels.update(self.estimate_more_samples(every_nth, params["soft"]))

        return estimated_labels

    def select_partial(self, gold_labels: Dict, triples: List, n_more: int) -> Dict:
        """
        Selects the n triples for the triples list and return a dict with triples and the corresponding gold labels.
        Output: {triple: gold label}
        """
        result = {}
        for triple in triples:  # select n triples which has a score that is close to 0.5
            if triple not in self.partial_labels.keys():
                if len(result) < n_more:
                    result[triple] = gold_labels[triple]
                else:
                    break
        return result

    def select_by_uncertainty(self, gold_labels: Dict, n_more: int, weight_by_distance: bool = False) -> Dict:
        """
        The n_more samples are selected based on the "uncertainty score":
            - the scores for all samples are predicted by the self.predictor basing on the scores from the emb. model
            - (if weight_by_distance set to True) for each sample, the sum of the distances to other samples are
                calculated.
            - (if weight_by_distance set to False) sum_distance for each sample = 1
            - the n samples with (the closest to 0.5 predicted score) * distance are selected as gold samples
                (= the samples that are the closest to the middle threshold).
        Returns: the dictionary {triples selected as gold samples: gold label}
        """
        uncertainty_keys = []
        triples = list(gold_labels.keys())
        scores = [self.all_scores[key_triple] for key_triple in triples]

        if weight_by_distance:
            sum_distances = calculate_sample_distances(np.array(scores))
        else:
            sum_distances = np.ones((len(scores)))

        for key_triple, score, distance in zip(triples, scores, sum_distances):
            prediction = self.predict(score)  # for each entry predict the score basing on the model score
            # (score diff from 0.5 * dist, triple)
            uncertainty_keys.append((abs(0.5 - prediction) * distance, key_triple))
        uncertainty_keys = sorted(uncertainty_keys)  # sorted tuples (uncertainty score, (ent1, rel, ent2))

        # select n triples which has a score that is close to 0.5
        return self.select_partial(
            gold_labels,
            [key_triple for (_, key_triple) in uncertainty_keys],  # triples sorted according to the uncertainty scores
            n_more
        )

    def select_by_density_only(self, gold_labels: Dict, n_more: int) -> Dict:
        """
        NB! no predictions from the predictor are used here.
        The n_more samples are selected based on the density:
            - for each sample, the sum of the distances to other samples is calculated -> sum_distances
            - the n samples with the smallest distances to other samples are selected as gold samples.
        Returns: the dictionary {triples selected as gold samples: gold label}
        """
        distances = []
        triples = list(gold_labels.keys())
        scores = np.array([self.all_scores[key_triple] for key_triple in triples])
        sum_distances = calculate_sample_distances(scores)

        for key_triple, score, distance in zip(triples, scores, sum_distances):
            # (sum of distances from the current triple to others, triple)
            distances.append((distance, key_triple))
        distances = sorted(distances)

        return self.select_partial(
            gold_labels,
            [key_triple for (_, key_triple) in distances],
            n_more
        )

    def estimate_more_samples(self, every_nth: int, soft: bool, threshold: Union[Dict, float] = 0.5) -> Dict:
        """
        Estimate the labels of additional triples from the self.all_scores.
            - for what triples should the labels be predicted?
                - if estimation_type="estimate_all", take every nth triple
                - if estimation_type="estimate_unknown", additionally check that the triple is not estimated yet
                    (i.e., it is not in self.partial_labels yet)
            - predict the score for the selected triples
            - how the labels should look like?
                - if soft==True: predict the score in range [0, 1] -> label := score
                - if soft==False: turn the prediction into 1 if it is larger than 0.5, 0 otherwise -> label := 0/1
        Return: dictionary {triple: label}
        """

        if threshold is Dict and len(threshold) == 0:
            threshold = 0.5
        elif threshold is Dict and len(threshold) > 0:
            threshold = sum(threshold.values()) / len(threshold)

        newly_estimated_samples = {}
        for i, (key, score) in enumerate(self.all_scores.items()):
            if (i + 1) % every_nth == 0 and (key not in self.partial_labels or self.estimation_type == "estimate_all"):
                prediction = self.predict(score)  # float(self.predictor.predict_proba([[score]])[0, 1])
                if not soft:
                    prediction = 1.0 if prediction > threshold else 0.0
                newly_estimated_samples[key] = prediction
        return newly_estimated_samples

    def estimate_more_samples_density_weighted(self, every_nth: int, soft: bool, threshold: float = 0.5) -> Dict:
        from sklearn import preprocessing
        sum_distances = calculate_sample_distances(np.array(list(self.all_scores.values())))
        newly_estimated_samples = {}
        keys, predictions, predictions_weighted = [], [], []
        for distance, (i, (key, score)) in zip(sum_distances, enumerate(self.all_scores.items())):
            if (i + 1) % every_nth == 0 and (key not in self.partial_labels or self.estimation_type == "estimate_all"):
                keys.append(key)
                predictions_weighted.append(abs(0.5 - self.predict(score)) * distance)
        normalized_scores = np.ndarray.tolist(preprocessing.normalize([np.array(predictions_weighted)]))[0]
        for key, pred in zip(keys, normalized_scores):
            if not soft:
                pred = 1.0 if pred > threshold else 0.0
            newly_estimated_samples[key] = pred
        return newly_estimated_samples

    def get_partial_scores_labels(self) -> Tuple[List, List]:
        min_score = min(self.all_scores.values())
        max_score = max(self.all_scores.values())

        scores = [[min_score], [max_score]]
        labels = [0.0, 1.0]
        for key in self.partial_labels:
            scores.append([self.all_scores[key]])
            labels.append(self.partial_labels[key])
        return scores, labels


class UniPredictionOptimizer(BasePredictionOptimizer, ABC):
    def __init__(
            self,
            all_scores,
            partial_labels: Dict = None,
            thres_gap: float = 0.01,
            iteration_limit: int = 2,
            metric: str = "f1_score",
            selection_mechanism: str = "random",
            estimation_type: str = "estimate_all",
            **kwargs
    ):
        super().__init__(
            all_scores, partial_labels, thres_gap, iteration_limit, metric, selection_mechanism, estimation_type,
            **kwargs
        )

    def optimal_thresholds(self, **kwargs) -> Dict:
        """
        Optimize thresholds basing on selected samples. The samples are selected based on the self.estimation_type
            parameter value.
        Returns: thresholds dict {relation : threshold}
        """
        relations = self.get_relations()
        global_threshold = self.find_optimal_threshold(kwargs["params"])
        thresholds = {self.rel_key: global_threshold}
        for r in relations:
            thresholds[r] = global_threshold
        return thresholds


class PredictionOptimizer(BasePredictionOptimizer, ABC):
    def __init__(
            self,
            all_scores,
            partial_labels: Dict = None,
            thres_gap: float = 0.01,
            iteration_limit: int = 2,
            metric: str = "f1_score",
            selection_mechanism: str = "random",
            estimation_type: str = "estimate_all",
            **kwargs
    ):
        super().__init__(
            all_scores, partial_labels, thres_gap, iteration_limit, metric, selection_mechanism, estimation_type,
            **kwargs
        )

    def optimal_thresholds(self, **kwargs) -> Dict:
        # print(f"Overall partial samples: {len(self.partial_labels)}")
        # self.estimate_predictor()

        all_partial = self.partial_labels.copy()

        # get the relations from the triples with known gold labels
        relations = self.get_relations()

        if kwargs["params"]["estimate_for_unknown"]:
            # including additionally estimated samples
            global_threshold = self.find_optimal_threshold(kwargs["params"])
        else:
            # global threshold is predicted only based on the manually annotated samples
            global_threshold = choose_threshold_with_expected_metric(self.all_scores, self.partial_labels, self.metric)

        # initialize thresholds dict; the default threshold (for -1 relation) basing on all known scores and labels
        thresholds = {self.rel_key: global_threshold}

        for r in relations:
            # get the dictionary {triple: label} for each relation
            partial_labels_with_r = {triple: label for triple, label in all_partial.items() if triple[1] == r}
            assert len(partial_labels_with_r) > 0
            self.partial_labels = partial_labels_with_r
            thresholds[r] = self.find_optimal_threshold(kwargs["params"])
            # print(f"For relation {r}: partial samples {len(partial_labels_with_r)}")

        self.partial_labels = all_partial
        # print("=========================== DONE ===========================")
        return thresholds
