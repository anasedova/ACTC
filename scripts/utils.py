import os
import csv
import random
import statistics
from typing import Dict, Tuple, Union, List

import torch
import numpy as np
import pandas as pd
import matplotlib as mpl

from scipy.stats import sem
from matplotlib import pyplot as plt
from sklearn.metrics import f1_score, accuracy_score
from pylab import cm

mpl.rcParams['font.family'] = 'Avenir'
plt.rcParams['font.size'] = 13
plt.rcParams['axes.linewidth'] = 2

REL_KEY = -1


def read_data(path_to_models, size, model) -> Tuple[Dict, Dict, Dict, Dict]:
    """
    Args:
        path_to_models:
        size:
        model:
    Output:
        gd_scores - a dictionary with all validation triples as keys, gold scores as values
        gd_labels - a dictionary with all validation triples as keys, gold labels as values
        gt_scores - a dictionary with all test triples as keys, gold scores as values
        gt_labels - a dictionary with all test triples as keys, gold labels as values
    """
    print("=== " + size + "/" + model + " ===")

    gd_scores = {}
    with open(os.path.join(path_to_models, f"codex-{size}", model, "valid_score.tsv"), "r") as vs:
        gd_score_reader = csv.reader(vs, delimiter="\t")
        for row in gd_score_reader:
            gd_scores[tuple([i for i in row[:3]])] = float(row[3])

    gd_labels = {}
    with open(os.path.join(path_to_models, f"codex-{size}", "valid_label.tsv"), "r") as vl:
        gd_label_reader = csv.reader(vl, delimiter="\t")
        for row in gd_label_reader:
            gd_labels[tuple([i for i in row[:3]])] = float(row[3])

    gt_scores = {}
    with open(os.path.join(path_to_models, f"codex-{size}", model, "test_score.tsv"), "r") as ts:
        gt_score_reader = csv.reader(ts, delimiter="\t")
        for row in gt_score_reader:
            gt_scores[tuple([i for i in row[:3]])] = float(row[3])

    gt_labels = {}
    with open(os.path.join(path_to_models, f"codex-{size}", "test_label.tsv"), "r") as tl:
        gt_label_reader = csv.reader(tl, delimiter="\t")
        for row in gt_label_reader:
            gt_labels[tuple([i for i in row[:3]])] = float(row[3])
    return gd_scores, gd_labels, gt_scores, gt_labels


def choose_threshold_with_expected_metric(
        gd_scores: Dict[Tuple, float], partial_labels: Dict[Tuple, float], metric: str = "accuracy"
) -> float:
    """
    Args:
        gd_scores: the scores for all samples
        partial_labels: gold entries (for manually annotated samples)
        metric: metric, basing on which the new thresholds will be calculated; accuracy or f1 score
    Output:
    """

    # the tensors of known gold labels
    expected_labels = (torch.FloatTensor([label for _, label in partial_labels.items()])).view(-1)
    # the tensor of scores for known gold entries
    scores = (torch.FloatTensor([gd_scores[triple] for triple, _ in partial_labels.items()])).view(-1, 1)

    # print("apply all thresholds for all scores ...")
    predictions_from_thresholds = ((scores.view(-1, 1) >= scores.view(1, -1)).long()).t()
    # print(" ... done!")

    # prediction=True, should_probably_be=True
    true_positives = (predictions_from_thresholds * expected_labels[None, :]).sum(dim=1).float()

    if metric == "accuracy":
        # prediction=False, should_probably_be=False
        true_negatives = ((1 - predictions_from_thresholds) * (1 - expected_labels[None, :])).sum(dim=1).float()
        results_for_thresholds = (true_positives + true_negatives).float()
    elif metric == "f1_score":
        # prediction=True, should_probably_be=False
        false_positives = (predictions_from_thresholds * (1 - expected_labels[None, :])).sum(dim=1).float()
        # prediction=False, should_probably_be=True
        false_negatives = ((1 - predictions_from_thresholds) * (expected_labels[None, :])).sum(dim=1).float()
        results_for_thresholds = true_positives / (true_positives + 0.5 * false_positives + 0.5 * false_negatives)
    else:
        raise ValueError(f"{metric} metric is not supported. Currently supported metrics are accuracy and f1 score")

    max_result = results_for_thresholds.max()
    threshold = scores[max_result == results_for_thresholds].min().item()

    return threshold


def evaluate_acc_f1(
        dict_scores: Dict[Tuple, float], dict_labels: Dict[Tuple, float], thresholds: Union[Dict[str, float], float]
) -> Tuple[float, float]:
    pred_labels = []
    true_labels = []

    for triple, label in dict_labels.items():
        true_labels.append(label)
        if type(thresholds) is dict:
            relation = triple[1] if triple[1] in thresholds.keys() else REL_KEY
            # append 1 if score of the entry > threshold for the corresponding relation, otherwise 0
            pred_labels.append(int(dict_scores[triple] >= thresholds[relation]))
        else:
            pred_labels.append(int(dict_scores[triple] >= thresholds))

    return accuracy_score(true_labels, pred_labels), f1_score(true_labels, pred_labels, zero_division=1)


def create_plots(options: Dict, params: Dict, n_more: List) -> None:
    colors = cm.get_cmap('tab10').colors[:len(options)]
    fig, ax = plt.subplots()
    assert len(options) >= 1
    for i, (opt, result) in enumerate(options.items()):
        print("option: ", opt)
        result *= 100
        x = list(range(len(n_more)))
        # y = [val * 100 for val in list(result[0].values())]
        y = list(result[0].values())
        err = list(result[1].values())
        x_idx = list(result[0].keys())

        y = [ys * 100 for ys in y]
        err = [errs * 100 for errs in err]

        # error bars
        # plt.errorbar(x, y, err, label=opt, color=colors(i))
        # continuous error (filling the error space)
        err_1 = [n_y - n_err for n_y, n_err in zip(y, err)]
        err_2 = [n_y + n_err for n_y, n_err in zip(y, err)]
        plt.plot(x, y, label=opt, marker='o', color=colors[i])
        plt.fill_between(x, err_2, err_1, alpha=0.2, color=colors[i])
        ax.set_xticks(x)
        ax.set_xticklabels(x_idx)

    ax.xaxis.set_tick_params(which='major', size=8, width=1, direction='out')
    ax.yaxis.set_tick_params(which='major', size=8, width=1, direction='out')
    ax.grid(which='major', alpha=0.3)
    plt.xlabel("Number of Labeled Samples")
    plt.ylabel(f"Accuracy (in %)")
    # plt.title(f"{params}", ha='center', fontdict=None, loc='center', pad=None)
    plt.subplots_adjust(bottom=0.35)
    plt.figtext(0.01, 0.04, str(params), fontsize=10, wrap=True)

    # plt.legend()
    plt.legend(handletextpad=0.3, labelspacing=0.001, borderpad=0.05)
    plt.setp(plt.gca().get_legend().get_texts(), fontsize='9')

    plt.show()


def update_scores_dict(curr_score: float, curr_key: int, scores_dict: Dict) -> Dict:
    new_scores = []
    if curr_key in scores_dict.keys():  # update the dict with scores
        new_scores = scores_dict.get(curr_key)
    new_scores.append(curr_score)
    scores_dict[curr_key] = new_scores
    return scores_dict


def calculate_mean_sem(n_more_to_acc, n_more_to_f1) -> Tuple[Dict, Dict, Dict, Dict]:
    return {key: np.mean(n_more_to_acc[key]) for key in n_more_to_acc.keys()}, \
           {np.log(key): sem(n_more_to_acc[key]) for key in n_more_to_acc.keys()}, \
           {key: np.mean(n_more_to_f1[key]) for key in n_more_to_f1.keys()}, \
           {np.log(key): sem(n_more_to_f1[key]) for key in n_more_to_f1.keys()}


def calculate_sample_distances(scores: np.array) -> np.array:
    """ Calculate squared distances between all scores, summed for each instance"""
    return ((scores - scores.reshape((-1, 1))) ** 2).sum(axis=0)


def set_seed(seed: int) -> None:
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def calculate_overall_metrics(means_acc, means_f1) -> Dict:
    mean_acc = round(statistics.mean(list(means_acc.values())), 2)
    sem_acc = round(sem(list(means_acc.values())), 2)
    mean_f1 = round(statistics.mean(list(means_f1.values())), 2)
    sem_f1 = round(sem(list(means_f1.values())), 2)
    return {"mean_acc": mean_acc, "sem_acc": sem_acc, "mean_f1": mean_f1, "sem_f1": sem_f1}


def add_res_to_df(optimizer: str, params: Dict, rel_res: Dict) -> pd.Series:
    curr_res = pd.Series(
        {
            **{"opt": optimizer},
            **dict(list(params.items())[1:]),
            **{"acc": rel_res["mean_acc"], "f1": rel_res["mean_f1"]}
        }
    )
    curr_res["sem_acc"] = rel_res["sem_acc"]
    curr_res["sem_f1"] = rel_res["sem_f1"]
    return curr_res


def save_results_csv(df_results, output_path) -> None:
    os.makedirs(output_path, exist_ok=True)
    output_file_path = os.path.join(output_path, "output.csv")
    if os.path.exists(output_file_path):
        df_results = pd.concat([pd.read_csv(output_file_path), df_results])
    df_results.to_csv(output_file_path, index=False)
