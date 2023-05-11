import argparse
import os
import random
import json
from typing import Dict

import pandas as pd

from tqdm import tqdm

from scripts.optimizers.global_optimizer import GlobalOptimizer
from scripts.optimizers.local_accuracy_optimizer import LocalOptAcc
from scripts.optimizers.local_accuracy_optimizer_uni import UniLocalOptAcc
from scripts.optimizers.local_f1_optimizer import LocalOptF1
from scripts.optimizers.local_f1_optimizer_uni import UniLocalOptF1
from scripts.optimizers.gp_optimizer import GaussianProcessOptimizer
from scripts.optimizers.logreg_optimizer import LogisticRegressionOptimizer
from scripts.optimizers.gp_optimizer_global import UniGaussianProcessOptimizer
from scripts.optimizers.logreg_optimizer_uni import UniLogisticRegressionOptimizer
from scripts.optimizers.simple_pred_optimizer import BasePredictionOptimizer
from scripts.utils import read_data, evaluate_acc_f1, create_plots, update_scores_dict, calculate_mean_sem, set_seed, \
    calculate_overall_metrics, add_res_to_df, save_results_csv

N_more = [1, 1, 3, 5, 10, 30, 50, 100, 300, 500]  # number of new labels randomly sampled from gold validation labels

Optimizers = {
    "Pred": BasePredictionOptimizer,
    "LogReg": LogisticRegressionOptimizer,
    "UniLogReg": UniLogisticRegressionOptimizer,
    "GP": GaussianProcessOptimizer,
    "UniGP": UniGaussianProcessOptimizer,
    "Acc": LocalOptAcc,
    "UniAcc": UniLocalOptAcc,
    "F1": LocalOptF1,
    "UniF1": UniLocalOptF1,
    "GlobalF1": GlobalOptimizer
}

OPTIONS_TO_ACC, OPTIONS_TO_F1 = {}, {}


def run_threshold_finding(optimizer, gd_scores, gd_labels, gt_scores, gt_labels, par):

    print(f"Optimizer: {optimizer}")
    print(f"params: {str(par)}")

    # set seed
    if par["seed"]:
        set_seed(par["seed"])

    n_more_to_acc, n_more_to_f1 = {}, {}  # {number of gold entries : acc/f1 score}

    for _ in tqdm(range(par["num_exp"])):
        partial_labels = {}  # a dictionary with selected validation triples as keys, gold labels as values
        t_opt = Optimizers[optimizer](
            all_scores=gd_scores,  # scores from dev set
            partial_labels=partial_labels,
            init_threshold_option=par["init_threshold_option"],
            thres_gap=par["thres_gap"],
            iteration_limit=par["iteration_limit"],
            metric=par["metric"],
            selection_mechanism=par["selection_mechanism"],
            estimation_type=par["estimation_type"],
            kernel=par["kernel"]
        )

        # retain_ratio = 0.5
        retain_ratio = 1
        subsampled_gd_labels = {k: v for k, v in gd_labels.items() if random.random() <= retain_ratio}
        tuned_thresholds = {}

        for n_more in N_more:
            new_labels = t_opt.more_labels(n_more, subsampled_gd_labels)  # add new gold entries from dev set
            t_opt.partial_labels.update(new_labels)  # add new gold entries to the entries dict
            tuned_thresholds = t_opt.optimal_thresholds(params=par, thresholds=tuned_thresholds)  # calculate thresholds
            curr_acc, curr_f1 = evaluate_acc_f1(gt_scores, gt_labels, tuned_thresholds)  # eval current state
            curr_size = len(t_opt.partial_labels)  # total amount of current gold entries

            # update the dictionary with metrics for the current n_more update
            n_more_to_acc = update_scores_dict(curr_acc, curr_size, n_more_to_acc)
            n_more_to_f1 = update_scores_dict(curr_f1, curr_size, n_more_to_f1)

    # calculate mean and sem for accuracy and f1 across all experiments for each iteration
    means_acc, sems_acc, means_f1, sems_f1 = calculate_mean_sem(n_more_to_acc, n_more_to_f1)

    # save the metrics for the current optimizer
    OPTIONS_TO_ACC[optimizer] = (means_acc, sems_acc)
    OPTIONS_TO_F1[optimizer] = (means_f1, sems_f1)

    # calculate global averages
    global_results = calculate_overall_metrics(means_acc, means_f1)

    return add_res_to_df(optimizer, par, global_results)


def main(path_to_models: str, output_dir: str = None, par: Dict = None):
    df_results = pd.DataFrame(columns=["opt"] + list(par.keys())[1:] + ["acc", "f1", "sem_f1", "sem_acc"])
    gd_scores, gd_labels, gt_scores, gt_labels = read_data(path_to_models, par["size"], par["model"])

    if type(par["optimizer_options"]) == str:
        res = run_threshold_finding(par["optimizer_options"], gd_scores, gd_labels, gt_scores, gt_labels, par)
        df_results = df_results.append(res, ignore_index=True)
    else:
        for i, opt_option in enumerate(par["optimizer_options"]):
            res = run_threshold_finding(opt_option, gd_scores, gd_labels, gt_scores, gt_labels, par)
            df_results = df_results.append(res, ignore_index=True)

    save_results_csv(df_results, os.path.join(output_dir, par["size"]))

    for i, (opt, result) in enumerate(OPTIONS_TO_ACC.items()):
        sems = list(result[1].values())
        print("option: ", opt)
        avg_result = 0
        for j, (size, metric) in enumerate(result[0].items()):
            if size in (1, 10, 50):
                print(f"size: {size}, \t acc: {round(metric, 2)}, \t sem: {round(sems[j], 1)}")
            avg_result += metric
        print(avg_result / len(result[0]))

    for i, (opt, result) in enumerate(OPTIONS_TO_F1.items()):
        sems = list(result[1].values())
        print("option: ", opt)
        avg_result = 0
        for j, (size, metric) in enumerate(result[0].items()):
            if size in (1, 10, 50):
                print(f"size: {size}, \t F1: {round(metric, 2)}, \t sem: {round(sems[j] * 100, 1)}")
            avg_result += metric
        print(avg_result / len(result[0]))

    if par["draw_plots"]:
        create_plots(OPTIONS_TO_ACC, par, n_more=N_more)
        create_plots(OPTIONS_TO_F1, par, n_more=N_more)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_to_models", type=str, default="./")
    parser.add_argument("--output_dir", default="../out/", type=str)
    parser.add_argument("--path_to_config", type=str, default="./config.json")
    args = parser.parse_args()

    with open(args.path_to_config) as config_file:
        params = json.load(config_file)
        print(params)

    main(args.path_to_models, args.output_dir, params)
