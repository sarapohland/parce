import os
import torch
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from pathlib import Path
from sklearn import metrics
from tabulate import tabulate
import matplotlib.pyplot as plt
from pytorch_ood.utils import OODMetrics

from src.analysis.plot import plot_comp_vs_acc_bins
from src.analysis.utils import load_results_by_property, factors
from src.analysis.modify import ALL_MODIFICATIONS, EVAL_MODIFICATIONS
from src.comparison.overall.methods import ALL_OVERALL

def main():
    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('--test_data', type=str, default='lunar')
    parser.add_argument('--data_dir', type=str, default='results/lunar/competency/modified/data/')
    parser.add_argument('--plot_dir', type=str, default='results/lunar/competency/modified/plots/')
    args = parser.parse_args()

    # Create folders to save results
    roc_folder = os.path.join(args.plot_dir, 'roc')
    if not os.path.exists(roc_folder):
        os.makedirs(roc_folder)

    distr_folder = os.path.join(args.plot_dir, 'comp_distr')
    if not os.path.exists(distr_folder):
        os.makedirs(distr_folder)

    results_folder = Path(args.data_dir).parent

    # Create dictionary to store data to be displayed in table
    data = {'Method': [],
            'H-M Dist': [], 'H-M AUROC': [], 'H-M FPR': [],
            'H-L Dist': [], 'H-L AUROC': [], 'H-L FPR': [],
            'M-L Dist': [], 'M-L AUROC': [], 'M-L FPR': [],
            }
    
    # Collect data for each competency method
    for method in ALL_OVERALL:
        data['Method'].append(method)
        accuracies, competencies = [], []
        # for property in ALL_MODIFICATIONS:
        for property in EVAL_MODIFICATIONS:
            # Load saved data for given property
            folder = os.path.join(args.data_dir, property)
            _, datas = load_results_by_property(folder)
            # _, datas = load_results_by_property(folder, factors[args.test_data][property])
            accuracy = np.array([data['accuracy'] for data in datas])
            competency = np.array([data[method] for data in datas])

            # Save all accuracy and competency data
            accuracies.append(accuracy)
            competencies.append(competency)
        accuracies = np.hstack(accuracies)
        competencies = np.vstack(competencies)

        # Bin the accuracy values
        bin_edges = np.array([0.0, 0.5, 0.9, 1.0])
        # bin_edges = np.array([0.0, 0.5, 0.85, 1.0])
        bin_indices = np.digitize(accuracies, bins=bin_edges)

        # Collect competencies for each bin
        high = competencies[bin_indices == 3].flatten()
        med  = competencies[bin_indices == 2].flatten()
        low  = competencies[bin_indices == 1].flatten()

        # Replace NaN values with minimum
        min_value = np.minimum(np.minimum(np.nanmin(high), np.nanmin(med)), np.nanmin(low))
        high[np.isnan(high)] = min_value
        med[np.isnan(med)] = min_value
        low[np.isnan(low)] = min_value

        # Plot score distributions
        fig, ax = plt.subplots()
        sns.boxplot(data=[high, med, low], ax=ax)
        ax.set_xticklabels(['High ({}-{})'.format(bin_edges[2], bin_edges[3]), 
                        'Med ({}-{})'.format(bin_edges[1], bin_edges[2]), 
                        'Low ({}-{})'.format(bin_edges[0], bin_edges[1])])
        plt.title('{} Competency Estimates'.format(method.capitalize()))
        plt.xlabel('Prediction Accuracy')
        plt.ylabel('Competency Score')
        plt.savefig(os.path.join(distr_folder, '{}.png'.format(method)))

        # Compare each set of samples
        hm_scores = torch.from_numpy(np.hstack([high, med]))
        hl_scores = torch.from_numpy(np.hstack([high, low]))
        ml_scores = torch.from_numpy(np.hstack([med, low]))
        hm_labels = torch.from_numpy(np.hstack([np.ones_like(high), np.full_like(med, -1)]))
        hl_labels = torch.from_numpy(np.hstack([np.ones_like(high), np.full_like(low, -1)]))
        ml_labels = torch.from_numpy(np.hstack([np.ones_like(med), np.full_like(low, -1)]))

        # Plot ROC curves
        fig, axs = plt.subplots(1, 3, figsize=(10,4))
        hm_roc = metrics.RocCurveDisplay.from_predictions(hm_labels, hm_scores, ax=axs[0])
        hl_roc = metrics.RocCurveDisplay.from_predictions(hl_labels, hl_scores, ax=axs[1])
        ml_roc = metrics.RocCurveDisplay.from_predictions(ml_labels, ml_scores, ax=axs[2])
        axs[0].set(xlabel=None)
        axs[0].set(ylabel=None)
        axs[0].legend().set_visible(False)
        axs[0].set_title('High vs. Medium')
        axs[1].set(xlabel=None)
        axs[1].set(ylabel=None)
        axs[1].legend().set_visible(False)
        axs[1].set_title('High vs. Low')
        axs[2].set(xlabel=None)
        axs[2].set(ylabel=None)
        axs[2].legend().set_visible(False)
        axs[2].set_title('Medium vs. Low')
        fig.suptitle('ROC Curves for {} Method'.format(method.capitalize()))
        fig.supxlabel('False Positive Rate (FPR)')
        fig.supylabel('True Positive Rate (TPR)')
        plt.savefig(os.path.join(roc_folder, '{}.png'.format(method)))
 
        # Compute KS distances between distributions
        data['H-M Dist'].append(stats.kstest(high, med, alternative='less').statistic)
        data['H-L Dist'].append(stats.kstest(high, low, alternative='less').statistic)
        data['M-L Dist'].append(stats.kstest(med, low, alternative='less').statistic)

        # Calculate AUROC and FPR@95TPR
        ood_metrics = OODMetrics()
        ood_metrics.update(-hm_scores, hm_labels)
        metric_dict = ood_metrics.compute()
        data['H-M AUROC'].append(metric_dict['AUROC'])
        data['H-M FPR'].append(metric_dict['FPR95TPR'])
        ood_metrics.update(-hl_scores, hl_labels)
        metric_dict = ood_metrics.compute()
        data['H-L AUROC'].append(metric_dict['AUROC'])
        data['H-L FPR'].append(metric_dict['FPR95TPR'])
        ood_metrics.update(-ml_scores, ml_labels)
        metric_dict = ood_metrics.compute()
        data['M-L AUROC'].append(metric_dict['AUROC'])
        data['M-L FPR'].append(metric_dict['FPR95TPR'])

    # Display results
    df = pd.DataFrame(data)
    print(os.path.join(results_folder, 'results.csv'))
    df.to_csv(os.path.join(results_folder, 'results.csv'))
    print(tabulate(df, headers='keys', tablefmt='psql', floatfmt=".4f"))
    # print(tabulate(df, headers='keys', tablefmt='latex', floatfmt=".2f"))


if __name__ == "__main__":
    main()
