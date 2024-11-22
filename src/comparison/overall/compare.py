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

def main():
    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('--data_dir', type=str, default='results/lunar/competency/unmodified/data/')
    parser.add_argument('--plot_dir', type=str, default='results/lunar/competency/unmodified/plots/')
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
    data = {'Method': [], 'Computation Time': [],
            'C-M Dist': [], 'C-M AUROC': [], 'C-M FPR': [],
            'C-O Dist': [], 'C-O AUROC': [], 'C-O FPR': [],
            'M-O Dist': [], 'M-O AUROC': [], 'M-O FPR': [],
            }
    
    # Try to read each file in data directory
    for file in os.listdir(args.data_dir):
        if not file.endswith('.csv'):
            continue
        
        # Read results
        filename = Path(file).stem
        data['Method'].append(filename)
        df = pd.read_csv(os.path.join(args.data_dir, file))
        
        # Get average computation time
        data['Computation Time'].append(df['time'].mean())

        # Get scores of sets of samples
        id_df = df[df['ood'] == 0]
        correct = id_df[id_df['pred'] == id_df['label']]['score'].to_numpy()
        incorrect = id_df[id_df['pred'] != id_df['label']]['score'].to_numpy()
        ood = df[df['ood'] == 1]['score'].to_numpy()

        # Replace NaN values with minimum
        min_value = np.minimum(np.minimum(np.nanmin(correct), np.nanmin(incorrect)), np.nanmin(ood))
        correct[np.isnan(correct)] = min_value
        incorrect[np.isnan(incorrect)] = min_value
        ood[np.isnan(ood)] = min_value

        print(np.shape(correct), np.shape(incorrect), np.shape(ood))

        # Plot score distributions
        fig, ax = plt.subplots()
        sns.boxplot(data=[correct, incorrect, ood], ax=ax)
        ax.set_xticklabels(['Correctly Classified', 'Misclassified', 'Out-of-Distribution'])
        plt.title('{} Competency Estimates'.format(filename.capitalize()))
        plt.ylabel('Competency Score')
        plt.savefig(os.path.join(distr_folder, '{}.png'.format(filename)))

        # Compare each set of samples
        cm_scores = torch.from_numpy(np.hstack([correct, incorrect]))
        co_scores = torch.from_numpy(np.hstack([correct, ood]))
        mo_scores = torch.from_numpy(np.hstack([incorrect, ood]))
        cm_labels = torch.from_numpy(np.hstack([np.ones_like(correct), np.full_like(incorrect, -1)]))
        co_labels = torch.from_numpy(np.hstack([np.ones_like(correct), np.full_like(ood, -1)]))
        mo_labels = torch.from_numpy(np.hstack([np.ones_like(incorrect), np.full_like(ood, -1)]))

        # Plot ROC curves
        fig, axs = plt.subplots(1, 3, figsize=(10,4))
        cm_roc = metrics.RocCurveDisplay.from_predictions(cm_labels, cm_scores, ax=axs[0])
        co_roc = metrics.RocCurveDisplay.from_predictions(co_labels, co_scores, ax=axs[1])
        mo_roc = metrics.RocCurveDisplay.from_predictions(mo_labels, mo_scores, ax=axs[2])
        axs[0].set(xlabel=None)
        axs[0].set(ylabel=None)
        axs[0].legend().set_visible(False)
        axs[0].set_title('Correct vs. Incorrect')
        axs[1].set(xlabel=None)
        axs[1].set(ylabel=None)
        axs[1].legend().set_visible(False)
        axs[1].set_title('Correct vs. OOD')
        axs[2].set(xlabel=None)
        axs[2].set(ylabel=None)
        axs[2].legend().set_visible(False)
        axs[2].set_title('Incorrect vs. OOD')
        fig.suptitle('ROC Curves for {} Method'.format(filename.capitalize()))
        fig.supxlabel('False Positive Rate (FPR)')
        fig.supylabel('True Positive Rate (TPR)')
        plt.savefig(os.path.join(roc_folder, '{}.png'.format(filename)))
 
        # Compute KS distances between distributions
        data['C-M Dist'].append(stats.kstest(correct, incorrect, alternative='less').statistic)
        data['C-O Dist'].append(stats.kstest(correct, ood, alternative='less').statistic)
        data['M-O Dist'].append(stats.kstest(incorrect, ood, alternative='less').statistic)

        # Calculate AUROC and FPR@95TPR
        ood_metrics = OODMetrics()
        ood_metrics.update(-cm_scores, cm_labels)
        metric_dict = ood_metrics.compute()
        data['C-M AUROC'].append(metric_dict['AUROC'])
        data['C-M FPR'].append(metric_dict['FPR95TPR'])
        ood_metrics.update(-co_scores, co_labels)
        metric_dict = ood_metrics.compute()
        data['C-O AUROC'].append(metric_dict['AUROC'])
        data['C-O FPR'].append(metric_dict['FPR95TPR'])
        ood_metrics.update(-mo_scores, mo_labels)
        metric_dict = ood_metrics.compute()
        data['M-O AUROC'].append(metric_dict['AUROC'])
        data['M-O FPR'].append(metric_dict['FPR95TPR'])

    # Display results
    df = pd.DataFrame(data)
    df.to_csv(os.path.join(results_folder, 'results.csv'))
    print(tabulate(df, headers='keys', tablefmt='psql', floatfmt=".4f"))
    # print(tabulate(df, headers='keys', tablefmt='latex', floatfmt=".2f"))


if __name__ == "__main__":
    main()
