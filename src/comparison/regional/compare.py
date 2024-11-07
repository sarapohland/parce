import os
import torch
import pickle
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from pathlib import Path
from sklearn import metrics
from ast import literal_eval
from tabulate import tabulate
import matplotlib.pyplot as plt
from pytorch_ood.utils import OODMetrics

from src.datasets.setup_dataloader import setup_loader

def main():
    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('--decoder_dir', type=str, default='models/lunar/inpaint/')
    parser.add_argument('--data_dir', type=str, default='results/lunar/competency/regional/data/')
    parser.add_argument('--plot_dir', type=str, default='results/lunar/competency/regional/plots/')
    parser.add_argument('--height', type=int, default=240)
    parser.add_argument('--width', type=int, default=320)
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
            'I-U Dist': [], 'I-U AUROC': [], 'I-U FPR': [],
            'F-U Dist': [], 'F-U AUROC': [], 'F-U FPR': [],
            }
    
    # Load OOD segmentation labels
    segmentation = pickle.load(open(os.path.join(args.decoder_dir, 'ood_labels.p'), 'rb'))
    seg_labels = segmentation['labels']
    seg_pixels = segmentation['pixels']

    # Create images from true labels of OOD set
    label_imgs = []
    for these_pixels, these_labels in zip(seg_pixels, seg_labels):
        label_img = torch.zeros((1, args.height, args.width))
        for pixels, label in zip(these_pixels, these_labels):
            label_img[:, pixels[0, :], pixels[1, :]] = label
        label_imgs.append(label_img)

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
        id_df  = df.loc[df['ood'] == 0]
        ood_df = df.loc[df['ood'] == 1]

        id, familiar, unfamiliar = [], [], []
        for index, row in id_df.iterrows():
            id += list(np.load(row['score'], allow_pickle=True).flatten())
        for count, (index, row) in enumerate(ood_df.iterrows()):
            label_img = label_imgs[count]
            scores = np.load(row['score'], allow_pickle=True)
            if filename == 'rkde':
                scores = scores[0,:,:,:]
            familiar += list(scores[label_img == 0])
            unfamiliar += list(scores[label_img == 1])

        # Plot score distributions
        fig, ax = plt.subplots()
        sns.boxplot(data=[id, familiar, unfamiliar], ax=ax)
        ax.set_xticklabels(['ID Regions', 'Familiar OOD Regions', 'Unfamiliar OOD Regions'])
        plt.title('{} Competency Estimates'.format(filename.capitalize()))
        plt.ylabel('Competency Score')
        plt.savefig(os.path.join(distr_folder, '{}.png'.format(filename)))

        # Compare each set of samples
        iu_scores = torch.from_numpy(np.hstack([id, unfamiliar]))
        fu_scores = torch.from_numpy(np.hstack([familiar, unfamiliar]))
        iu_labels = torch.from_numpy(np.hstack([np.ones_like(id), np.full_like(unfamiliar, -1)]))
        fu_labels = torch.from_numpy(np.hstack([np.ones_like(familiar), np.full_like(unfamiliar, -1)]))

        # Plot ROC curves
        fig, axs = plt.subplots(1, 2, figsize=(8,4))
        cm_roc = metrics.RocCurveDisplay.from_predictions(iu_labels, iu_scores, ax=axs[0])
        co_roc = metrics.RocCurveDisplay.from_predictions(fu_labels, fu_scores, ax=axs[1])
        axs[0].set(xlabel=None)
        axs[0].set(ylabel=None)
        axs[0].legend().set_visible(False)
        axs[0].set_title('ID vs. Unfamiliar OOD')
        axs[1].set(xlabel=None)
        axs[1].set(ylabel=None)
        axs[1].legend().set_visible(False)
        axs[1].set_title('Familiar OOD vs. Unfamiliar OOD')
        fig.suptitle('ROC Curves for {} Method'.format(filename.capitalize()))
        fig.supxlabel('False Positive Rate (FPR)')
        fig.supylabel('True Positive Rate (TPR)')
        plt.savefig(os.path.join(roc_folder, '{}.png'.format(filename)))
 
        # Compute KS distances between distributions
        data['I-U Dist'].append(stats.kstest(id, unfamiliar, alternative='less').statistic)
        data['F-U Dist'].append(stats.kstest(familiar, unfamiliar, alternative='less').statistic)

        # Calculate AUROC and FPR@95TPR
        ood_metrics = OODMetrics()
        ood_metrics.update(-iu_scores, iu_labels)
        metric_dict = ood_metrics.compute()
        data['I-U AUROC'].append(metric_dict['AUROC'])
        data['I-U FPR'].append(metric_dict['FPR95TPR'])
        ood_metrics.update(-fu_scores, fu_labels)
        metric_dict = ood_metrics.compute()
        data['F-U AUROC'].append(metric_dict['AUROC'])
        data['F-U FPR'].append(metric_dict['FPR95TPR'])

    # Display results
    df = pd.DataFrame(data)
    df.to_csv(os.path.join(results_folder, 'results.csv'))
    print(tabulate(df, headers='keys', tablefmt='psql', floatfmt=".4f"))
    # print(tabulate(df, headers='keys', tablefmt='latex', floatfmt=".3f"))


if __name__ == "__main__":
    main()
