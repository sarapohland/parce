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
    parser.add_argument('data_dir', type=str)
    parser.add_argument('--test_data', type=str, default='lunar')
    parser.add_argument('--save_dir', type=str, default='results/regional/visualize/')
    parser.add_argument('--example', type=int, default=0)
    args = parser.parse_args()

    # Create folder to save figures
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
        
    # Save input image
    ood_test_loader = setup_loader(args.test_data, ood=True, batch_size=1)
    for batch, (X, y) in enumerate(ood_test_loader):
        if not batch == args.example:
            continue
        input = np.squeeze(X.numpy() * 255).astype(np.uint8)
        input = np.swapaxes(np.swapaxes(input, 0, 1), 1, 2)
        plt.figure()
        plt.imshow(input)
        plt.tight_layout(pad=0)
        plt.savefig(os.path.join(args.save_dir, '{}.png'.format(batch)))
        plt.close()

    for file in os.listdir(args.data_dir):
        if not file.endswith('.csv'):
            continue
        
        # Read results
        filename = Path(file).stem
        df = pd.read_csv(os.path.join(args.data_dir, file))
        id_df  = df.loc[df['ood'] == 0]
        ood_df = df.loc[df['ood'] == 1]

        # Save competency maps
        for count, (index, row) in enumerate(ood_df.iterrows()):
            if not count == args.example:
                continue
            scores = np.load(row['score'], allow_pickle=True)
            height, width = np.shape(scores)[-2], np.shape(scores)[-1]
            scores = np.reshape(scores, (height, width))
            plt.figure()
            plt.imshow(scores, cmap='coolwarm_r')
            # plt.colorbar()
            plt.colorbar(shrink=0.8)
            plt.tight_layout(pad=0)
            plt.savefig(os.path.join(args.save_dir, '{}_{}.png'.format(count, filename)))
            plt.close()

if __name__ == "__main__":
    main()
