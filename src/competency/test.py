import os
import json
import torch
import pickle
import string
import argparse
import torchvision
import configparser
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from src.networks.model import NeuralNet
from src.datasets.setup_dataloader import setup_loader

torch.manual_seed(0)
   
# Plot reconstruction loss distribution(s)
def plot_distributions(losses, labels, file):
    fig, ax = plt.subplots()
    for loss, label in zip(losses, labels):
        sns.kdeplot(data=loss, ax=ax, label=label)
    ax.legend()
    plt.xlabel('Reconstruction Loss')
    plt.ylabel('Probability Density')
    plt.title('Reconstruction Loss Distributions')
    plt.savefig(file)
    plt.close()

# Plot probabilistic competency estimates
def plot_competency(scores, labels, file):
    # fig, ax = plt.subplots()
    # filtered_score = []
    # for these_scores in scores:
    #     filtered_score.append(these_scores[~np.isnan(these_scores)])
    # plt.boxplot(filtered_score, labels=labels)
    # plt.title('Probabilistic Competency Estimates')
    # fig.savefig(file)

    filtered_scores = []
    for these_scores in scores:
        filtered_scores.append(these_scores[~np.isnan(these_scores)])
    fig, ax = plt.subplots()
    sns.boxplot(data=filtered_scores, ax=ax)
    # sns.stripplot(data=filtered_scores, alpha=0.4, ax=ax)
    ax.set_xticklabels(labels)
    plt.title('Probabilistic Competency Estimates')
    plt.savefig(file)
    plt.close()
    
def main():
    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('method', type=str)
    parser.add_argument('--test_data', type=str, default='lunar')
    parser.add_argument('--model_dir', type=str, default='models/lunar/classify/')
    parser.add_argument('--decoder_dir', type=str, default='models/lunar/reconstruct/')
    parser.add_argument('--use_gpu', action='store_true')
    args = parser.parse_args()

    # Load trained perception model
    with open(os.path.join(args.model_dir, 'layers.json')) as file:
        layer_args = json.load(file)
    model = NeuralNet(layer_args)
    model.load_state_dict(torch.load(os.path.join(args.model_dir, 'model.pth')))
    model.eval()

    # Load trained competency estimator
    file = os.path.join(args.decoder_dir, 'parce.p')
    estimator = pickle.load(open(file, 'rb'))
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.use_gpu else "cpu")
    estimator.set_device(device)

    # Create data loaders
    id_test_loader = setup_loader(args.test_data, test=True)
    ood_test_loader = setup_loader(args.test_data, ood=True)

    # Overall competency estimation with reconstruction model
    if args.method == 'overall':
        # Collect data from ID test set
        correct_losses, correct_scores = [], []
        incorrect_losses, incorrect_scores = [], []
        for X, y in id_test_loader:
            # Get prediction from perception model
            output = model(X)
            preds = torch.argmax(output, dim=1).detach().numpy()
            trues = torch.flatten(y).detach().numpy()

            # Compute reconstruction losses and estimate competency scores
            output = output.detach().numpy()
            scores, losses = estimator.comp_scores_and_losses(X, output)
            correct_losses.append(losses[preds == trues])
            correct_scores.append(scores[preds == trues])
            incorrect_losses.append(losses[[not x for x in (preds == trues)]])
            incorrect_scores.append(scores[[not x for x in (preds == trues)]])

        correct_losses, correct_scores = np.hstack(correct_losses), np.hstack(correct_scores)
        incorrect_losses, incorrect_scores = np.hstack(incorrect_losses), np.hstack(incorrect_scores)

        # Collect data from OOD test set
        ood_losses, ood_scores = [], []
        for X, y in ood_test_loader:
            # Get prediction from perception model
            output = model(X)
            output = output.detach().numpy()

            # Compute reconstruction losses and estimate competency scores
            scores, losses = estimator.comp_scores_and_losses(X, output)
            ood_losses.append(losses)
            ood_scores.append(scores.flatten())

        ood_losses, ood_scores = np.hstack(ood_losses), np.hstack(ood_scores)

        # Generate plots of loss distributions
        all_losses = [correct_losses, incorrect_losses, ood_losses]
        all_labels = ['Correctly Classified', 'Misclassified', 'Out-of-Distribution']
        file = os.path.join(args.decoder_dir, 'test_distributions.png')
        plot_distributions(all_losses, all_labels, file)

        # Generate plots of competency estimates
        all_scores = [correct_scores, incorrect_scores, ood_scores]
        all_labels = ['Correctly Classified', 'Misclassified', 'Out-of-Distribution']
        file = os.path.join(args.decoder_dir, 'test_competency_estimates.png')
        plot_competency(all_scores, all_labels, file)

    # Regional competency estimation with inpainting model
    elif args.method == 'regional':
        # Collect data from ID test set
        id_losses, id_scores = [], []
        for X, y in id_test_loader:
            # Get prediction from perception model
            output = model(X)
            output = output.detach().numpy()

            # Compute reconstruction losses and estimate competency scores
            scores, losses = estimator.comp_scores_and_losses(X, output)
            id_losses.append(losses)
            id_scores.append(scores.flatten())

        id_losses, id_scores = np.hstack(id_losses), np.hstack(id_scores)

        # Load OOD segmentation labels
        segmentation = pickle.load(open(os.path.join(args.decoder_dir, 'ood_labels.p'), 'rb'))
        seg_labels = segmentation['labels']

        # Collect data from OOD test set
        familiar_losses, familiar_scores = [], []
        unfamiliar_losses, unfamiliar_scores = [], []
        for batch, (X, y) in enumerate(ood_test_loader):

            # Grab segment labels for this batch
            batch_size = len(X)
            if batch == 0:
                default_batch_size = batch_size
            these_labels = seg_labels[batch * default_batch_size : (batch + 1) * batch_size]

            # Ignore data without labeled regions
            if len(these_labels) < 1:
                break
            elif len(these_labels) < batch_size:
                X = X[:len(these_labels),:,:,:]
                y = y[:len(these_labels)]
            these_labels = [x for xs in these_labels for x in xs]

            # Get prediction from perception model
            output = model(X)
            output = output.detach().numpy()

            # Compute reconstruction losses and estimate competency scores
            scores, losses = estimator.comp_scores_and_losses(X, output)

            # Separate data into familiar and unfamiliar regions
            familiar_losses.append(losses[[(label == 0) for label in these_labels]])
            familiar_scores.append(scores[[(label == 0) for label in these_labels]])
            unfamiliar_losses.append(losses[[(label == 1) for label in these_labels]])
            unfamiliar_scores.append(scores[[(label == 1) for label in these_labels]])

        familiar_losses, familiar_scores = np.hstack(familiar_losses), np.hstack(familiar_scores)
        unfamiliar_losses, unfamiliar_scores = np.hstack(unfamiliar_losses), np.hstack(unfamiliar_scores)

        # Generate plots of loss distributions
        all_losses = [id_losses, familiar_losses, unfamiliar_losses]
        all_labels = ['ID Regions', 'Familiar OOD Regions', 'Unfamiliar OOD Regions']
        file = os.path.join(args.decoder_dir, 'test_distributions.png')
        plot_distributions(all_losses, all_labels, file)

        # Generate plots of competency estimates
        all_scores = [id_scores, familiar_scores, unfamiliar_scores]
        all_labels = ['ID Regions', 'Familiar OOD Regions', 'Unfamiliar OOD Regions']
        file = os.path.join(args.decoder_dir, 'competency_estimates.png')
        plot_competency(all_scores, all_labels, file)

    else:
        raise NotImplementedError('Unknown competency estimation method.')


if __name__ == "__main__":
    main()
