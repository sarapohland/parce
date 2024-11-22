import os
import pickle
import argparse
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from src.analysis.modify import get_default_factor
from src.analysis.modify import ALL_MODIFICATIONS, EVAL_MODIFICATIONS
from src.analysis.utils import load_results_by_property


def plot_acc_vs_factors(factors, accuracies, property):
    # Sort data by factor
    factors = np.array(factors)
    idxs = np.argsort(factors)
    factors = factors[idxs]
    accuracies = accuracies[idxs]

    # Plot prediction accuracy vs. factor
    fig = plt.figure(figsize=(8, 6))
    plt.plot(factors, accuracies, 'b.-', label='Test Accuracy')
    plt.axvline(x=get_default_factor(property), color='g', linestyle='--', label='Original Factor')
    plt.title('Prediction Accuracy vs. Image {}'.format(property.capitalize()), fontsize=16)
    plt.xlabel('{} Factor'.format(property.capitalize()), fontsize=14)
    plt.ylabel('Prediction Accuracy', fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.gca().set_ylim(0, 1)
    plt.legend(prop={'size': 14})
    return fig

def plot_comp_vs_factors(factors, competencies, property):
    # Sort data by factor
    factors = np.array(factors)
    idxs = np.argsort(factors)
    factors = factors[idxs]
    competencies = competencies[idxs]

    # Compute statistics for competency
    med_competencies = np.quantile(competencies, 0.5, axis=1)
    low_competencies = np.quantile(competencies, 0.25, axis=1)
    high_competencies = np.quantile(competencies, 0.75, axis=1)

    # Plot competency vs. factor
    fig = plt.figure(figsize=(8, 6))
    plt.plot(factors, med_competencies, 'b.-', label='Median Score')
    plt.fill_between(factors, low_competencies, high_competencies, color='blue', alpha=0.2, label='25-75% Range')
    plt.axvline(x=get_default_factor(property), color='g', linestyle='--', label='Original Factor')
    plt.title('Competency Score vs. Image {}'.format(property.capitalize()), fontsize=16)
    plt.xlabel('{} Factor'.format(property.capitalize()), fontsize=14)
    plt.ylabel('Competency Score', fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.gca().set_ylim(0, 1)
    plt.legend(prop={'size': 14})
    return fig

def plot_comp_vs_acc_bins(accuracies, competencies, bin_edges):
    # Bin the accuracy values
    num_bins = len(bin_edges) - 1
    bin_indices = np.digitize(accuracies, bins=bin_edges)

    # Collect competencies for each bin
    competency_bins = []
    for i in range(1, len(bin_edges)):
        # Find indices corresponding to the current accuracy bin
        bin_mask = (bin_indices == i)
        
        # Get competency scores for those accuracies
        if len(bin_mask) == 0:
            competency_bins.append([])
        else:
            bin_competencies = competencies[bin_mask]
            # print(i, np.shape(bin_competencies))
            competency_bins.append(bin_competencies.flatten())

    # Plot the distribution of competencies for each bin
    if num_bins == 3:
        fig = plt.figure(figsize=(8, 6))
        sns.boxplot(data=competency_bins, ax=plt.gca())
        plt.xticks(ticks=np.arange(0, num_bins), fontsize=14,
                labels=['High ({}-{})'.format(bin_edges[1], bin_edges[0]), 
                        'Med ({}-{})'.format(bin_edges[2], bin_edges[1]), 
                        'Low ({}-{})'.format(bin_edges[3], bin_edges[2])])
    else:
        fig = plt.figure(figsize=(10, 8))
        sns.boxplot(data=competency_bins, ax=plt.gca())
        plt.xticks(ticks=np.arange(0, num_bins), fontsize=14,
           labels=[f'{bin_edges[i+1]:.1f}-{bin_edges[i]:.1f}' for i in range(num_bins)])
    plt.xlabel('Prediction Accuracy', fontsize=14)
    plt.ylabel('Competency Score', fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(True)
    return fig

def plot_comp_distributions(labels, outputs, competencies):
    # Get scores for correctly classified and misclassified samples
    predictions = np.argmax(outputs, axis=-1)
    correct = competencies[predictions == labels]
    incorrect = competencies[predictions != labels]

    # Replace NaN values with minimum
    min_value = np.minimum(np.nanmin(correct), np.nanmin(incorrect))
    correct[np.isnan(correct)] = min_value
    incorrect[np.isnan(incorrect)] = min_value

    # Plot the distribution of competency scores
    fig = plt.figure(figsize=(8, 6))
    sns.boxplot(data=[correct, incorrect], ax=plt.gca())
    plt.gca().set_xticklabels(['Correctly Classified', 'Misclassified'])
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.ylabel('Competency Score', fontsize=14)
    return fig


def main():
    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('--property', type=str, default='all')
    parser.add_argument('--competency', type=str, default='parce')
    parser.add_argument('--test_data', type=str, default='lunar')
    parser.add_argument('--data_dir', type=str, default='results/lunar/competency/modified/data/')
    parser.add_argument('--plot_dir', type=str, default='results/lunar/competency/modified/plots/')
    args = parser.parse_args()

    # properties = ALL_MODIFICATIONS if args.property == 'all' else [args.property]
    properties = EVAL_MODIFICATIONS if args.property == 'all' else [args.property]

    # Set accuracy bin edges
    if args.test_data == 'pavilion':
        bin_edges = np.array([1.0, 0.85, 0.5, 0.0])
    else:
        bin_edges = np.array([1.0, 0.9, 0.5, 0.0])

    # Generate plots for specified properties
    labels, outputs, accuracies, competencies = [], [], [], []
    for property in properties:
        # Create folders to save figures
        plot_folder = os.path.join(args.plot_dir, property)
        if not os.path.exists(plot_folder):
            os.makedirs(plot_folder)
        
        # Load saved data for given property
        data_folder = os.path.join(args.data_dir, property)
        factors, datas = load_results_by_property(data_folder)
        label = np.array([data['labels'] for data in datas])
        output = np.array([data['outputs'] for data in datas])
        accuracy = np.array([data['accuracy'] for data in datas])
        competency = np.array([data[args.competency] for data in datas])

        # Plot accuracy vs. factor for given image property
        plot_acc_vs_factors(factors, accuracy, property)
        file = os.path.join(plot_folder, 'acc_vs_factor.png')
        plt.savefig(file)
        plt.close()

        # Plot competency vs. accuracy for given image property
        plot_comp_vs_factors(factors, competency, property)
        file = os.path.join(plot_folder, 'comp_vs_factor.png')
        plt.savefig(file)
        plt.close()

        # Plot competency vs. accuracy for given image property
        plot_comp_vs_acc_bins(accuracy, competency, bin_edges)
        plt.title('Competency vs. Accuracy for Image {}'.format(property.capitalize()), fontsize=16)
        file = os.path.join(plot_folder, 'comp_vs_acc.png')
        plt.savefig(file)
        plt.close()

        # Plot competency distributions for given image property
        plot_comp_distributions(label, output, competency)
        plt.title('Competency Estimates for Image {}'.format(property.capitalize()), fontsize=16)
        file = os.path.join(plot_folder, 'comp_distr.png')
        plt.savefig(file)
        plt.close()

        # Save all accuracy and competency data
        labels.append(label)
        outputs.append(output)
        accuracies.append(accuracy)
        competencies.append(competency)
    labels = np.vstack(labels)
    outputs = np.vstack(outputs)
    accuracies = np.hstack(accuracies)
    competencies = np.vstack(competencies)

    if len(properties) > 1:
        # Create folder to save figures
        plot_folder = os.path.join(args.plot_dir, 'all')
        if not os.path.exists(plot_folder):
            os.makedirs(plot_folder)
        
        # Generate competency vs. accuracy plot across properties
        plot_comp_vs_acc_bins(accuracies, competencies, bin_edges)
        plt.title('Competency vs. Accuracy Across Image Properties', fontsize=16)
        file = os.path.join(plot_folder, 'comp_vs_acc.png')
        plt.savefig(file)
        plt.close()

        # Generate competency vs. accuracy plot with 10 bins
        bin_edges = np.linspace(1.0, 0.0, 11)
        plot_comp_vs_acc_bins(accuracies, competencies, bin_edges)
        plt.title('Competency vs. Accuracy Across Image Properties', fontsize=16)
        file = os.path.join(plot_folder, 'comp_vs_acc-10.png')
        plt.savefig(file)
        plt.close()

        # Generate competency distribution plot across properties
        plot_comp_distributions(labels, outputs, competencies)
        plt.title('Competency Estimates Across Image Properties', fontsize=16)
        file = os.path.join(plot_folder, 'comp_distr.png')
        plt.savefig(file)
        plt.close()


if __name__=="__main__":
    main()