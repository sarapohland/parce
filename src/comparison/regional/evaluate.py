import os
import json
import time
import torch
import pickle
import argparse
import configparser
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

from src.networks.model import NeuralNet
from src.datasets.setup_dataloader import setup_loader
from src.comparison.regional.methods import load_estimator

torch.manual_seed(0)

   
def main():
    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('method', type=str)
    parser.add_argument('--test_data', type=str, default='lunar')
    parser.add_argument('--model_dir', type=str, default='models/lunar/classify/')
    parser.add_argument('--decoder_dir', type=str, default='models/lunar/inpaint/')
    parser.add_argument('--data_file', type=str, default='results/lunar/regional/parce.csv')
    parser.add_argument('--estimator_file', type=str, default='None')
    args = parser.parse_args()

    # Load trained perception model
    with open(os.path.join(args.model_dir, 'layers.json')) as file:
        layer_args = json.load(file)
    model = NeuralNet(layer_args)
    model.load_state_dict(torch.load(os.path.join(args.model_dir, 'model.pth')))
    model.eval()

    # Load trained competency estimator
    estimator_file = None if args.estimator_file == 'None' else args.estimator_file
    estimator = load_estimator(args.method, model=model, model_dir=args.model_dir, 
                               decoder_dir=args.decoder_dir, test_data=args.test_data,
                               save_file=estimator_file)

    # Create data loaders
    id_test_loader = setup_loader(args.test_data, test=True, batch_size=1)
    ood_test_loader = setup_loader(args.test_data, ood=True, batch_size=1)

    # Create folder to dump score images
    folder = os.path.dirname(args.data_file)
    if not os.path.exists(folder):
        os.makedirs(folder)
    sub_folder = os.path.join(folder, Path(args.data_file).stem)
    if not os.path.exists(sub_folder):
        os.makedirs(sub_folder)

    # Set up dictionary to store results
    results = {'label': [], 'pred': [], 'ood': [], 'score': [], 'time': []}

    # Collect data from ID test set
    for batch, (X, y) in enumerate(id_test_loader):
        # Get prediction from perception model
        output = model(X)

        # Estimate competency score image
        start = time.time()
        score_img = estimator.map_scores(X, output.detach().numpy())[None,:,:]
        lapsed = time.time() - start

        # Dump score image
        filename = os.path.join(sub_folder, '{}.npz'.format(len(results['score'])))
        score_img.numpy().dump(filename)

        # Save results
        results['label'].append(y.detach().item())
        results['pred'].append(torch.argmax(output, dim=1).detach().item())
        results['ood'].append(0)
        results['score'].append(filename)
        results['time'].append(lapsed)

    # Load OOD segmentation labels
    segmentation = pickle.load(open(os.path.join(args.decoder_dir, 'ood_labels.p'), 'rb'))
    seg_labels = segmentation['labels']

    # Collect data from OOD test set
    for batch, (X, y) in enumerate(ood_test_loader):
        # Ignore data without labeled regions
        batch_size = len(X)
        these_labels = seg_labels[batch * batch_size : (batch + 1) * batch_size]
        if len(these_labels) < 1:
            break
        elif len(these_labels) < batch_size:
            X = X[:len(these_labels),:,:,:]
            y = y[:len(these_labels)]

        # Get prediction from perception model
        output = model(X)

        # Estimate competency score image
        start = time.time()
        score_img = estimator.map_scores(X, output.detach().numpy())[None,:,:]
        lapsed = time.time() - start
    
        # Dump score image
        filename = os.path.join(sub_folder, '{}.npz'.format(len(results['score'])))
        score_img.numpy().dump(filename)

        # Save results
        results['label'].append(y.detach().item())
        results['pred'].append(torch.argmax(output, dim=1).detach().item())
        results['ood'].append(1)
        results['score'].append(filename)
        results['time'].append(lapsed)

    # Save results to CSV file
    df = pd.DataFrame(results)
    df.to_csv(args.data_file, index=False)


if __name__ == "__main__":
    main()
