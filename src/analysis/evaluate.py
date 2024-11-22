import os
import json
import torch
import pickle
import argparse
import numpy as np

from src.networks.model import NeuralNet
from src.datasets.setup_dataloader import setup_loader
from src.comparison.overall.methods import load_estimator
from src.comparison.overall.methods import ALL_OVERALL

import warnings
warnings.filterwarnings("ignore")

def get_accurary(true, pred):
    num_correct = np.sum(pred == true)
    accuracy = num_correct / len(true)
    return accuracy

def main():
    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('--test_data', type=str, default='lunar')
    parser.add_argument('--model_dir', type=str, default='models/lunar/classify/')
    parser.add_argument('--decoder_dir', type=str, default='models/lunar/reconstruct/')
    parser.add_argument('--property', type=str, default='saturation')
    parser.add_argument('--factor', type=float, default='1.0')
    parser.add_argument('--data_dir', type=str, default='results/lunar/competency/modified/data/')
    args = parser.parse_args()

    # Load trained model
    with open(os.path.join(args.model_dir, 'layers.json')) as file:
        layer_args = json.load(file)
    model = NeuralNet(layer_args)
    model.load_state_dict(torch.load(os.path.join(args.model_dir, 'model.pth')))
    model.eval()

    # Load trained competency estimators
    estimators = [load_estimator(method, model=model, model_dir=args.model_dir, 
                               decoder_dir=args.decoder_dir, test_data=args.test_data) 
                               for method in ALL_OVERALL]
    
    # Create data loader
    params = {'property': args.property, 'factor': args.factor}
    test_loader = setup_loader(args.test_data, test=True, modify=params)

    # Create dictionary to store data
    all_data = {key: [] for key in ALL_OVERALL}
    all_data['labels'] = []
    all_data['outputs'] = []
    all_data['accuracy'] = None

    # Process input data in batches
    for X, y in test_loader:
        # Store labels of input images
        labels = torch.flatten(y).numpy()
        all_data['labels'].append(labels)

        # Get output from perception model
        output = model(X).detach().numpy()
        all_data['outputs'].append(output)

        # Estimate competency scores
        for estimator, key in zip(estimators, ALL_OVERALL):
            score = estimator.comp_scores(X, output)
            all_data[key].append(score)

    # Collect all output data
    all_data['labels'] = np.hstack(all_data['labels'])
    all_data['outputs'] = np.vstack(all_data['outputs'])
    for key in ALL_OVERALL:
        all_data[key] = np.hstack(all_data[key])

    # Compute model accuracy
    all_preds = np.argmax(all_data['outputs'], axis=1)
    all_data['accuracy'] = get_accurary(all_data['labels'], all_preds)
    print('Image {}: {}   Accuracy: {}'.format(args.property, args.factor, all_data['accuracy']))

    # Save data outputs
    output_dir = os.path.join(args.data_dir, args.property)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    file = os.path.join(output_dir, '{}.p'.format(str(args.factor)))
    pickle.dump(all_data, open(file, 'wb'))
    
if __name__=="__main__":
    main()