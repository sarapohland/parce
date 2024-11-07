import os
import json
import shutil
import pickle
import argparse
import numpy as np
import configparser
from tqdm import tqdm

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt

from src.networks.model import NeuralNet
from src.datasets.setup_dataloader import setup_loader

def test(dataloader, model, decoder, device, method='reconstruct', architect='decoder'):
    inputs, labels, preds = [], [], []
    with torch.no_grad():
        for X, y in dataloader:
            # Load input data and label
            if method == 'reconstruct':
                input = X.to(device)
                true = X
            elif method == 'inpaint':
                input = X.to(device)
                true = y
            else:
                raise NotImplementedError('Unknown reconstruction method.')

            # Get prediction from image decoder
            if architect == 'decoder':
                z = model.get_feature_vector(input)
                pred = decoder(z.to(device))
            elif architect == 'autoencoder':
                pred = decoder(input)
            else:
                raise NotImplementedError('Unknown model architecture.')

            # Save input, label, and prediction
            inputs.append(input.cpu())
            labels.append(true)
            preds.append(pred.cpu())
    return torch.concatenate(inputs), torch.concatenate(labels), torch.concatenate(preds)

def main():
    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('method', type=str)
    parser.add_argument('--architecture', type=str, default='decoder')
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

    # Load trained decoder model
    with open(os.path.join(args.decoder_dir, 'layers.json')) as file:
        layer_args = json.load(file)
    decoder = NeuralNet(layer_args)
    decoder.load_state_dict(torch.load(os.path.join(args.decoder_dir, 'model.pth')))
    decoder.eval()

    # Get segmentation parameters
    if args.method == 'inpaint':
        config_file = os.path.join(args.decoder_dir, 'train.config')
        train_config = configparser.RawConfigParser()
        train_config.read(config_file)
        sigma = train_config.getfloat('segmentation', 'sigma')
        scale = train_config.getint('segmentation', 'scale')
        min_size = train_config.getint('segmentation', 'min_size')
        seg_params = {'sigma': sigma, 'scale': scale, 'min_size': min_size}
    else:
        seg_params = None

    # Set up dataset
    test_loader = setup_loader(args.test_data, batch_size=1, val=True, segment=seg_params)
    # test_loader = setup_loader(args.test_data, batch_size=1, ood=True, segment=seg_params)

    # Get model predictions for ID data
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.use_gpu else "cpu")
    inputs, labels, preds = test(test_loader, model.to(device), decoder.to(device), device, args.method, args.architecture)

    # Save reconstructed images
    output_dir = os.path.join(args.decoder_dir, 'reconstruction')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    loss_func = nn.MSELoss()
    for i, (input, label, pred) in enumerate(zip(inputs, labels, preds)):
        if i > 200:
            break 
        loss = loss_func(label, pred).item()

        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        label = np.float32(label.numpy())
        label = np.swapaxes(np.swapaxes(label, 0, 1), 1, 2)
        plt.imshow(label)
        plt.title('Original Image')

        plt.subplot(1, 3, 2)
        input = np.float32(input.numpy())
        input = np.swapaxes(np.swapaxes(input, 0, 1), 1, 2)
        plt.imshow(input)
        plt.title('Input/Masked Image')

        plt.subplot(1, 3, 3)
        pred = np.float32(pred.numpy())
        pred = np.swapaxes(np.swapaxes(pred, 0, 1), 1, 2)
        plt.imshow(pred)
        plt.title('Reconstructed Image')

        plt.suptitle('Reconstruction Loss: {}'.format(round(loss, 3)))
        file = os.path.join(output_dir, '{}.png'.format(i))
        # file = os.path.join(output_dir, '{}.png'.format(i+201))
        plt.savefig(file)
        plt.close()

if __name__ == "__main__":
    main()
