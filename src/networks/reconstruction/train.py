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

def train(dataloader, model, decoder, criterion, optimizer, device, method='reconstruct', architect='decoder'):
    model.eval()
    decoder.train()
    train_loss = 0
    for X, y in dataloader:
        # Load input data and label
        if method == 'reconstruct':
            input = X.to(device)
            true = X.to(device)
        elif method == 'inpaint':
            input = X.to(device)
            true = y.to(device)
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

        # Compute reconstruction loss
        loss = criterion(pred, true)
        train_loss += loss.item()

        # Update decoder model
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Compute average loss
    avg_loss = train_loss / len(dataloader)
    return avg_loss

def test(dataloader, model, decoder, criterion, device, method='reconstruct', architect='decoder'):
    model.eval()
    decoder.eval()
    test_loss = 0
    with torch.no_grad():
        for X, y in dataloader:
            # Load input data and label
            if method == 'reconstruct':
                input = X.to(device)
                true = X.to(device)
            elif method == 'inpaint':
                input = X.to(device)
                true = y.to(device)
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

            # Compute reconstruction loss
            loss = criterion(pred, true)
            test_loss += loss.item()

    # Compute average loss
    avg_loss = test_loss / len(dataloader)
    return avg_loss

def main():
    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('method', type=str)
    parser.add_argument('--architecture', type=str, default='decoder')
    parser.add_argument('--train_data', type=str, default='lunar')
    parser.add_argument('--model_dir', type=str, default='models/lunar/classify/')
    parser.add_argument('--output_dir', type=str, default='models/lunar/reconstruct/')
    parser.add_argument('--train_config', type=str, default='src/networks/reconstruction/train.config')
    parser.add_argument('--network_file', type=str, default='src/networks/reconstruction/layers.json')
    parser.add_argument('--init_model', type=str, default=None)
    parser.add_argument('--use_gpu', action='store_true')
    args = parser.parse_args()

    # Configure paths
    make_new_dir = True
    if os.path.exists(args.output_dir):
        key = input('Output directory already exists! Overwrite the folder? (y/n)')
        if key == 'y':
            shutil.rmtree(args.output_dir)
        else:
            make_new_dir = False
            args.train_config = os.path.join(args.output_dir, os.path.basename(args.train_config))
            args.network_file = os.path.join(args.output_dir, os.path.basename(args.network_file))
    if make_new_dir:
        os.makedirs(args.output_dir)
        shutil.copy(args.train_config, os.path.join(args.output_dir, 'train.config'))
        shutil.copy(args.network_file, os.path.join(args.output_dir, 'layers.json'))
    model_file = os.path.join(args.output_dir, 'model.pth')
    loss_file  = os.path.join(args.output_dir, 'loss.png')

    # Read training parameters
    if args.train_config is None:
        parser.error('Train config file has to be specified.')
    train_config = configparser.RawConfigParser()
    train_config.read(args.train_config)
    optimizer = train_config.get('optimizer', 'optimizer')
    learning_rate = train_config.getfloat('optimizer', 'learning_rate')
    momentum = train_config.getfloat('optimizer', 'momentum')
    betas = [float(x) for x in train_config.get('optimizer', 'betas').split(', ')]
    epsilon = train_config.getfloat('optimizer', 'epsilon')
    weight_decay = train_config.getfloat('optimizer', 'weight_decay')
    loss_func = train_config.get('training', 'loss')
    batch_size_train = train_config.getint('training', 'batch_size_train')
    batch_size_test = train_config.getint('training', 'batch_size_test')
    epochs = train_config.getint('training', 'epochs')

    # Read segmentation parameters
    if args.method == 'inpaint':
        sigma = train_config.getfloat('segmentation', 'sigma')
        scale = train_config.getint('segmentation', 'scale')
        min_size = train_config.getint('segmentation', 'min_size')
        seg_params = {'sigma': sigma, 'scale': scale, 'min_size': min_size}
    else:
        seg_params = None

    # Load trained perception model
    with open(os.path.join(args.model_dir, 'layers.json')) as file:
        layer_args = json.load(file)
    model = NeuralNet(layer_args)
    model.load_state_dict(torch.load(os.path.join(args.model_dir, 'model.pth')))

    # Build the decoder network
    with open(args.network_file) as file:
        layer_args = json.load(file)
    decoder = NeuralNet(layer_args)

    # Initialize decoder model
    if args.init_model is not None:
        decoder.load_state_dict(torch.load(os.path.join(args.init_model, 'model.pth')))

    # Set up dataset
    train_loader = setup_loader(args.train_data, batch_size=batch_size_train, train=True, segment=seg_params)
    test_loader = setup_loader(args.train_data, batch_size=batch_size_test, val=True, segment=seg_params)

    # Define loss function
    if loss_func == 'mse':
        criterion = nn.MSELoss()
    else:
        raise ValueError('Unknown loss function! Currently only mse is implemented for reconstruction.')

    # Define optimizer
    if optimizer == 'sgd':
        optimizer = torch.optim.SGD(decoder.parameters(), lr=learning_rate,
                        momentum=momentum, weight_decay=weight_decay)
    elif optimizer == 'adam':
        optimizer = torch.optim.Adam(decoder.parameters(), lr=learning_rate,
                        betas=betas, eps=epsilon, weight_decay=weight_decay)
    else:
        raise ValueError('Unknown optimizer! Please choose sgd or adam.')

    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.use_gpu else "cpu")
    model.to(device)
    decoder.to(device)

    # Run training process over several epochs
    train_loss, test_loss = [], []
    for t in tqdm(range(epochs)):
        # Run training step
        train_loss += [train(train_loader, model, decoder, criterion, optimizer, device, args.method, args.architecture)]
        print('Epoch {} Training loss: {}'.format(t, train_loss[-1]))
        
        # Run evaluation step
        test_loss  += [test(test_loader, model, decoder, criterion, device, args.method, args.architecture)]
        print('Epoch {} Evaluation loss: {}'.format(t, test_loss[-1]))

        # Save trained model
        torch.save(decoder.state_dict(), model_file)

        # Save plot of loss over epochs
        plt.plot(train_loss, '-b', label='Training')
        plt.plot(test_loss, '-r', label='Evaluation')
        plt.legend(loc="upper right")
        plt.xlabel('Epoch')
        plt.ylabel('Average Loss Across Batches')
        plt.title('Average Training and Evaluation Loss')
        plt.savefig(loss_file)
        plt.close()

if __name__ == "__main__":
    main()