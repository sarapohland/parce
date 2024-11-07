import os
import json
import torch
import pickle
import argparse
import numpy as np

from tqdm import tqdm

from src.competency.parce import PARCE

def main():
    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('method', type=str)
    parser.add_argument('--train_data', type=str, default='lunar')
    parser.add_argument('--model_dir', type=str, default='models/lunar/classify/')
    parser.add_argument('--decoder_dir', type=str, default='models/lunar/reconstruct/')
    parser.add_argument('--use_gpu', action='store_true')
    args = parser.parse_args()

    # Fit competency estimator from training data
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.use_gpu else "cpu")
    estimator = PARCE(args.method, args.train_data, args.model_dir, args.decoder_dir, device)
    file = os.path.join(args.decoder_dir, 'parce.p')
    pickle.dump(estimator, open(file, 'wb'))

if __name__ == "__main__":
    main()
