import os
import json
import torch
import pickle
import string
import argparse
import torchvision
import numpy as np
import matplotlib.pyplot as plt

from src.networks.model import NeuralNet
from src.datasets.setup_dataloader import setup_loader

torch.manual_seed(0)

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

    # Overall competency estimation with reconstruction model
    if args.method == 'overall':
        # Create data loaders
        id_test_loader = setup_loader(args.test_data, test=True)
        ood_test_loader = setup_loader(args.test_data, ood=True)

        # Collect data from ID test set
        correct_imgs, incorrect_imgs = [], []
        correct_scores, incorrect_scores = [], []
        for X, y in id_test_loader:
            # Get prediction from perception model
            output = model(X)
            preds = torch.argmax(output, dim=1).detach().numpy()
            trues = torch.flatten(y).detach().numpy()

            # Estimate competency scores
            output = output.detach().numpy()
            scores = estimator.comp_scores(X, output)
            correct_scores.append(scores[preds == trues])
            incorrect_scores.append(scores[[not x for x in (preds == trues)]])

            # Save images for visualization
            correct_imgs.append(X[preds == trues, :, :, :])
            incorrect_imgs.append(X[[not x for x in (preds == trues)], :, :, :])

        correct_imgs, incorrect_imgs = torch.vstack(correct_imgs), torch.vstack(incorrect_imgs)
        correct_scores, incorrect_scores = np.hstack(correct_scores), np.hstack(incorrect_scores)

        # Collect data from OOD test set
        ood_imgs = []
        ood_scores = []
        for X, y in ood_test_loader:
            # Get prediction from perception model
            output = model(X)
            output = output.detach().numpy()

            # Estimate competency scores
            scores = estimator.comp_scores(X, output)
            ood_scores.append(scores.flatten())

            # Save images for visualization
            ood_imgs.append(X)

        ood_imgs = torch.vstack(ood_imgs)
        ood_scores = np.hstack(ood_scores)

        # Create folders to save results
        output_dir = os.path.join(args.decoder_dir, 'competency')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        correct_dir = os.path.join(output_dir, 'correct')
        if not os.path.exists(correct_dir):
            os.makedirs(correct_dir)
        incorrect_dir = os.path.join(output_dir, 'incorrect')
        if not os.path.exists(incorrect_dir):
            os.makedirs(incorrect_dir)
        ood_dir = os.path.join(output_dir, 'ood')
        if not os.path.exists(ood_dir):
            os.makedirs(ood_dir)

        # Visualize images and competency estimates
        output_dirs = [ood_dir, correct_dir, incorrect_dir]
        img_tensors = [ood_imgs, correct_imgs, incorrect_imgs]
        score_arrs  = [ood_scores, correct_scores, incorrect_scores]
        for output_dir, imgs, scores in zip(output_dirs, img_tensors, score_arrs):
            for idx, (img, score) in enumerate(zip(imgs, scores)):
                img = np.squeeze(img.numpy() * 255).astype(np.uint8)
                img = np.swapaxes(np.swapaxes(img, 0, 1), 1, 2)
                plt.imshow(img)
                plt.title('Probabilistic Competency Estimate: {}'.format(round(score, 4)))
                plt.savefig(os.path.join(output_dir, '{}.png'.format(idx)))
                plt.close()

    # Regional competency estimation with inpainting model
    elif args.method == 'regional':
        # Create data loaders
        id_test_loader = setup_loader(args.test_data, test=True, batch_size=1)
        ood_test_loader = setup_loader(args.test_data, ood=True, batch_size=1)

        # Set smoothing parameters
        estimator.set_smoothing(method='none')

        # Collect data from ID test set
        id_imgs, id_labels, id_scores = [], [], []
        for X, y in id_test_loader:
            # Get prediction from perception model
            output = model(X)
            output = output.detach().numpy()

            # Estimate competency scores
            score_img = estimator.map_scores(X, output)[None,:,:]
            id_scores.append(score_img)

            # Save images for visualization
            id_imgs.append(X)
            id_labels.append(torch.zeros_like(score_img))

        id_imgs = torch.vstack(id_imgs)
        id_labels = torch.vstack(id_labels)
        id_scores = torch.vstack(id_scores)

        # Load OOD segmentation labels
        segmentation = pickle.load(open(os.path.join(args.decoder_dir, 'ood_labels.p'), 'rb'))
        seg_labels = segmentation['labels']
        seg_pixels = segmentation['pixels']

        # Collect data from OOD test set
        ood_imgs, ood_labels, ood_scores = [], [], []
        for batch, (X, y) in enumerate(ood_test_loader):

            # Grab segment labels and pixels for this image
            if batch >= len(seg_labels):
                break
            these_labels = seg_labels[batch]
            these_pixels = seg_pixels[batch]

            # Get prediction from perception model
            output = model(X)
            output = output.detach().numpy()

            # Estimate competency scores
            score_img = estimator.map_scores(X, output)[None,:,:]
            ood_scores.append(score_img)

            # Create image from true labels
            label_img = torch.zeros((1, X.size()[2], X.size()[3]))
            for pixels, label in zip(these_pixels, these_labels):
                label_img[:, pixels[0, :], pixels[1, :]] = float(label)
            ood_labels.append(label_img)

            # Save images for visualization
            ood_imgs.append(X)

        ood_imgs = torch.vstack(ood_imgs)
        ood_labels = torch.vstack(ood_labels)
        ood_scores = torch.vstack(ood_scores)

        # Create folders to save results
        output_dir = os.path.join(args.decoder_dir, 'competency')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        id_dir = os.path.join(output_dir, 'id')
        if not os.path.exists(id_dir):
            os.makedirs(id_dir)
        ood_dir = os.path.join(output_dir, 'ood')
        if not os.path.exists(ood_dir):
            os.makedirs(ood_dir)

        # Visualize orginal image, true labels, and competency estimates
        output_dirs = [id_dir, ood_dir]
        img_tensors = [id_imgs, ood_imgs]
        label_arrs  = [id_labels, ood_labels]
        score_arrs  = [id_scores, ood_scores]
        for output_dir, imgs, labels, scores in zip(output_dirs, img_tensors, label_arrs, score_arrs):
            for idx, (img, label, score) in enumerate(zip(imgs, labels, scores)):
                fig, axs = plt.subplots(1, 3, figsize=(12, 4))
                plt.subplot(1, 3, 1)
                img = np.squeeze(img.numpy() * 255).astype(np.uint8)
                img = np.swapaxes(np.swapaxes(img, 0, 1), 1, 2)
                plt.imshow(img)
                plt.title('Original Image')
                
                plt.subplot(1, 3, 2)
                label[label == -1] = 0.5
                plt.imshow(label, cmap='coolwarm', vmin=0, vmax=1)
                plt.title('Labeled Image')

                plt.subplot(1, 3, 3)
                im = plt.imshow(score, cmap='coolwarm_r', vmin=0, vmax=1)
                plt.title('Estimated Competency')

                fig.subplots_adjust(right=0.8)
                cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
                fig.colorbar(im, cax=cbar_ax)
                plt.savefig(os.path.join(output_dir, '{}.png'.format(idx)))
                plt.close()

    else:
        raise NotImplementedError('Unknown competency estimation method.')


if __name__ == "__main__":
    main()