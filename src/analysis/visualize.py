import argparse
import matplotlib.pyplot as plt

from src.datasets.setup_dataloader import setup_loader
from src.utils.visualize import *

# Plot original and edited image
def plot_image_mod(X, Xnew, property, factor, filename):

    fig = plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.title('Original Image')
    visualize_img(X)

    plt.subplot(1, 2, 2)
    plt.title('Edited Image')
    visualize_img(Xnew)

    plt.suptitle('{}: {}'.format(property.capitalize(), round(factor, 2)))
    if filename is not None:
        plt.savefig(filename)
        plt.close()
    else:
        plt.show()


def main():

    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('--test_data', type=str, default='lunar')
    parser.add_argument('--property', type=str, default='saturation')
    parser.add_argument('--factor', type=float, default='1.0')
    parser.add_argument('--file', type=str, default='None')
    parser.add_argument('--idx', type=int, default='-1')
    args = parser.parse_args()

    filename = None if args.file == 'None' else args.file
    
    # Create data loader
    params = {'property': args.property, 'factor': args.factor}
    orig_loader = setup_loader(args.test_data, batch_size=1, test=True)
    modi_loader = setup_loader(args.test_data, batch_size=1, test=True, modify=params)

    for i, (orig, modi) in enumerate(zip(orig_loader, modi_loader)):
        if i < args.idx:
            continue
        elif args.idx > 0 and i > args.idx:
            break
        X, _ = orig
        Xnew, _ = modi
        plot_image_mod(X, Xnew, args.property, args.factor, filename)

if __name__=="__main__":
    main()