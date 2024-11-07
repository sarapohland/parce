import os
import torch
import pickle
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def numpy_to_pil(img):
    img = np.squeeze(img * 255).astype(np.uint8)
    img = np.swapaxes(np.swapaxes(img, 0, 1), 1, 2)
    img = Image.fromarray(img)
    return img

def pil_to_numpy(img):
    img = np.array(img) / 255
    img = np.swapaxes(np.swapaxes(img, 1, 2), 0, 1)[None,:]
    return img

def tensor_to_pil(img):
    img = np.squeeze(img.numpy() * 255).astype(np.uint8)
    img = np.swapaxes(np.swapaxes(img, 0, 1), 1, 2)
    img = Image.fromarray(img)
    return img

def pil_to_tensor(img):
    img = np.array(img) / 255
    img = np.swapaxes(np.swapaxes(img, 1, 2), 0, 1)[None,:]
    img = torch.from_numpy(img)
    return img

def visualize_img(img):
    img = np.squeeze(img.detach().numpy() * 255).astype(np.uint8)
    if img.ndim == 3:
        img = np.swapaxes(np.swapaxes(img, 0, 1), 1, 2)
    plt.imshow(img)