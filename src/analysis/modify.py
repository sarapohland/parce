import random
import argparse
import matplotlib.pyplot as plt
from PIL import ImageEnhance, ImageFilter, ImageDraw

from src.utils.visualize import *

ALL_MODIFICATIONS = ['brightness', 'contrast', 'saturation', 'sharpness', 'noise', 'blur', 'pixelate', 'blackout', 'whiteout', 'mask']
EVAL_MODIFICATIONS = ['brightness', 'contrast', 'saturation', 'noise', 'pixelate']

class Blur(ImageEnhance._Enhance):
    def __init__(self, img):
        super().__init__()
        self.img = img

    def enhance(self, factor):
        # Add Gaussian blur to original image
        img_blurred = self.img.filter(ImageFilter.GaussianBlur(radius=factor))
        return img_blurred

class Noise(ImageEnhance._Enhance):
    def __init__(self, img):
        super().__init__()
        self.img = img

    def enhance(self, factor):
        # Add uniform random noise to original image
        orig_img = np.array(self.img)
        noise = np.random.uniform(-factor, factor, orig_img.shape)
        noisy_img = np.clip(orig_img + noise, 0, 255).astype(np.uint8)
        noisy_img = Image.fromarray(noisy_img)
        return noisy_img

class Pixelate(ImageEnhance._Enhance):
    def __init__(self, img):
        super().__init__()
        self.img = img

    def enhance(self, factor):
        # Compress image using the nearest neighbor for resampling
        width, height = self.img.size
        small_width = int(max(1, width // factor))
        small_height = int(max(1, height // factor))
        small_img = self.img.resize((small_width, small_height), Image.NEAREST)

        # Expand the image back to the original size
        pixelated_img = small_img.resize((width, height), Image.NEAREST)
        return pixelated_img

class Mask(ImageEnhance._Enhance):
    def __init__(self, img, color=None):
        super().__init__()
        self.img = img
        self.color = color if color is not None else self.get_average_color()

    def get_average_color(self):
        img_array = np.array(self.img)
        avg_color = tuple(np.mean(img_array, axis=(0, 1)).astype(int))
        return avg_color

    def enhance(self, factor):
        # Find the bounds of a rectangle that takes up some portion of the original image
        width, height = self.img.size
        region_width = int(width * factor)
        region_height = int(height * factor)
        top_left_x = (width - region_width) // 2
        top_left_y = (height - region_height) // 2
        bottom_right_x = top_left_x + region_width
        bottom_right_y = top_left_y + region_height

        # Draw a rectangle of the desired size and color on top of the original image
        draw = ImageDraw.Draw(self.img)
        draw.rectangle([top_left_x, top_left_y, bottom_right_x, bottom_right_y], fill=self.color)
        return self.img  

def get_default_factor(property):
    if property == 'saturation':
        return 1.0
    elif property == 'contrast':
        return 1.0
    elif property == 'brightness':
        return 1.0
    elif property == 'sharpness':
        return 1.0
    elif property == 'blur':
        return 0.0
    elif property == 'noise':
        return 0.0
    elif property == 'pixelate':
        return 1.0
    elif property == 'blackout':
        return 0.0
    elif property == 'whiteout':
        return 0.0
    elif property == 'mask':
        return 0.0
    else:
        raise NotImplementedError('Unknown image property: ', property)

def modify_image(img, property, factor):
    # Convert tensor/array to PIL image
    is_tensor = isinstance(img, torch.Tensor)
    is_array  = isinstance(img, np.ndarray)
    if is_tensor:
        img = tensor_to_pil(img)
    elif is_array:
        img = numpy_to_pil(img)

    # Select enhancement class
    if property == 'saturation':
        enhancer = ImageEnhance.Color(img)
    elif property == 'contrast':
        enhancer = ImageEnhance.Contrast(img)
    elif property == 'brightness':
        enhancer = ImageEnhance.Brightness(img)
    elif property == 'sharpness':
        enhancer = ImageEnhance.Sharpness(img)
    elif property == 'blur':
        enhancer = Blur(img)
    elif property == 'noise':
        enhancer = Noise(img)
    elif property == 'pixelate':
        enhancer = Pixelate(img)
    elif property == 'blackout':
        enhancer = Mask(img, color=(0,0,0))
    elif property == 'whiteout':
        enhancer = Mask(img, color=(255,255,255))
    elif property == 'mask':
        enhancer = Mask(img)
    else:
        raise NotImplementedError('Unknown image property: ', property)
    
    # Adjust image property by specified factor
    new_img = enhancer.enhance(factor)
    
    # Convert PIL image to tensor/array
    if is_tensor:
        new_img = pil_to_tensor(new_img)
    elif is_array:
        new_img = pil_to_numpy(new_img)
    return new_img

def modify_images(images, params):
    # Get segmentation parameters
    property = params['property']
    factor = params['factor']

    # Modify images one at a time
    modify_imgs = []
    for img in images:
        modify_imgs.append(modify_image(img, property, factor))

    # Return modified set of images
    if isinstance(modify_imgs[0], torch.Tensor):
        return torch.vstack(modify_imgs) 
    elif isinstance(modify_imgs[0], np.ndarray):
        return np.vstack(modify_imgs)
    else:
        return modify_images