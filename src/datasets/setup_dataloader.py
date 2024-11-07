import string
import numpy as np

import torch
import torchvision
from torch.utils.data import ConcatDataset

from src.datasets.custom_dataset import *

def get_class_names(data):
    if data == 'lunar':
        return ['sunny smooth', 'gray smooth', 'sunny bumpy', 'gray bumpy', 'light crater',
                'dark crater', 'edge of crater', 'hill'] + ['structures']
    elif data == 'speed':
        return ['speed limit 30', 'speed limit 50', 'speed limit 60', 'speed limit 70', 
                'speed limit 80', 'speed limit 100', 'speed limit 120'] + ['speed limit 20']
    elif data == 'pavilion':
        return ['facing building', 'facing parking', 'facing pavilion', 'hitting tree', 
                'middle area', 'open space 1', 'open space 2', 'wooded'] + ['pavilion']
    else:
        raise NotImplementedError('The {} dataset is not currently available.'.format(data))

def get_num_classes(data):
    if data == 'lunar':
        return 8
    elif data == 'speed':
        return 7
    elif data == 'pavilion':
        return 8
    else:
        raise NotImplementedError('The {} dataset is not currently available.'.format(data))

def make_weights_for_balanced_classes(labels):
    total_lbls = len(labels)
    unique_lbls = np.unique(labels)
    weights = np.zeros(len(labels))
    for lbl in unique_lbls:
        count = len(np.where(labels.flatten() == lbl)[0])
        weights[labels.flatten() == lbl] = total_lbls / count                           
    return weights 

def setup_loader(data, batch_size=None, train=False, val=False, test=False, ood=False, segment=None, modify=None, calibrate=None):
    
    if not data in ['lunar', 'speed', 'pavilion']:
        raise NotImplementedError('The {} dataset is not currently available.'.format(data))
    
    batch_size = 32 if batch_size is None else batch_size
    
    if train:
        if segment is not None:
            train_dataset = SegmentedDataset('./data/{}/'.format(data), 'train', segment)
            loader = torch.utils.data.DataLoader(train_dataset,
                        batch_size=batch_size, shuffle=True)
        elif modify is not None:
            train_dataset = ModifiedDataset('./data/{}/'.format(data), 'train', modify)
            weights = make_weights_for_balanced_classes(train_dataset.labels)
            sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))
            loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, 
                                                    sampler=sampler)
        else:
            train_dataset = CustomDataset('./data/{}/'.format(data), 'train')
            weights = make_weights_for_balanced_classes(train_dataset.labels)
            sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))
            loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, 
                                                    sampler=sampler)
        
    elif val:
        if segment is not None:
            val_dataset = SegmentedDataset('./data/{}/'.format(data), 'val', segment)
        elif modify is not None:
            val_dataset = ModifiedDataset('./data/{}/'.format(data), 'val', modify)
        else:
            val_dataset = CustomDataset('./data/{}/'.format(data), 'val')
        loader = torch.utils.data.DataLoader(val_dataset,
                    batch_size=batch_size, shuffle=True)
        
    elif test:
        if segment is not None:
            test_dataset = SegmentedDataset('./data/{}/'.format(data), 'test', segment)
        elif modify is not None:
            test_dataset = ModifiedDataset('./data/{}/'.format(data), 'test', modify)
        else:
            test_dataset = CustomDataset('./data/{}/'.format(data), 'test')
        loader = torch.utils.data.DataLoader(test_dataset,
                    batch_size=batch_size, shuffle=False)
        
    elif ood:
        if segment is not None:
            ood_dataset = SegmentedDataset('./data/{}/'.format(data), 'ood', segment)
        elif modify is not None:
            ood_dataset = ModifiedDataset('./data/{}/'.format(data), 'ood', modify)
        else:
            ood_dataset = CustomDataset('./data/{}/'.format(data), 'ood')
        loader = torch.utils.data.DataLoader(ood_dataset,
                    batch_size=batch_size, shuffle=False)

    return loader