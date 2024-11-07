
import os
import abc
import json
import copy
import torch
import pickle
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

import pytorch_ood.detector as oodd
import anomalib.models as models
import anomalib.data.utils as utils

from src.networks.model import NeuralNet
from src.datasets.setup_dataloader import setup_loader

ALL_OVERALL = ['parce', 'softmax', 'dropout', 'ensemble', 'temperature', 'kl', 'entropy', 'openmax', 'energy', 'odin', 'mahalanobis', 'knn', 'dice']
# ALL_OVERALL = ['parce', 'softmax', 'dropout', 'ensemble', 'temperature', 'kl', 'entropy', 'openmax', 'energy', 'odin', 'mahalanobis', 'vim', 'knn', 'she', 'dice', 'ganomaly']

class Detector:

    @abc.abstractmethod
    def comp_scores(self, inputs, outputs):
        pass

class Softmax(Detector):
    
    def __init__(self):
        self.name = 'Maximum Softmax'

    def comp_scores(self, inputs, outputs):
        # Get softmax probability of predicted class
        return np.max(outputs, axis=1)

class Dropout(Detector):

    def __init__(self, model, method='variance', prob=0.5, num_iterations=20):
        self.name = 'MC Dropout'
        
        # Apply dropout before last fully connected layer
        self.model = model
        self.model.layers.insert(-2, nn.Dropout(prob))
        self.model.net = nn.Sequential(*self.model.layers)

        # Define method for uncertainty estimation
        assert method in ['mean', 'variance', 'entropy']
        self.method = method
        self.num_iterations = num_iterations

    def comp_scores(self, inputs, outputs):
        # Determine predicted class
        preds = np.argmax(outputs, axis=-1)

        # Save softmax probability fo each iteration
        probs = []
        for itr in range(self.num_iterations):
            out = self.model(inputs).detach()
            probs.append(out[np.arange(len(out)),preds])
        probs = np.stack(probs, axis=-1)

        # Compute the average softmax probability
        if self.method == 'mean':
            return np.mean(probs, axis=-1).flatten()
       
        # Compute the variance of softmax probabilities
        elif self.method == 'variance':
            return -np.var(probs, axis=-1).flatten()
        
        # Compute the entropy of softmax probabilities
        elif self.method == 'entropy':
            return np.sum(probs * np.log(probs + 1e-16), axis=-1).flatten()

class Ensemble(Detector):

    def __init__(self, model_dir, method='variance'):
        self.name = 'Ensemble'

        # Load trained models
        with open(os.path.join(model_dir, 'layers.json')) as file:
            layer_args = json.load(file)

        self.models = []
        for file in os.listdir(model_dir):
            if not file.endswith('.pth'):
                continue
            model = NeuralNet(layer_args)
            model.load_state_dict(torch.load(os.path.join(model_dir, file)))
            model.eval()
            self.models.append(model)

        # Define method for uncertainty estimation
        assert method in ['mean', 'variance', 'entropy']
        self.method = method

    def comp_scores(self, inputs, outputs):
        # Determine predicted class
        preds = np.argmax(outputs, axis=-1)

        # Save softmax probability of predicted class for each model
        probs = []
        for model in self.models:
            out = model(inputs).detach()
            probs.append(out[np.arange(len(out)),preds])
        probs = np.stack(probs, axis=-1)

        # Compute the average softmax probability
        if self.method == 'mean':
            return np.mean(probs, axis=-1).flatten()
       
        # Compute the variance of softmax probabilities
        elif self.method == 'variance':
            return -np.var(probs, axis=-1).flatten()
        
        # Compute the entropy of softmax probabilities
        elif self.method == 'entropy':
            return np.sum(probs * np.log(probs + 1e-16), axis=-1).flatten()

class Temperature(Detector):

    def __init__(self, model, dataloader, filename=None, device):
        self.name = 'Temperature Scaling'

        if os.path.isfile(filename):
            self.estimator = pickle.load(open(filename, "rb"))
        else:
            # Determine the optimal temperature value
            self.estimator = oodd.TemperatureScaling(model)
            outputs, labels = [], []
            for X, y in dataloader:
                out = model(X.to(device))
                outputs.append(out)
                labels.append(y)
            outputs = torch.vstack(outputs)
            labels = torch.vstack(labels).flatten()
            self.estimator.fit_features(outputs.long(), labels.long())
            # print('Optimal temperature: {}'.format(self.estimator.t.item()))
        
            try:
                folder = os.path.dirname(filename)
                if not os.path.exists(folder):
                    os.makedirs(folder)
                pickle.dump(self.estimator, open(filename, "wb"))
            except:
                print('Warning: Trained {} model was not saved.'.format(self.name))

    def comp_scores(self, inputs, outputs):
        scores = self.estimator.predict_features(torch.from_numpy(outputs))
        return -scores.detach().numpy()

class KLMatching(Detector):

    def __init__(self, model, dataloader, filename=None, device):
        self.name = 'KL-Matching'

        if os.path.isfile(filename):
            self.estimator = pickle.load(open(filename, "rb"))
        else:
            # Estimate typical distributions for each class
            self.estimator = oodd.KLMatching(model)
            outputs, labels = [], []
            for X, y in dataloader:
                out = model(X.to(device))
                outputs.append(out)
                labels.append(y)
            outputs = torch.vstack(outputs)
            labels = torch.vstack(labels).flatten()
            self.estimator.fit_features(outputs, labels)
        
            try:
                folder = os.path.dirname(filename)
                if not os.path.exists(folder):
                    os.makedirs(folder)
                pickle.dump(self.estimator, open(filename, "wb"))
            except:
                print('Warning: Trained {} model was not saved.'.format(self.name))


    def comp_scores(self, inputs, outputs):
        scores = self.estimator.predict_features(torch.from_numpy(outputs))
        return scores.detach().numpy()

class Entropy(Detector):

    def __init__(self, model):
        self.name = 'Entropy'
        self.estimator = oodd.Entropy(model)
        
    def comp_scores(self, inputs, outputs):
        scores = self.estimator.score(torch.from_numpy(outputs))
        return -scores.detach().numpy()

class OpenMax(Detector):

    def __init__(self, model, dataloader, filename=None, device):
        self.name = 'OpenMax'

        if os.path.isfile(filename):
            self.estimator = pickle.load(open(filename, "rb"))
        else:
            # Estimate typical distributions for each class
            self.estimator = oodd.OpenMax(model)
            outputs, labels = [], []
            for X, y in dataloader:
                out = model(X.to(device))
                outputs.append(out)
                labels.append(y)
            outputs = torch.vstack(outputs).detach()
            labels = torch.vstack(labels).flatten().detach()
            self.estimator.fit_features(outputs, labels)
        
            try:
                folder = os.path.dirname(filename)
                if not os.path.exists(folder):
                    os.makedirs(folder)
                pickle.dump(self.estimator, open(filename, "wb"))
            except:
                print('Warning: Trained {} model was not saved.'.format(self.name))

    def comp_scores(self, inputs, outputs):
        scores = self.estimator.predict_features(torch.from_numpy(outputs))
        return -scores.detach().numpy()

class Energy(Detector):

    def __init__(self, model):
        self.name = 'Energy Based'
        self.estimator = oodd.EnergyBased(model)
        
    def comp_scores(self, inputs, outputs):
        scores = self.estimator.score(torch.from_numpy(outputs))
        return -scores.detach().numpy()

class ODIN(Detector):

    def __init__(self, model):
        self.name = 'ODIN'
        self.estimator = oodd.ODIN(model)
        
    def comp_scores(self, inputs, outputs):
        scores = self.estimator.predict(inputs)
        return -scores.detach().numpy()

class Mahalanobis(Detector):

    def __init__(self, model, dataloader, filename=None, device):
        self.name = 'Mahalanobis Distance'

        if os.path.isfile(filename):
            self.estimator = pickle.load(open(filename, "rb"))
        else:
            # Get feature extractor from model
            feature_extractor = copy.deepcopy(model)
            del feature_extractor.layers[-2:]
            feature_extractor.net = nn.Sequential(*feature_extractor.layers)

            # Estimate typical distributions for each class
            self.estimator = oodd.Mahalanobis(feature_extractor)
            features, labels = [], []
            for X, y in dataloader:
                z = model.get_feature_vector(X.to(device))
                features.append(z)
                labels.append(y)
            features = torch.vstack(features).detach()
            labels = torch.vstack(labels).flatten().detach()
            self.estimator.fit_features(features, labels)
        
            try:
                folder = os.path.dirname(filename)
                if not os.path.exists(folder):
                    os.makedirs(folder)
                pickle.dump(self.estimator, open(filename, "wb"))
            except:
                print('Warning: Trained {} model was not saved.'.format(self.name))

    def comp_scores(self, inputs, outputs):
        scores = self.estimator.predict(inputs)
        return -scores.detach().numpy()

class ViM(Detector):

    def __init__(self, model, dataloader, filename=None, device):
        self.name = 'Virtual Logit Matching'

        if os.path.isfile(filename):
            self.estimator = pickle.load(open(filename, "rb"))
        else:
            # Get feature extractor from model
            feature_extractor = copy.deepcopy(model)
            del feature_extractor.layers[-2:]
            feature_extractor.net = nn.Sequential(*feature_extractor.layers)

            # Get parameters of last linear layer
            d = model.layers[-2].in_features
            W = model.layers[-2].weight
            b = model.layers[-2].bias

            # Compute principle subspace
            self.estimator = oodd.ViM(feature_extractor, d, W, b)
            features, labels = [], []
            for X, y in dataloader:
                z = model.get_feature_vector(X.to(device))
                features.append(z)
                labels.append(y)
            features = torch.vstack(features).detach()
            labels = torch.vstack(labels).flatten().detach()
            self.estimator.fit_features(features, labels)
            
            try:
                folder = os.path.dirname(filename)
                if not os.path.exists(folder):
                    os.makedirs(folder)
                pickle.dump(self.estimator, open(filename, "wb"))
            except:
                print('Warning: Trained {} model was not saved.'.format(self.name))

    def comp_scores(self, inputs, outputs):
        scores = self.estimator.predict(inputs)
        return scores.detach().numpy()

class kNN(Detector):

    def __init__(self, model, dataloader, filename=None, device):
        self.name = 'k Nearest Neighbor'

        if os.path.isfile(filename):
            self.estimator = pickle.load(open(filename, "rb"))
        else:
            # Get feature extractor from model
            feature_extractor = copy.deepcopy(model)
            del feature_extractor.layers[-2:]
            feature_extractor.net = nn.Sequential(*feature_extractor.layers)

            # Fit nearest neighbor model
            self.estimator = oodd.KNN(feature_extractor)
            features, labels = [], []
            for X, y in dataloader:
                z = model.get_feature_vector(X.to(device))
                features.append(z)
                labels.append(y)
            features = torch.vstack(features).detach()
            labels = torch.vstack(labels).flatten().detach()
            self.estimator.fit_features(features, labels)
            
            try:
                folder = os.path.dirname(filename)
                if not os.path.exists(folder):
                    os.makedirs(folder)
                pickle.dump(self.estimator, open(filename, "wb"))
            except:
                print('Warning: Trained {} model was not saved.'.format(self.name))

    def comp_scores(self, inputs, outputs):
        scores = self.estimator.predict(inputs)
        return -scores[:,0].detach().numpy()

class SHE(Detector):

    def __init__(self, model, dataloader, filename=None, device):
        self.name = 'Simplified Hopfield Energy'

        if os.path.isfile(filename):
            self.estimator = pickle.load(open(filename, "rb"))
        else:
            # Get feature extractor and classifier from model
            feature_extractor = copy.deepcopy(model)
            del feature_extractor.layers[-2:]
            feature_extractor.net = nn.Sequential(*feature_extractor.layers)
            
            classifier = copy.deepcopy(model)
            del classifier.layers[:-2]
            classifier.net = nn.Sequential(*classifier.layers)

            # Estimate typical distributions for each class
            self.estimator = oodd.SHE(feature_extractor, classifier)
            features, labels = [], []
            for X, y in dataloader:
                z = model.get_feature_vector(X.to(device))
                features.append(z)
                labels.append(y)
            features = torch.vstack(features).detach()
            labels = torch.vstack(labels).flatten().detach()
            self.estimator.fit_features(features, labels)
            
            try:
                folder = os.path.dirname(filename)
                if not os.path.exists(folder):
                    os.makedirs(folder)
                pickle.dump(self.estimator, open(filename, "wb"))
            except:
                print('Warning: Trained {} model was not saved.'.format(self.name))

    def comp_scores(self, inputs, outputs):
        scores = self.estimator.predict(inputs)
        return scores.detach().numpy()

class DICE(Detector):

    def __init__(self, model, dataloader, filename=None, device):
        self.name = 'DICE'

        if os.path.isfile(filename):
            self.estimator = pickle.load(open(filename, "rb"))
        else:
            # Get feature extractor from model
            feature_extractor = copy.deepcopy(model)
            del feature_extractor.layers[-2:]
            feature_extractor.net = nn.Sequential(*feature_extractor.layers)

            # Get parameters of last linear layer
            W = model.layers[-2].weight
            b = model.layers[-2].bias

            # Compute principle subspace
            self.estimator = oodd.DICE(feature_extractor, W, b, p=0.7)
            features, labels = [], []
            for X, y in dataloader:
                z = model.get_feature_vector(X.to(device))
                features.append(z)
                labels.append(y)
            features = torch.vstack(features).detach()
            labels = torch.vstack(labels).flatten().detach()
            self.estimator.fit_features(features, labels)
            
            try:
                folder = os.path.dirname(filename)
                if not os.path.exists(folder):
                    os.makedirs(folder)
                pickle.dump(self.estimator, open(filename, "wb"))
            except:
                print('Warning: Trained {} model was not saved.'.format(self.name))


    def comp_scores(self, inputs, outputs):
        scores = self.estimator.predict(inputs)
        return -scores.detach().numpy()

class GANomaly(Detector):
    
    def __init__(self, dataloader, epochs, filename=None, device):
        self.name = 'GANomaly'

        if os.path.isfile(filename):
            self.estimator = pickle.load(open(filename, "rb"))
        else:
            # Initialize GANomaly model
            self.estimator = models.ganomaly.torch_model.GanomalyModel(
                input_size=(224,224),
                num_input_channels=3,
                n_features=64,
                latent_vec_size=100,
                extra_layers=0,
                add_final_conv_layer=True,
            )

            # Set optimization parameters
            generator_loss = models.ganomaly.loss.GeneratorLoss(1, 50, 1)
            discriminator_loss = models.ganomaly.loss.DiscriminatorLoss()
            d_opt = torch.optim.Adam(
                self.estimator.discriminator.parameters(),
                lr=0.0002, betas=(0.5, 0.999),
            )
            g_opt = torch.optim.Adam(
                self.estimator.generator.parameters(),
                lr=0.0002, betas=(0.5, 0.999),
            )

            # Train the GAN model
            for t in tqdm(range(epochs)):
                for X, y in dataloader:
                    # forward pass
                    inputs = F.interpolate(X.to(device), size=(224, 224))
                    padded, fake, latent_i, latent_o = self.estimator(inputs)
                    pred_real, _ = self.estimator.discriminator(padded)

                    # generator update
                    pred_fake, _ = self.estimator.discriminator(fake)
                    g_loss = generator_loss(latent_i, latent_o, padded, fake, pred_real, pred_fake)
                    g_opt.zero_grad()
                    g_loss.backward(retain_graph=True)
                    g_opt.step()

                    # discrimator update
                    pred_fake, _ = self.estimator.discriminator(fake.detach())
                    d_loss = discriminator_loss(pred_real, pred_fake)
                    d_opt.zero_grad()
                    d_loss.backward()
                    d_opt.step()
            
            try:
                folder = os.path.dirname(filename)
                if not os.path.exists(folder):
                    os.makedirs(folder)
                pickle.dump(self.estimator, open(filename, "wb"))
            except:
                print('Warning: Trained {} model was not saved.'.format(self.name))


        # Set min and max scores for normalization
        self.estimator.training = False
        self.min_score = torch.tensor(float("inf"), dtype=torch.float32)
        self.max_score = torch.tensor(float("-inf"), dtype=torch.float32)
        for X, y in dataloader:
            inputs = F.interpolate(X, size=(224, 224))
            scores = self.estimator(inputs)
            self.max_score = max(self.max_score, torch.max(scores))
            self.min_score = min(self.min_score, torch.min(scores))

    def comp_scores(self, inputs, outputs):
        batch, _, height, width = inputs.size()
        inputs = F.interpolate(inputs, size=(224, 224))
        scores = self.estimator(inputs)
        scores = (scores - self.min_score) / (self.max_score - self.min_score)
        return -scores.detach().numpy()
  
def load_estimator(method, model=None, model_dir=None, decoder_dir=None, test_data=None, save_file=None, device='cpu'):

    if method == 'parce':
        file = os.path.join(decoder_dir, 'parce.p')
        estimator = pickle.load(open(file, 'rb'))
    
    elif method == 'softmax':
        estimator = Softmax()

    elif method == 'dropout':
        estimator = Dropout(model)

    elif method == 'ensemble':
        estimator = Ensemble(model_dir)

    elif method == 'temperature':
        train_loader = setup_loader(test_data, val=True)
        estimator = Temperature(model, train_loader, save_file, device)

    elif method == 'kl':
        train_loader = setup_loader(test_data, val=True)
        estimator = KLMatching(model, train_loader, save_file, device)

    elif method == 'entropy':
        estimator = Entropy(model)

    elif method == 'openmax':
        train_loader = setup_loader(test_data, val=True)
        estimator = OpenMax(model, train_loader, save_file, device)

    elif method == 'energy':
        estimator = Energy(model)

    elif method == 'odin':
        estimator = ODIN(model)

    elif method == 'mahalanobis':
        train_loader = setup_loader(test_data, val=True)
        estimator = Mahalanobis(model, train_loader, save_file, device)

    elif method == 'vim':
        train_loader = setup_loader(test_data, val=True)
        estimator = ViM(model, train_loader, save_file, device)

    elif method == 'knn':
        train_loader = setup_loader(test_data, val=True)
        estimator = kNN(model, train_loader, save_file, device)

    elif method == 'she':
        train_loader = setup_loader(test_data, val=True)
        estimator = SHE(model, train_loader, save_file, device)

    elif method == 'dice':
        train_loader = setup_loader(test_data, val=True)
        estimator = DICE(model, train_loader, save_file, device)

    elif method == 'ganomaly':
        train_loader = setup_loader(test_data, val=True)
        estimator = GANomaly(train_loader, 10, save_file)
    
    else:
        raise NotImplementedError('Unknown Method for Competency Estimation')
    
    return estimator