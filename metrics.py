import torch
import torchvision as tv
import torch.functional as F
import torch.nn.functional
import torch.nn as nn
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torchvision.datasets.vision import VisionDataset
import pycocotools
from pycocotools.coco import COCO
import matplotlib.pyplot as plt
import PIL
import skimage.io as io
import pickle as pk
from tqdm import tqdm
import os.path as osp
from torch.utils.data import DataLoader
import time
from PIL import Image
import os
import os.path
from classifier import Classifier
from WGAN_WC import WGAN

def KL_div(input, target, eps=1e-16):
    kl_div = input*(torch.log((input+eps)/(target+eps)))
    return torch.mean(torch.sum(kl_div, dim=1))

def inception_score(generator, classifier,  noise_dim, n_classes=10, num_samples=50000):
    with torch.no_grad():
        softmax = nn.Softmax(dim=1)

        #KL(log(prior), posterior) == Dkl(posteropr || prior)
        noise = torch.randn((num_samples, noise_dim, 1, 1))
        posteriors = softmax(classifier.forward(generator.forward(noise)))

        priors = torch.mean(posteriors, dim=0).repeat((num_samples, 1))

    return torch.exp(KL_div(posteriors, priors))

wgan = WGAN()

state_dict = torch.load("classifier_checkpoints/classifier_checkpoints_ndf32/epoch10")
classifier = state_dict["model"]

#gan = whatever the fuck
#print(inception_score(gan.generator, classifier, 100, 10, 50000))
