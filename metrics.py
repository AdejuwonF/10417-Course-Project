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
import matplotlib as plt
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

def inception_score(generator, classifier,  noise_dim, n_classes=10, num_samples=50000):
    KL = nn.KLDivLoss()
    posteriors = torch.zeros(num_samples, n_classes)

    #KL(log(prior), posterior) == Dkl(posteropr || prior)
    for i in range(num_samples):
        noise = torch.randn(noise_dim)
        sample = generator.forward(noise)
        posteriors[i] = sample

    priors = torch.mean(posteriors, dim=0).repeat((num_samples, 1))

    return torch.exp(KL(torch.log(priors), posteriors)/num_samples).item()








