import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import torchvision.datasets as dset
from PIL import Image
import util
import MNIST_GAN.MNIST_GAN as models
from torch import optim

image_size = models.image_size
batch_size = models.batch_size
workers = models.workers


dataroot = "/Users/adejuwon/Desktop/Pet Projects/PokeGan/pokemon"

dataset = dset.MNIST(root="/Users/adejuwon/Desktop/Pet Projects/PokeGan/MNIST",
                     transform=transforms.Compose([
                         transforms.CenterCrop(image_size),
                         transforms.ToTensor()
                     ]),
                     download=True,
                     train=True)


dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True, num_workers=workers)

model = models.WGAN()
model.load("/Users/adejuwon/Desktop/Pet Projects/PokeGan/MNIST_GAN/gan_wc_10_epochs2020_11_14_17:20:12")
#model.train(10, dataloader)
#model.save("/Users/adejuwon/Desktop/Pet Projects/PokeGan/MNIST_GAN")

