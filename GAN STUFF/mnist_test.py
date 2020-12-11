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
import MNIST_GAN as models
from torch import optim
import time
from torchvision.utils import save_image

image_size = models.image_size
batch_size = models.batch_size
workers = models.workers


dataroot = "pokemon"

# dataset = dset.MNIST(root="./",
#                      transform=transforms.Compose([
#                          transforms.CenterCrop(image_size),
#                          transforms.ToTensor()
#                      ]),
#                      download=True,
#                      train=True)
dataset = dset.MNIST(root="", train=True, download=True,
                    transform=transforms.Compose([transforms.Resize(image_size), transforms.ToTensor()]))


dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True, num_workers=workers)

model = models.WGAN()
start = time.time()
model.train(30, dataloader)
print(time.time() - start)
i = 0
for img in (model.img_list):
    save_image(img, "MNIST_playground_output/epoch_" + str(i) + ".png")
    i += 1

