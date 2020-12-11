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
import models
from torch import optim

image_size = models.image_size
batch_size = models.batch_size
workers = models.workers


dataroot = "/Users/adejuwon/Desktop/Pet Projects/PokeGan/pokemon"

"""dataset = dset.ImageFolder(root=dataroot,
                           transform=transforms.Compose([
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               util.alphaToWhite(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                           ]),
                           loader=util.my_loader)"""

dataset = dset.ImageFolder(root=dataroot,
                           transform=transforms.Compose([
                               transforms.Grayscale(),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               util.fillIn(),
                               transforms.Normalize((0.5), (0.5))
                           ]),
                           loader=util.my_loader)

dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True, num_workers=workers)

# optimizer = optim.RMSprop(netG.parameters(), lr=.001)
# lossFn = nn.BCELoss()
# for i in range(3000):
#     optimizer.zero_grad()
#     z = torch.randn((1, 100, 1, 1))
#     out = netG.forward(z)
#     loss = lossFn.forward(out*.5 + .5, dataset[803][0].view(1, 3, 32, 32)*.5+.5)
#     print(i, loss)
#     loss.backward()
#     optimizer.step()
#
# with torch.no_grad():
#     plt.imshow(dataset[803][0].permute(1,2,0)*.5 + .5)
#     plt.imshow(netG.forward(torch.randn((1, 100, 1, 1))).view(3,32,32).permute(1,2,0)*.5 + .5)

model = models.WGAN()
model.load("/Users/adejuwon/Desktop/Pet Projects/PokeGan/model_checkpoints/poke_gan_wc_1800_epochs_2020_11_14_05:38:08")
#model.train(500, dataloader)
#model = models.WGAN_GP()
#model.train(60, dataloader)
#model.save()

"""model.train(500, dataloader)
model.save()
model.train(500, dataloader)
model.save()
model.train(500, dataloader)
model.save()
model.train(300, dataloader)
model.save()"""

