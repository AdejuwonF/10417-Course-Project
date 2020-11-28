import torch
import torchvision as tv
import torch.functional as F
import torch.nn.functional
import torchvision.datasets as dset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import pycocotools
from pycocotools.coco import COCO
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib
from matplotlib import colors
from matplotlib.colors import ListedColormap
import PIL
import skimage.io as io
from baseline_models import DeconvNet
import torch.nn as nn
import torch.optim as optim
import util
from util import CenterCropTensor, TransformAnn, transformCoCoPairs, CocoDetectionCatIds, transformCoCoPairsResize
import pickle as pk
import time
import math
import numpy as np
import os
import subprocess
import glob
import random

def zero_init(m):
    return

class CAGenerator(nn.Module):
    def __init__(self, state_size=20, nc=1, hidden_layer_size=128, nz=100, shape=(32,32)):
        super(CAGenerator, self).__init__()
        self.H, self.W = shape
        self.nz = nz
        self.state_size = state_size
        self.hidden_layer_size = hidden_layer_size
        self.nc = nc

        self.init_state = nn.Sequential(
            nn.Conv2d(self.nz, 50, 1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(50, state_size-1, 1, stride=1),
        )

        self.percieve = nn.Sequential(
            nn.Conv2d(self.state_size, hidden_layer_size, 3, stride=1, padding=1, groups=1)
        )

        self.update = nn.Sequential(
            nn.Conv2d(hidden_layer_size, hidden_layer_size, 1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_layer_size, self.state_size, 1, stride=1)
        )
        self.max_pool = nn.MaxPool2d(3, 1, 1)
        self.sigmoid = nn.Sigmoid()

        # Initialize final weights to 0, default to do nothing behavior
        self.update._modules['2'].weight.data.fill_(0)
        self.update._modules['2'].bias.data.fill_(0)

    def stochastic_update(self, state_grid, updates, p=1):
        mask = torch.rand_like(state_grid[:, 0:1, :, :]) < p
        updates = mask * updates
        return state_grid + updates

    def get_living_mask(self, state_grid, threshold=.1, alpha_channel=0):
        mask = self.max_pool.forward(state_grid[:, alpha_channel:alpha_channel+1, :, :]) > threshold
        return mask

    def ca_step(self, state_grid):
        pre_living_mask = self.get_living_mask(state_grid)

        perception = self.percieve.forward(state_grid)
        updates = self.update.forward(perception)

        state_grid = self.stochastic_update(state_grid, updates)

        # Currently clamp values to fall in range, prob better eway to handle this.
        # state_grid = torch.cat([(state_grid[:, :self.nc, :, :]).clamp(0,1),# * self.get_living_mask(state_grid),
        #                            state_grid[:, self.nc:, :, :]], dim=1)

        post_living_mask = self.get_living_mask(state_grid)

        living_mask = post_living_mask & pre_living_mask
        state_grid = state_grid * living_mask

        return state_grid

    def forward(self, z, steps_range=(64, 64), return_state=False):
        """The first 3 channels of the state grid are immutable and correspond to the color channels of the input
        image.  The remaining channels are hidden and used to represent a cells state.  The 3rd channel is currently
        being used as an 'alive' channel, which may be used to threshold which cells are considered alive and thus
        are allowed to have cell states.  The 4th channel will represent the probability that a cell believes itself
        to be part of a person mask."""
        N = z.shape[0]
        state_grid = torch.zeros((N, self.state_size, self.H, self.W))
        centerH = self.H//2
        centerW = self.W//2
        init_hidden = self.init_state.forward(z)
        state_grid[:, 1:, centerH:centerH+1, centerW:centerW+1] = init_hidden
        state_grid[:, 0, centerH, centerW] = .5

        steps = random.randint(steps_range[0], steps_range[1])
        for step in range(steps):
            state_grid = self.ca_step(state_grid)

        # state_grid = torch.cat([(state_grid[:, :self.nc, :, :]).clamp(0,1),# * self.get_living_mask(state_grid),
        #                            state_grid[:, self.nc:, :, :]], dim=1)

        if return_state:
            return state_grid
        else:
            return state_grid[:, :self.nc, :, :]


# dataset = dset.MNIST(root="", train=True, download=True,
#                     transform=transforms.Compose([transforms.Resize(32), transforms.ToTensor()]))
#
# img, label = dataset[254]
# img = img.unsqueeze(0)
#
# #criterion = nn.MSELoss()
# def criterion(pred, tg):
#     return torch.mean(torch.sum((tg-pred)**2, dim=[1,2,3])/2)
# model = CAGenerator()
# optimizer = optim.Adam(model.parameters(), lr=1e-4)
#
# in_size = 100
# with torch.no_grad():
#     plt.imshow(model.forward(torch.randn((1, in_size, 1, 1))).squeeze(0).permute([1,2,0]))
#     plt.show()
#
# for i in range(1000):
#     out = model.forward(torch.randn((1, in_size, 1, 1)))
#     loss = criterion(out, img.repeat(1,1,1,1))
#
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
#
#     print("Step:{0} Loss:{1}".format(i, loss.item()))
#
#
#
# with torch.no_grad():
#     plt.imshow(model.forward(torch.randn((1, in_size, 1, 1))).squeeze(0).permute([1,2,0]), cmap="Greys")
#     plt.show()
#
# with torch.no_grad():
#     state = model.forward(torch.randn((1,in_size,1,1)), 0, True)
#     outs = [None]*51
#     outs[0] = state
#     for i in range(1, 51):
#         state = model.ca_step(state)
#         outs[i] = state

def generate_video(predictions, image, video_name="video"):
    folder = "MNIST_EXAMPLES"
    for i in range(len(predictions)):
        plt.imshow(predictions[i][:, 0:1, :, :].squeeze(0).permute([1,2,0]), cmap="Greys")
        plt.savefig(folder + "/file%02d.png" % i)

    os.chdir("MNIST_EXAMPLES")
    subprocess.call([
        './ffmpeg', '-framerate', '8', '-i', 'file%02d.png', '-r', '30', '-pix_fmt', 'yuv420p',
        video_name + ".mp4"
    ])
    for file_name in glob.glob("*.png"):
        os.remove(file_name)