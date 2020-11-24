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
import PIL
import skimage.io as io
from baseline_models import DeconvNet
import torch.nn as nn
import torch.optim as optim
import util
from util import CenterCropTensor, TransformAnn, transformCoCoPairs, CocoDetectionCatIds, transformCoCoPairsResize
import pickle as pk


nc = 3


class CA_Model(nn.Module):
    def __init__(self, hidden_state_size=16, perception_vectors=5):
        super(CA_Model, self).__init__()
        self.state_size = hidden_state_size + nc
        self.hidden_state_size = hidden_state_size
        self.perception_vectors = perception_vectors
        self.percieve = nn.Sequential(
            nn.Conv2d(self.state_size, self.perception_vectors*self.state_size, 3, stride=1, padding=1, groups=self.state_size)
        )

        self.update = nn.Sequential(
            nn.Conv2d(self.perception_vectors*self.state_size, 128, 1, stride=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, self.hidden_state_size, 1, stride=1)
        )
        self.max_pool = nn.MaxPool2d(3, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def stochastic_update(self, state_grid, updates, p=1):
        mask = torch.rand_like(state_grid[:, 3:, :, :]) < p
        updates = mask * updates
        state_grid[:, 3:, :, :] += updates
        return state_grid

    def alive_mask(self, state_grid, threshold=0):
        mask = self.max_pool.forward(state_grid[:, 4:5, :, :]) > threshold
        state_grid[:, 3:, :, :] *= mask
        return state_grid

    def forward(self, x, steps=200):
        N, C, H, W = x.shape
        """The first 3 channels of the state grid are immutable and correspond to the color channels of the input
        image.  The remaining channels are hidden and used to represent a cells state.  The 3rd channel is currently
        being used as an 'alive' channel, which may be used to threshold which cells are considered alive and thus
        are allowed to have cell states.  The 4th channel will represent the probability that a cell believes itself
        to be part of a person mask."""
        state_grid = torch.cat([x, torch.zeros(N, self.hidden_state_size, H, W)], dim=1)
        for step in range(steps):
            perception = self.percieve.forward(state_grid)
            updates = self.update.forward(perception)
            state_grid = self.stochastic_update(state_grid, updates)
            state_grid = self.alive_mask(state_grid)
            state_grid[:, 4, :, :] = self.sigmoid.forward(state_grid[:, 4, :, :])

        return state_grid


