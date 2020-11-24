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
import time


nc = 3


class CA_Model(nn.Module):
    def __init__(self, hidden_state_size=13, perception_vectors=4):
        super(CA_Model, self).__init__()
        self.state_size = hidden_state_size + nc
        self.hidden_state_size = hidden_state_size
        self.perception_vectors = perception_vectors
        self.percieve = nn.Sequential(
            nn.Conv2d(self.state_size, self.perception_vectors*self.state_size, 3, stride=1, padding=1, groups=self.state_size)
        )

        self.update = nn.Sequential(
            nn.Conv2d(self.perception_vectors*self.state_size, 128, 1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, self.hidden_state_size, 1, stride=1)
        )
        self.max_pool = nn.MaxPool2d(3, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def stochastic_update(self, hidden_state, updates, p=1):
        mask = torch.rand_like(hidden_state[:, 0:1, :, :]) < p
        updates = mask * updates
        #hidden_state = hidden_state + updates
        return hidden_state + updates

    def alive_mask(self, hidden_state, threshold=0):
        mask = self.max_pool.forward(hidden_state[:, :1, :, :]) > threshold
        # hidden_state = hidden_state * mask
        return hidden_state * mask

    def ca_step(self, state_grid):
        perception = self.percieve.forward(state_grid)
        updates = self.update.forward(perception)

        visible_state = state_grid[:, :3, :, :]
        hidden_state = state_grid[:, 3:, :, :]

        hidden_state = self.stochastic_update(hidden_state, updates)
        hidden_state = self.alive_mask(hidden_state)
        hidden_state = torch.cat([hidden_state[:, :1, :, :],
                                  self.sigmoid.forward(hidden_state[:, 1:2, :, :]),
                                  hidden_state[:, 2:, :, :]], dim=1)
        state_grid = torch.cat((visible_state, hidden_state), dim=1)
        return state_grid

    def forward(self, x, steps=60):
        N, C, H, W = x.shape
        """The first 3 channels of the state grid are immutable and correspond to the color channels of the input
        image.  The remaining channels are hidden and used to represent a cells state.  The 3rd channel is currently
        being used as an 'alive' channel, which may be used to threshold which cells are considered alive and thus
        are allowed to have cell states.  The 4th channel will represent the probability that a cell believes itself
        to be part of a person mask."""
        if C == nc + self.hidden_state_size:
            state_grid = x
        else:
            state_grid = torch.cat([x, torch.zeros(N, self.hidden_state_size, H, W)], dim=1)
        for step in range(steps):
            #print(step)
            state_grid = self.ca_step(state_grid)
        return state_grid


# start = time.time()
# model = CA_Model()
# x = torch.zeros((16,3,64,64))
# out = model.forward(x, 100)
# z = torch.mean(out)
# z.backward()
# print(time.time() - start)

# path = ""
# coco_val = dset.CocoDetection(root=path + 'COCO_DATASET/val2017',
#                                annFile=path + 'COCO_DATASET/annotations/instances_val2017.json',
#                                transforms=transformCoCoPairsResize(64))
# net = CA_Model()
# dataloader = DataLoader(coco_val, batch_size=1, shuffle=False, num_workers=0)
# ims, tgs = next(iter(dataloader))
# print(ims.shape, ims, ims.type())
# outs = net.forward(ims)
#
# img = ims[0,:,:,:]
# tg = tgs[0,:,:,:]
# out = outs[0,4:5,:,:]
# print(img.size(), tg.size(), out.size())
#
# plt.imshow(img.permute(1, 2, 0))
# plt.show()
#
# combinedMasks, indices = torch.max(tg, dim=0)
# plt.imshow(combinedMasks.unsqueeze(0).permute(1, 2, 0))
# plt.show()
#
# plt.imshow(out.permute(1, 2, 0).detach().numpy())
# plt.show()
#
# optimizer = optim.Adam(net.parameters(), lr=0.0001)
# criterion = nn.BCELoss()
#
# for i in range(200):
#     optimizer.zero_grad()
#     outs = net.forward(ims)[:, 4:5, :, :]
#     target = combinedMasks.unsqueeze(0).unsqueeze(0)
#     loss = criterion(outs, target)
#     loss.backward()
#     optimizer.step()
#     print(i, loss)
#
# outs = net.forward(ims)
# out = outs[0, 4:5, :, :]
# print(img.size(), tg.size(), out.size())
#
# plt.imshow(img.permute(1, 2, 0))
# plt.show()
#
# combinedMasks, indices = torch.max(tg, dim=0)
# plt.imshow(combinedMasks.unsqueeze(0).permute(1, 2, 0))
# plt.show()
#
# plt.imshow(out.permute(1, 2, 0).detach().numpy())
# plt.show()
