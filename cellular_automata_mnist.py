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
nc = 1

def zero_init(m):
    return

class CAMnist(nn.Module):
    def __init__(self, hidden_state_size=19, perception_vectors=3):
        super(CAMnist, self).__init__()
        self.state_size = hidden_state_size + nc
        self.hidden_state_size = hidden_state_size
        self.perception_vectors = perception_vectors
        self.percieve = nn.Sequential(
            nn.Conv2d(self.state_size, 80, 3, stride=1, padding=1, groups=1)
        )

        self.update = nn.Sequential(
            nn.Conv2d(80, 80, 1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(80, self.hidden_state_size, 1, stride=1)
        )
        self.max_pool = nn.MaxPool2d(3, 1, 1)
        self.sigmoid = nn.Sigmoid()

        self.update._modules['2'].weight.data.fill_(0)
        self.update._modules['2'].bias.data.fill_(0)

    def stochastic_update(self, hidden_state, updates, p=1):
        mask = torch.rand_like(hidden_state[:, 0:1, :, :]) < p
        updates = mask * updates
        #hidden_state = hidden_state + updates
        return hidden_state + updates

    def get_living_mask(self, hidden_state, threshold=.1):
        mask = self.max_pool.forward(hidden_state[:, :1, :, :]) > threshold
        # hidden_state = hidden_state * mask
        return mask

    def ca_step(self, state_grid):
        perception = self.percieve.forward(state_grid)
        updates = self.update.forward(perception)

        visible_state = state_grid[:, :nc, :, :]
        hidden_state = state_grid[:,nc:, :, :]

        hidden_state = self.stochastic_update(hidden_state, updates)
        # hidden_state = torch.cat([hidden_state[:, :1, :, :],
        #                           self.sigmoid.forward(hidden_state[:, 1:2, :, :]),
        #                           hidden_state[:, 2:, :, :]], dim=1)

        living_mask = visible_state > 0.1
        hidden_state = hidden_state * living_mask

        state_grid = torch.cat((visible_state, hidden_state), dim=1)
        return state_grid

    def forward(self, x, steps=20):
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

class CAMnistDataset(tv.datasets.vision.VisionDataset):
    def __init__(self, root):
        super(CAMnistDataset, self).__init__(root)
        self.dataset = dset.MNIST(root="", train=True, download=True,
                             transform=transforms.Compose([
                                 transforms.ToTensor()]
                             ))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img, target = self.dataset[index]

        target_state = torch.zeros((10, 28, 28))
        alive_cells = (img > .1)
        target_state[target:target+1, :, :] += alive_cells

        return (img, target_state)


dataset = CAMnistDataset(root="")
net = CAMnist()
dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0)

optimizer = optim.Adam(net.parameters(), lr=.0001)
#criterion = nn.MSELoss()
def criterion(pred, tg):
    return torch.mean(torch.sum((tg-pred)**2, dim=[1,2,3])/2)
losses = []

#Train Loop
for epoch in range(1,1):
    iter_loss = 0
    for i, (inputs, targets) in enumerate(dataloader):
        state = net.forward(inputs)
        preds = state[:, -10:, :, :]
        loss = criterion(preds, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        iter_loss += loss.item()
        if i % 200 == 0 and i!=0:
            print("Iteration {0} of Epoch {1}\n Cumulative Loss:{2}".format(i, epoch, math.log10(iter_loss/200)))
            losses.append(math.log10(iter_loss/200))
            iter_loss = 0

net.load_state_dict(torch.load("CAMNIST.pk"))


#Sets up colors
BW = cm.get_cmap("Greys", 11)
colors = np.array([[256,256,256,256],
            [128, 0, 0, 256],
            [230, 25, 75, 256],
            [70, 240, 240, 256],
            [210, 245, 60, 256],
            [250, 190, 190, 256],
            [170, 110, 40, 256],
            [170, 255, 195, 256],
            [165, 163, 159, 256],
            [0, 128, 128, 256],
            [128, 128, 0, 256]])/256 # This is the background.

newcolors = BW(np.linspace(-1, 10, 11))
newcolors[:11, :] = colors
newcmp = ListedColormap(newcolors)
minima = -1
maxima = 9
norm = matplotlib.colors.Normalize(vmin=minima, vmax=maxima, clip=True)
mapper = cm.ScalarMappable(norm=norm, cmap=newcmp)


# Gets results of 20 steps
img, tg = dataset[1000]
k=0
with torch.no_grad():
    outs = [net.forward(img.unsqueeze(0), i) for i in range(20)]



#Displays target colors
plt.imshow(mapper.to_rgba(torch.argmax(tg, dim=0)*torch.sum(tg, dim=0) - (1-torch.sum(tg, dim=0))), cmap=newcmp)

#Can repeatedly call this block to see how cells change
colored = (torch.argmax(outs[k][:, 10:, :, :], dim=1)*torch.sum(tg, dim=0) - (1-torch.sum(tg, dim=0))).permute([1,2,0]).squeeze(2)
plt.imshow(mapper.to_rgba(colored), cmap=newcmp)
k+=1
