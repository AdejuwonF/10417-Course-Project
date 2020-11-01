import torch.nn as nn
import torch
import torch.functional as F
import util
import torchvision.datasets as dset
from torch.utils.data import DataLoader
from util import CenterCropTensor, TransformAnn, transformCoCoPairs
import matplotlib.pyplot as plt
path = ""
class DeconvNet(nn.Module):
    def __init__(self):
        super(DeconvNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 4, 3, padding=1)
        self.relu1 = nn.ReLU()
        self.maxPool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(4, 8, 3, padding=1)
        self.relu2 = nn.ReLU()
        self.maxPool2 = nn.MaxPool2d(2)


        self.convT1 = nn.ConvTranspose2d(8, 4, 4, 2, 1)
        self.relu3 = nn.ReLU()
        self.convT2 = nn.ConvTranspose2d(4, 1, 4, 2, 1)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxPool1(x)
        print("first downsample layer: {0}".format(x.shape))
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxPool2(x)
        print("second downsample layer: {0}".format(x.shape))

        x = self.convT1(x)
        x =self.relu3(x)
        print("first upsample layer: {0}".format(x.shape))

        x = self.convT2(x)
        x = self.sigmoid(x)
        print("second upsample layer: {0}".format(x.shape))
        return x

net = DeconvNet()
coco_val = dset.CocoDetection(root=path + 'COCO_DATASET/val2017',
                              annFile=path + 'COCO_DATASET/annotations/instances_val2017.json',
                              transforms=transformCoCoPairs(256))

dataloader = DataLoader(coco_val, batch_size=1, shuffle=False, num_workers=0)
ims, tgs = next(iter(dataloader))
outs = net.forward(ims)

for i in range(ims.size()[0]):
        img = ims[i,:,:,:]
        tg = tgs[i,:,:,:]
        out = outs[i,:,:,:]
        print(img.size(), tg.size(), out.size())

        plt.imshow(img.permute(1, 2, 0))
        plt.show()

        combinedMasks, indices = torch.max(tg, dim=0)
        plt.imshow(combinedMasks.unsqueeze(0).permute(1, 2, 0))
        plt.show()

        plt.imshow(out.permute(1, 2, 0).detach().numpy())
        plt.show()

