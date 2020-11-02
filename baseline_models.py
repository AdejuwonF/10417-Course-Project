import torch.nn as nn
import torch
import torch.optim as optim
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

        self.conv1 = nn.Conv2d(3, 8, 3, padding=1)
        self.relu1 = nn.ReLU()
        self.maxPool1 = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(8, 12, 3, padding=1)
        self.relu2 = nn.ReLU()
        self.maxPool2 = nn.MaxPool2d(2)

        self.conv3 = nn.Conv2d(12, 16, 3, padding=1)
        self.relu3 = nn.ReLU()
        self.maxPool3 = nn.MaxPool2d(2)

        self.convT1 = nn.ConvTranspose2d(16, 12, 4, 2, 1)
        self.relu4 = nn.ReLU()

        self.convT2 = nn.ConvTranspose2d(12, 6, 4, 2, 1)
        self.relu5 = nn.ReLU()

        self.convT3 = nn.ConvTranspose2d(6, 1, 4, 2, 1)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxPool1(x)
        #print("first downsample layer: {0}".format(x.shape))
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxPool2(x)
        #print("second downsample layer: {0}".format(x.shape))
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.maxPool3(x)
        #print("third downsample layer: {0}".format(x.shape))


        x = self.convT1(x)
        x =self.relu4(x)
        #print("first upsample layer: {0}".format(x.shape))

        x = self.convT2(x)
        x = self.relu5(x)
        #print("second upsample layer: {0}".format(x.shape))
        x = self.convT3(x)
        x = self.sigmoid(x)
        # print("third upsample layer: {0}".format(x.shape))
        return x

# net = DeconvNet()
# coco_val = dset.CocoDetection(root=path + 'COCO_DATASET/val2017',
#                               annFile=path + 'COCO_DATASET/annotations/instances_val2017.json',
#                               transforms=transformCoCoPairs(128))
#
# """coco_train = dset.CocoDetection(root=path + 'COCO_DATASET/train2017',
#                                 annFile=path + 'COCO_DATASET/annotations/instances_train2017.json',
#                                 transforms=transformCoCoPairs(128))"""
#
# optimizer = optim.Adam(net.parameters(), lr=0.01)
# loss_func = nn.BCELoss()
# print("kadjshfjkahdsf")
# train_loss, val_loss = util.train(net, coco_val, 5, 50, coco_val, optimizer, loss_func)

# dataloader = DataLoader(coco_val, batch_size=1, shuffle=False, num_workers=0)
# ims, tgs = next(iter(dataloader))
# outs = net.forward(ims)
#
# img = ims[0,:,:,:]
# tg = tgs[0,:,:,:]
# out = outs[0,:,:,:]
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
# optimizer = optim.Adam(net.parameters(), lr=0.01)
# criterion = nn.BCELoss()
#
# for i in range(10000):
#     optimizer.zero_grad()
#     outs = net.forward(ims)
#     target = combinedMasks.unsqueeze(0)
#     loss = criterion(outs, target)
#     loss.backward()
#     optimizer.step()
#     print(i, loss)





# outs = net.forward(ims)
# out = outs[0, :, :, :]
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