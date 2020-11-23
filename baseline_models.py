import torch.nn as nn
import torch
import torch.optim as optim
import torch.functional as F
import util
import torchvision.datasets as dset
from torch.utils.data import DataLoader
from util import CenterCropTensor, TransformAnn, transformCoCoPairs, transformCoCoPairsResize
import matplotlib.pyplot as plt
path = ""

nf = 64
img_size = 128
class DeconvNet(nn.Module):
    def __init__(self):
        super(DeconvNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, nf, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(nf, 2 * nf, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(2 * nf, 4 * nf, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(4 * nf, 2 * nf, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(2 * nf, nf, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(nf, 1, 4, 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoding = self.encoder.forward(x)
        mask = self.decoder.forward(encoding)
        return mask


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

# coco_val = dset.CocoDetection(root=path + 'COCO_DATASET/val2017',
#                                annFile=path + 'COCO_DATASET/annotations/instances_val2017.json',
#                                transforms=transformCoCoPairsResize(128))
# net = DeconvNet()
# dataloader = DataLoader(coco_val, batch_size=1, shuffle=False, num_workers=0)
# ims, tgs = next(iter(dataloader))
# print(ims.shape, ims, ims.type())
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
# optimizer = optim.Adam(net.parameters(), lr=0.0001)
# criterion = nn.BCELoss()
#
# for i in range(1000):
#     optimizer.zero_grad()
#     outs = net.forward(ims)
#     target = combinedMasks.unsqueeze(0).unsqueeze(0)
#     loss = criterion(outs, target)
#     loss.backward()
#     optimizer.step()
#     print(i, loss)
#
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