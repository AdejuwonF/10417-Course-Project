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
from util import CenterCropTensor, TransformAnn, transformCoCoPairs, CocoDetectionCatIds

annFile = 'COCO_DATASET/annotations/instances_val2017.json'
coco=COCO(annFile)

filter_classes = "person"

path=""

# coco_val = dset.CocoDetection(root='COCO_DATASET/val2017',
#                         annFile='COCO_DATASET/annotations/instances_val2017.json',
#                         transform=transforms.ToTensor())

catIds = coco.getCatIds(catNms=filter_classes)
imgIds = coco.getImgIds(catIds=catIds)
print("Number of images containing all the classes:", len(imgIds))
print(imgIds)

coco_val_people = CocoDetectionCatIds(root=path + 'COCO_DATASET/val2017',
                              annFile=path + 'COCO_DATASET/annotations/instances_val2017.json',
                              catIds = catIds,
                              transforms=transformCoCoPairs(256))
print(len(coco_val_people))


net = DeconvNet()
optimizer = optim.Adam(net.parameters(), lr=0.01)
loss_func = nn.BCELoss()

train_loss, val_loss = util.train(net, coco_val_people, 5, 50, coco_val_people, optimizer, loss_func)
