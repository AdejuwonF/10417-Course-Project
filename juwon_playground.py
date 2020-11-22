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
                              transforms=transformCoCoPairs(128))
print(len(coco_val_people))


net = torch.load("/Users/adejuwon/Desktop/CMU School Junk/Fall 2020/10417 Intermediate Deep Learning/Course Project/baseline_model_epoch_100")

img, mask = coco_val_people[3]
print(img.shape, mask.shape)
plt.imshow(img.permute(1, 2, 0))
plt.show()

#  Puts all the masks into one channel
plt.imshow(mask[1,:,:].unsqueeze(0).permute(1, 2, 0))
plt.show()

with torch.no_grad():
    pred_mask = net.forward(img.unsqueeze(0))
    plt.imshow(pred_mask.squeeze(0).permute(1, 2, 0))
    plt.show()

with torch.no_grad():
    pred_mask = net.forward(img.unsqueeze(0))
    plt.imshow(pred_mask.squeeze(0).permute(1, 2, 0))
    plt.show()




