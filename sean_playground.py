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

from util import CenterCropTensor, TransformAnn, transformCoCoPairs, pixel_accuracy, intersection_over_union
import pickle as pk

# annFile = 'COCO_DATASET/annotations/instances_val2017.json'
# coco=COCO(annFile)
#
# filter_classes = "person"

# coco_val = dset.CocoDetection(root='COCO_DATASET/val2017',
#                         annFile='COCO_DATASET/annotations/instances_val2017.json',
#                         transform=transforms.ToTensor())

# with open("/Users/sean/Downloads/baseline_train_loss.pk", 'rb') as f:
#     loss = pk.load(f)
#     print(loss)
#
#
# catIds = coco.getCatIds(catNms=filter_classes)
# imgIds = coco.getImgIds(catIds=catIds)
# print("Number of images containing all the classes:", len(imgIds))

# y = (torch.rand((3,30,40)) > 0.5).to(torch.int32)
# tgs = (torch.rand((3,30,40)) > 0.5).to(torch.int32)
# print(pixel_accuracy(y, tgs))
# print(intersection_over_union(y, tgs))

a = [[1, 2], [3, 4]]
b = torch.as_tensor(a)
print(b.sum(dim=0))