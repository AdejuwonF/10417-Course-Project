import torch
import torchvision as tv
import torch.functional as F
import torchvision.datasets as dset
import torchvision.transforms as transforms
import pycocotools
from pycocotools.coco import COCO
import matplotlib as plt
import PIL
import skimage.io as io


coco_val = dset.CocoDetection(root = '/Users/adejuwon/Desktop/CMU School Junk/Fall 2020/10417 Intermediate Deep Learning/Course Project/COCO_DATASET/val2017',
                        annFile = '/Users/adejuwon/Desktop/CMU School Junk/Fall 2020/10417 Intermediate Deep Learning/Course Project/COCO_DATASET/annotations/instances_val2017.json',
                        transform=transforms.ToTensor())

print('Number of samples: ', len(coco_val))
img, target = coco_val[3] # load 4th sample
target=target[0]
target_id = target['category_id']

print("Image Size: ", img.size())
print(target)

coco = COCO(annotation_file="/Users/adejuwon/Desktop/CMU School Junk/Fall 2020/10417 Intermediate Deep Learning/Course Project/COCO_DATASET/annotations/instances_val2017.json")

target_mask = coco.annToMask(target)
cat_seg = coco.loadCats([target_id])

catIDs = coco.getCatIds()
cats = coco.loadCats(catIDs)

print(cats)



