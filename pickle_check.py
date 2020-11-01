import pickle as pk
import torch
import torchvision as tv
import torch.functional as F
import torch.nn.functional
import torchvision.datasets as dset
import torchvision.transforms as transforms
import pycocotools
from pycocotools.coco import COCO
import matplotlib.pyplot as plt
import PIL
import skimage.io as io

with open("val_images.pk", "rb") as f:
    images = pk.load(f)
with open("val_masks.pk", "rb") as f:
    masks = pk.load(f)

i = 0

img = images[i]
mask = masks[i]

plt.imshow(img.permute(1, 2, 0))
plt.show()

for layer in range(91):
    if torch.sum(mask[layer]).item() > 0:
        print(layer)
        plt.imshow(mask[layer])
        plt.show()