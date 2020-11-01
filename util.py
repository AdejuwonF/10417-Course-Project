import torch
import torchvision as tv
import torch.functional as F
import torch.nn.functional
import torchvision.datasets as dset
import torchvision.transforms as transforms
import pycocotools
from pycocotools.coco import COCO
import matplotlib as plt
import PIL
import skimage.io as io
import pickle as pk

# pickle and return a list of images and masks from the given dataset and annotations
# file_prefix defines the prefix of the pickled files
# size is the size to reshape all the images to
# num is the number of images/masks to process from the dataset. -1 processes all of them
def preprocess(dataset, annotations, file_prefix, size=256, num=-1):
    coco_val = dset.CocoDetection(root=dataset,
                                  annFile=annotations,
                                  transform=transforms.ToTensor())
    coco = COCO(annotation_file=annotations)
    pk_images = []
    pk_masks = []

    num_to_process = num

    if num_to_process == -1:
        num_to_process = len(coco_val)

    for i in range(num_to_process):
        img, anns = coco_val[i]

        # reshape img to be a tensor with dimensions [3, size, size] and add it to the image list
        img = img.unsqueeze(0)
        img = torch.nn.functional.interpolate(img, size=size)
        img = img.squeeze(0)
        pk_images.append(img)

        # we create masks as a tensor with dimensions [91, size, size]. layer i represents the mask from the ith category
        # WARNING: ALL MASKS OF THE SAME CATEGORY WILL BE COMBINED
        # if the ith category is not present, then it will be all 0s
        # for the categories that don't correspond to any object, their respective layers will also be 0
        masks = torch.zeros((91, size, size))
        for j in range(len(anns)):
            mask = coco.annToMask(anns[j])
            mask = torch.from_numpy(mask).unsqueeze(0)
            mask = mask.unsqueeze(0)
            mask = torch.nn.functional.interpolate(mask, size=size)
            mask = mask.squeeze(0)
            masks[int(anns[j]['category_id']), :, :] += mask.squeeze(0)
        pk_masks.append(masks)

    with open(file_prefix + "_images.pk", 'wb') as file:
        pk.dump(pk_images, file)
    with open(file_prefix + "_masks.pk", "wb") as file:
        pk.dump(pk_masks, file)

    return pk_images, pk_masks



