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

path = ""
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



# Crops a tensor into a tensor of the desired size, centered about middle
# Probably can move to utils.py
# Pytorch's builtin crop only works on PIL images, which I don't think works for our masks? Not entirely sure
# Can use this as an alternative to downsampling, also can crop form different places for data augmentation
class CenterCropTensor(object):
    def __init__(self, size):
        if isinstance(size, int):
            self.out_size = (int(size), int(size))
        else:
            self.out_size = size

    def __call__(self, tensor):
        tensor_height, tensor_width = tensor.size()[1:]
        crop_height, crop_width = self.out_size
        crop_top = int(round((tensor_height - crop_height) / 2.))
        crop_left = int(round((tensor_width - crop_width) / 2.))
        crop_bot = tensor_height-crop_top - self.out_size[0]
        crop_right = tensor_width-crop_left - self.out_size[0]
        return tensor[:, crop_top:tensor_height-crop_bot, crop_left:tensor_width-crop_right]


#  Example of lazy processing some code stolen from utils
class TransformAnn(object):
    # Could modify to take in cat_ids, and thus only consider masks for certain ids i.e people
    def __init__(self, cat_ids=None):
        self.coco = COCO(annotation_file=path + "COCO_DATASET/annotations/instances_val2017.json")
        pass

    def __call__(self, annotations):
        """
        we create masks as a tensor with dimensions [91, size, size]. layer i represents the mask from the ith category
        WARNING: ALL MASKS OF THE SAME CATEGORY WILL BE COMBINED
        if the ith category is not present, then it will be all 0s
        for the categories that don't correspond to any object, their respective layers will also be 0
        """
        height, width = self.coco.annToMask(annotations[0]).shape
        masks = torch.zeros((91, height, width))
        for j in range(len(annotations)):
            mask = self.coco.annToMask(annotations[j])
            masks[int(annotations[j]['category_id']), :, :] += mask

        #  Could consider returning original annotation if we need that as well
        return masks


class transformCoCoPairs(object):
    def __init__(self, size, cat_ids=None):
        self.crop = CenterCropTensor(size)
        self.annTransform = TransformAnn(cat_ids)

        self.input_transform = transforms.Compose(
                                  [transforms.ToTensor(),
                                   self.crop])
        self.target_transform = transforms.Compose(
                                  [self.annTransform ,
                                   self.crop])

    def __call__(self, image, annotations):
        print("image{0}:, annoations_size:{1}".format(image, len(annotations)))
        return self.input_transform(image), self.target_transform(annotations)

