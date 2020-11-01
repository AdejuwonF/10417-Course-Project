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

#path = "./"
path = ""

coco_val = dset.CocoDetection(root = path + 'COCO_DATASET/val2017',
                        annFile = path + 'COCO_DATASET/annotations/instances_val2017.json',
                        transform=transforms.ToTensor())

print('Number of samples: ', len(coco_val))
img, target = coco_val[3] # load 4th sample

print(target)
target=target[0]
target_id = target['category_id']

print("Image Size: ", img.size())
print(target)

coco = COCO(annotation_file=path + "COCO_DATASET/annotations/instances_val2017.json")

target_mask = coco.annToMask(target)
cat_seg = coco.loadCats([target_id])

catIDs = coco.getCatIds()
cats = coco.loadCats(catIDs)

print(cats)

annIds = coco.getAnnIds(imgIds=target['image_id'], catIds=catIDs, iscrowd=None)
anns = coco.loadAnns(annIds)
print(len(anns))
for i in range(len(anns)):
    #print(coco.annToMask(anns[i]))
    print(anns[i])
    print(coco.annToMask(anns[i]).shape)


for i in range(1):
    im, t = coco_val[i]

    plt.imshow(im.permute(1, 2, 0))
    plt.show()

    #annIds = t[1]
    #annIds = coco.getAnnIds(imgIds=t['image_id'], catIds=catIDs, iscrowd=None)
    #anns = coco.loadAnns(annIds)
    anns = t
    layer = 4
    mask = coco.annToMask(anns[layer])
    plt.imshow(mask)
    plt.show()


    im = im.unsqueeze(0)
    print(im.size())
    resized = torch.nn.functional.interpolate(im, size=256)
    resized = resized.squeeze(0)
    plt.imshow(resized.permute(1, 2, 0))
    plt.show()

    mask = torch.from_numpy(mask).unsqueeze(0)
    mask = mask.unsqueeze(0)
    mask = torch.nn.functional.interpolate(mask, size=256)
    mask = mask.squeeze(0)
    plt.imshow(mask.permute(1, 2, 0))
    plt.show()
    print(mask.shape)

    print(anns[1])

    print(t)

    m = torch.zeros((91, 256, 256))
    m[int(anns[1]['category_id']), :, :] = mask
    print(torch.sum(mask))
    print(torch.sum(m[int(anns[1]['category_id'])]))
    print(len(anns))
    for j in range(len(anns)):
        print(anns[j]['category_id'])




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
        if len(tensor.size()) < 4:
            tensor = tensor.unsqueeze(0)
        tensor_height, tensor_width = tensor.size()[2:]
        crop_height, crop_width = self.out_size
        crop_top = int(round((tensor_height - crop_height) / 2.))
        crop_left = int(round((tensor_width - crop_width) / 2.))
        return tensor[:, :, crop_top:tensor_height-crop_top, crop_left:tensor_width-crop_left]


for i in range(1):
    centerCrop = CenterCropTensor(256)
    im, t = coco_val[i]
    im = im.unsqueeze(0)
    print(im.size())
    cropped = centerCrop(im)
    cropped = cropped.squeeze(0)
    plt.imshow(cropped.permute(1, 2, 0))
    plt.show()

    anns = t
    layer = 4
    mask = coco.annToMask(anns[layer])
    mask = torch.from_numpy(mask).unsqueeze(0)
    mask = mask.unsqueeze(0)
    mask = centerCrop(mask)
    mask = mask.squeeze(0)
    plt.imshow(mask.permute(1, 2, 0))
    plt.show()
    print(mask.shape)





# Example of lazy processing some code stolen from utils
class TransformAnn(object):
    def __init__(self):
        pass

    def __call__(self, anns):
        """
        we create masks as a tensor with dimensions [91, size, size]. layer i represents the mask from the ith category
        WARNING: ALL MASKS OF THE SAME CATEGORY WILL BE COMBINED
        if the ith category is not present, then it will be all 0s
        for the categories that don't correspond to any object, their respective layers will also be 0
        """
        height, width = coco.annToMask(anns[0]).shape
        masks = torch.zeros((91, height, width))
        for j in range(len(anns)):
            mask = coco.annToMask(anns[j])
            masks[int(anns[j]['category_id']), :, :] += mask
        return masks

coco_val = dset.CocoDetection(root=path + 'COCO_DATASET/val2017',
                              annFile=path + 'COCO_DATASET/annotations/instances_val2017.json',
                              transform=transforms.Compose(
                                  [transforms.ToTensor(),
                                   CenterCropTensor(256)]),
                              target_transform=transforms.Compose(
                                  [TransformAnn(),
                                   CenterCropTensor(256)]))


for i in range(1):
    im, t = coco_val[i]
    print(im.size(), t.size())
    plt.imshow(im.squeeze().permute(1, 2, 0))
    plt.show()

    masks = t
    layer = 4
    # 62 is cat id for chair
    plt.imshow(masks[:, 62, :, :].permute(1, 2, 0))
    plt.show()

    combinedMasks, indices = torch.max(masks, dim=1)
    plt.imshow(combinedMasks.permute(1, 2, 0))
    plt.show()
    print(masks.shape)