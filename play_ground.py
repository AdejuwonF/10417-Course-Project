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

from util import CenterCropTensor, TransformAnn, transformCoCoPairs, transformCoCoPairsResize

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
    resized = torch.nn.functional.interpolate(im, size=256, mode="bicubic")
    resized = resized.squeeze(0)
    plt.imshow(resized.permute(1, 2, 0))
    plt.show()

    mask = torch.from_numpy(mask).unsqueeze(0)
    mask = mask.unsqueeze(0)
    mask = mask.type(torch.DoubleTensor)
    mask = torch.nn.functional.interpolate(mask, size=256, mode="bicubic")
    mask = mask > .1
    mask = mask.squeeze(0)
    plt.imshow(mask.permute(1, 2, 0))
    plt.show()
    print(mask.shape, mask.type)

    print(anns[1])

    print(t)

    m = torch.zeros((91, 256, 256))
    m[int(anns[1]['category_id']), :, :] = mask
    print(torch.sum(mask))
    print(torch.sum(m[int(anns[1]['category_id'])]))
    print(len(anns))
    for j in range(len(anns)):
        print(anns[j]['category_id'])



for i in range(1):
    centerCrop = CenterCropTensor(256)
    im, t = coco_val[i]
    print(im.size())
    cropped = centerCrop(im)
    plt.imshow(cropped.permute(1, 2, 0))
    plt.show()

    anns = t
    layer = 4
    mask = coco.annToMask(anns[layer])
    mask = torch.from_numpy(mask).unsqueeze(0)
    mask = centerCrop(mask)
    plt.imshow(mask.permute(1, 2, 0))
    plt.show()
    print(mask.shape)


# loads the dataset, but this time automatically applies the given transforms when retrieving data
# I believe things are returned as (N, C, H, W) even if we only get one item
coco_val = dset.CocoDetection(root=path + 'COCO_DATASET/val2017',
                              annFile=path + 'COCO_DATASET/annotations/instances_val2017.json',
                              transforms=transformCoCoPairs(256))
# instead of using transformCoCoPairs we can add this I believe
"""transform=transforms.Compose(
  [transforms.ToTensor(),
   CenterCropTensor(256)]),
target_transform=transforms.Compose(
  [TransformAnn(),
   CenterCropTensor(256)]))"""



for i in range(1):
    im, t = coco_val[i]
    print(im.size(), t.size())
    plt.imshow(im.permute(1, 2, 0))
    plt.show()

    masks = t
    layer = 4
    # 62 is cat id for chair
    plt.imshow(masks[62, :, :].unsqueeze(0).permute(1, 2, 0))
    plt.show()

    #  Puts all the masks into one channel
    combinedMasks, indices = torch.max(masks, dim=0)
    plt.imshow(combinedMasks.unsqueeze(0).permute(1, 2, 0))
    plt.show()
    print(masks.shape)

    #  Need dataloader to get multiple things easily.  Datasets don't seem to like slicing
    dataloader = DataLoader(coco_val, batch_size=4, shuffle=True, num_workers=0)
    ims, tgs = next(iter(dataloader))
    print(ims.size(), tgs.size())

    print(len(coco_val))
    print(len(dataloader))
    for i in range(ims.size()[0]):
        img = ims[i,:,:,:]
        tg = tgs[i,:,:,:]

        plt.imshow(img.permute(1, 2, 0))
        plt.show()
        combinedMasks, indices = torch.max(tg, dim=0)
        plt.imshow(combinedMasks.unsqueeze(0).permute(1, 2, 0))
        plt.show()

coco_val = dset.CocoDetection(root=path + 'COCO_DATASET/val2017',
                              annFile=path + 'COCO_DATASET/annotations/instances_val2017.json',
                              transforms=transformCoCoPairsResize(64))
for i in range(1):
    im, t = coco_val[i]
    print(im.size(), t.size())
    plt.imshow(im.permute(1, 2, 0))
    plt.show()

    masks = t
    layer = 4
    # 62 is cat id for chair
    plt.imshow(masks[62, :, :].unsqueeze(0).permute(1, 2, 0))
    plt.show()

    #  Puts all the masks into one channel
    combinedMasks, indices = torch.max(masks, dim=0)
    plt.imshow(combinedMasks.unsqueeze(0).permute(1, 2, 0))
    plt.show()
    print(masks.shape)

    #  Need dataloader to get multiple things easily.  Datasets don't seem to like slicing
    dataloader = DataLoader(coco_val, batch_size=4, shuffle=True, num_workers=0)
    ims, tgs = next(iter(dataloader))
    print(ims.size(), tgs.size())

    print(len(coco_val))
    print(len(dataloader))
    for i in range(ims.size()[0]):
        img = ims[i,:,:,:]
        tg = tgs[i,:,:,:]

        plt.imshow(img.permute(1, 2, 0))
        plt.show()
        combinedMasks, indices = torch.max(tg, dim=0)
        plt.imshow(combinedMasks.unsqueeze(0).permute(1, 2, 0))
        plt.show()
