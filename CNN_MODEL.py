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
