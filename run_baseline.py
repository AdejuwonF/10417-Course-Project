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
from util import CenterCropTensor, TransformAnn, transformCoCoPairs, CocoDetectionCatIds, transformCoCoPairsResize
import pickle as pk



if __name__ == "__main__":
    path = ""
    net = DeconvNet()
    optimizer = optim.Adam(net.parameters(), lr=0.01)
    loss_func = nn.BCELoss()

    # annFile_train = 'COCO_DATASET/annotations/instances_train2017.json'
    # coco_train = COCO(annFile_train)
    # filter_classes = "person"
    # catIds_train = coco_train.getCatIds(catNms=filter_classes)
    # imgIds_train = coco_train.getImgIds(catIds=catIds_train)
    # print("Number of validation images containing all the classes:", len(imgIds_train))
    # print(imgIds_train)
    #
    # coco_train_people = CocoDetectionCatIds(root=path + 'COCO_DATASET/train2017',
    #                                       annFile=path + 'COCO_DATASET/annotations/instances_train2017.json',
    #                                       catIds=catIds_train,
    #                                       transforms=transformCoCoPairsResize(128))


    annFile_val = 'COCO_DATASET/annotations/instances_val2017.json'
    coco_val = COCO(annFile_val)
    filter_classes = "person"
    catIds_val = coco_val.getCatIds(catNms=filter_classes)
    imgIds_val = coco_val.getImgIds(catIds=catIds_val)
    print("Number of validation images containing all the classes:", len(imgIds_val))
    print(imgIds_val)

    coco_val_people = CocoDetectionCatIds(root=path + 'COCO_DATASET/val2017',
                                          annFile=path + 'COCO_DATASET/annotations/instances_val2017.json',
                                          catIds=catIds_val,
                                          transforms=transformCoCoPairsResize(128))
    train_size = int(0.8 * len(coco_val_people))
    test_size = len(coco_val_people) - train_size
    lengths = [train_size, test_size]
    train_dataset, valid_dataset = torch.utils.data.dataset.random_split(coco_val_people, lengths)

    print(len(train_dataset), len(valid_dataset))

    train_loss, val_loss = util.train(net, train_dataset, 1, 50, valid_dataset, optimizer, loss_func, layers=[1])

    with open("./baseline_train_loss.pk", "wb") as F:
        pk.dump(train_loss, F)
        F.close()

    with open("./baseline_valid_loss.pk", "wb") as F:
        pk.dump(val_loss, F)
        F.close()

    torch.save(net, "./baseline_model_epoch_1")
    loaded_net = torch.load("./baseline_model_epoch_100")

    torch.save({
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, "./baseline_model_epoch_1_checkpoint")
    loaded_dict = torch.load("./baseline_model_epoch_100_checkpoint")
    loaded_net.load_state_dict(loaded_dict["model_state_dict"])

