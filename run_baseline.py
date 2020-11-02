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
import pickle as pk



if __name__ == "__main__":
    path = ""
    net = DeconvNet()
    optimizer = optim.Adam(net.parameters(), lr=0.01)
    loss_func = nn.BCELoss()

    annFile = 'COCO_DATASET/annotations/instances_val2017.json'
    coco = COCO(annFile)
    filter_classes = "person"
    catIds = coco.getCatIds(catNms=filter_classes)
    imgIds = coco.getImgIds(catIds=catIds)
    print("Number of images containing all the classes:", len(imgIds))
    print(imgIds)

    coco_val_people = CocoDetectionCatIds(root=path + 'COCO_DATASET/val2017',
                                          annFile=path + 'COCO_DATASET/annotations/instances_val2017.json',
                                          catIds=catIds,
                                          transforms=transformCoCoPairs(128))
    train_size = int(0.8 * len(coco_val_people))
    test_size = len(coco_val_people) - train_size
    lengths = [train_size, test_size]
    train_dataset, valid_dataset = torch.utils.data.dataset.random_split(coco_val_people, lengths)

    print(len(train_dataset), len(valid_dataset))

    train_loss, val_loss = util.train(net, train_dataset, 2, 50, valid_dataset, optimizer, loss_func, layers=[1])

    with open("./baseline_train_loss.pk", "wb") as F:
        pk.dump(train_loss, F)
        F.close()

    with open("./baseline_valid_loss.pk", "wb") as F:
        pk.dump(val_loss, F)
        F.close()

    with torch.no_grad():
        img, _ = coco_val_people[3]
        pred_mask = net.forward(img.unsqueeze(0))
        plt.imshow(pred_mask.squeeze(0).permute(1, 2, 0))
        plt.show()

    torch.save(net.state_dict(), "./baseline_model_epoch_100")
    torch.save({
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, "./baseline_model_epoch_100")
    loaded_dict = torch.load("./baseline_model_epoch_100")
    loaded_net = DeconvNet()
    loaded_net.load_state_dict(loaded_dict["model_state_dict"])

    with torch.no_grad():
        img, _ = coco_val_people[3]
        pred_mask = loaded_net.forward(img.unsqueeze(0))
        plt.imshow(pred_mask.squeeze(0).permute(1, 2, 0))
        plt.show()
