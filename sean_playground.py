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
import DCGAN
import WGAN_WC
import CAGenerator
from torchvision.utils import save_image
import os
import subprocess
import glob


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

# model = DCGAN.DCGAN()
# model.load("MNIST_playground_output/dcgan_epochs2020_11_28_06:33:53")
# i = 0
# for img in (model.img_list):
#     if i % 38 == 0 or i == 379:
#         save_image(img, "MNIST_playground_output/epoch_" + str(i) + ".png")
#     i += 1

CAG = CAGenerator.CAGenerator()
model = WGAN_WC.WGAN(modelG=CAG)
model.load("CAGAN7/gan_wc_7_epochs2020_12_02_11:03:47")
i = 0
for img in model.img_list:
    save_image(img, "CAGAN7/img_" + str(i) + ".png")
    i += 1

print(model.G_losses)
print(model.D_losses)

def generate_video(predictions, image, video_name="video"):
    folder = "video"
    for i in range(len(predictions)):
        #plt.imshow(predictions[i][:, 0:1, :, :].squeeze(0).permute([1,2,0]), cmap="Greys")
        #plt.savefig(folder + "/file%02d.png" % i)
        save_image(predictions[i][:, 0:1, :, :].squeeze(0), folder + "/file%02d.png" % i)

    os.chdir(folder)
    subprocess.call([
        './ffmpeg', '-framerate', '8', '-i', 'file%02d.png', '-r', '30', '-pix_fmt', 'yuv420p',
        video_name + ".mp4"
    ])
    for file_name in glob.glob("*.png"):
        os.remove(file_name)

# with torch.no_grad():
#     state = model.generator.forward(torch.randn((1, 100, 1, 1)), [0, 0], True)
#     outs = [None] * 65
#     outs[0] = state
#     for i in range(1, 65):
#         state = model.generator.ca_step(state)
#         outs[i] = state
#         #print(state[:, 0:1, :, :].squeeze(0))
#         print(i, torch.sum(state[:, 0:1, :, :].squeeze(0)))
#     #generate_video(outs, None)