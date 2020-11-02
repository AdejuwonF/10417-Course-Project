import torch
import torchvision as tv
import torch.functional as F
import torch.nn.functional
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torchvision.datasets.vision import VisionDataset
import pycocotools
from pycocotools.coco import COCO
import matplotlib as plt
import PIL
import skimage.io as io
import pickle as pk
from tqdm import tqdm
import os.path as osp
from torch.utils.data import DataLoader
import time
from PIL import Image
import os
import os.path

path = ""

# get the device we are running on
def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

# training method with tensorboard and shit. not implemented yet due to scuffedness
def fancy_train(model, dataset, epochs, batch_size, validation_dataset, optimizer, loss_func, logdir):
    device = get_device()
    writer = SummaryWriter(osp.join(logdir, "tb"))

    for epoch in range(epochs):
        model.train()
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        prog = tqdm(loader, f"Epoch {epoch+1}", unit="batch")
        epoch_loss = 0.0
        for i, batch in enumerate(prog):
            # batch = batch.to(device)
            ims, tgs = batch
            optimizer.zero_grad()
            loss = loss_func(model(batch), batch.y)
            prog.set_postfix_str(s=f"Loss: {loss:.5f}", refresh=True)
            loss.backward()
            optimizer.step()
            writer.add_scalar("Loss/Training", loss, epoch * len(prog) + i)
            epoch_loss += loss.item() * batch.num_graphs
        prog.set_postfix_str(s=f"Loss: {epoch_loss / len(dataset):.5f}", refresh=True)
        # evaluate(model, dataset, epoch * len(prog), batch_size, "Training")
        # evaluate(model, dataset, epoch * len(prog), batch_size, "Validation")

# scuffed training method
# returns list of train loss and list of validation loss
def train(model, dataset, epochs, batch_size, validation_dataset, optimizer, loss_func, layers=None):
    len_dataset = len(dataset)
    train_loss = []
    val_loss = []
    for epoch in range(epochs):
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        num_batches = len(loader)
        print(num_batches)
        epoch_loss = 0.0
        for i in range(num_batches):
            print(i)
            start_time = time.time()
            ims, tgs = next(iter(loader))

            if layers is not None:
                tgs = tgs[layers]
            else:
                tgs, indices = torch.max(tgs, dim=1)
                tgs = tgs.unsqueeze(1)

            optimizer.zero_grad()
            outs = model.forward(ims)
            loss = loss_func(outs, tgs)
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
            print("Batch took: {0}s".format(time.time()-start_time))
        print("epoch: " + str(epoch) + " train loss: " + str(epoch_loss / num_batches))
        train_loss.append(epoch_loss / num_batches)

        model.eval()
        val_loader = DataLoader(validation_dataset, shuffle=False)
        ims, tgs = next(iter(val_loader))
        outs = model.forward(ims)
        val_loss = loss_func(outs, tgs)
        print("epoch: " + str(epoch) + " val loss: " + str(val_loss))
        val_loss.append(val_loss)

    return train_loss, val_loss


# some function for evaluating. not yet implemented
def evaluate(model, dataset, step, batch_size, data_name):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    with torch.no_grad():
        truth_list = []
        pred_list = []
        for batch in loader:
            batch = batch.to(get_device())
            pred_list.append(model(batch))
            truth_list.append(batch.y)
        truth = torch.cat(truth_list)
        pred = torch.cat(pred_list)


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
        if len(annotations) == 0:
            height, width = 256, 256
        else:
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
        #print("image{0}:, annoations_size:{1}".format(image, len(annotations)))
        return self.input_transform(image), self.target_transform(annotations)


class CocoDetectionCatIds(VisionDataset):
    """`MS Coco Detection <http://mscoco.org/dataset/#detections-challenge2016>`_ Dataset.

    Args:
        root (string): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    """

    def __init__(self, root, annFile, catIds=None, transform=None, target_transform=None, transforms=None):
        super(CocoDetectionCatIds, self).__init__(root, transforms, transform, target_transform)
        from pycocotools.coco import COCO
        self.coco = COCO(annFile)
        self.catIds = catIds
        if catIds is None:
            self.ids = list(sorted(self.coco.imgs.keys()))
        else:
            self.ids = sorted(list(self.coco.getImgIds(catIds=catIds)))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: Tuple (image, target). target is the object returned by ``coco.loadAnns``.
        """
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)

        path = coco.loadImgs(img_id)[0]['file_name']

        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.ids)