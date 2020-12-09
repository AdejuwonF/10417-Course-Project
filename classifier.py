import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import torchvision.datasets as dset
from PIL import Image
import util
import time
from torch import optim
from datetime import datetime
from torchvision.utils import save_image
num_classes = 10
nz = 100
ngf = 32
ndf = 2
nc = 1
batch_size = 32
image_size = 32
workers = 0
ngpu=0
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

class Classifier(nn.Module):

    def __init__(self):
        super(Classifier, self).__init__()
        self.main = nn.Sequential(
            # input is (nc) x 32 x 32
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 16 x 16
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 8 x 8
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 4 x 4
            nn.AvgPool2d(4)
        )
        self.fc = nn.Sequential(
            nn.Linear(4*ndf, num_classes),
        )


    def forward(self, x):
        out = self.main(x)
        out = self.fc.forward(out.view(out.shape[0],-1))
        return out


def train(model, train_loader, val_loader, num_epochs):
    train_loss_log = []
    train_acc_log = []
    val_loss_log = []
    val_acc_log = []

    model.train()
    epochs = 0
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=.0001)
    for epoch in range(num_epochs):
        start = time.time()
        epochs += 1
        train_loss = 0
        train_acc = 0
        for i, data in enumerate(train_loader):
            inputs, labels = data
            logits = model.forward(inputs)

            loss = criterion(logits, labels)

            train_loss += loss.item()/len(train_loader)
            train_acc += torch.sum(torch.argmax(logits, dim=1) == labels).item()/len(labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 200 == 0:
                print("Batch {0}/{1}".format(i, len(train_loader)))
        with torch.no_grad():
            val_loss = 0
            val_acc = 0
            for i, data in enumerate(val_loader):
                inputs, labels = data
                logits = model.forward(inputs)

                loss = criterion(logits, labels)

                val_loss += loss.item()/len(val_loader)
                val_acc += torch.sum(torch.argmax(logits, dim=1) == labels).item() / len(labels)

        train_loss_log.append(train_loss)
        train_acc_log.append(train_acc/len(train_loader))
        val_loss_log.append(val_loss)
        val_acc_log.append(val_acc/len(val_loader))

        print("Epoch {0}/{1}\nTrain Loss: {2}\tTrain Accuracy: {3}\n"
              "Validation Loss: {4}\tValidation Accuracy: {5}".format(epoch+1, num_epochs, train_loss_log[-1], train_acc_log[-1]
                                                                      , val_loss_log[-1], val_acc_log[-1]))
        if (epoch + 1) % 1 == 0:
            torch.save({
                "model": model,
                "train_loss": train_loss_log,
                "train_acc": train_acc_log,
                "val_loss": val_loss_log,
                "val_acc": val_acc_log,
                "optim_state": optimizer.state_dict()
            }, "classifier_checkpoints/epoch{0}".format((epoch + 1)))


if __name__ == "__main__":
    dataset = dset.MNIST(root="", train=True, download=True,
                         transform=transforms.Compose([transforms.Resize(image_size), transforms.ToTensor()]))

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    lengths = [train_size, val_size]
    train_dataset, val_dataset = torch.utils.data.dataset.random_split(dataset, lengths,
                                                                      generator=torch.Generator().manual_seed(42))

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                             shuffle=True, num_workers=0)

    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size,
                                             shuffle=True, num_workers=0)

    model = Classifier()

    train(model, train_dataloader, val_dataloader, 50)







